"""ElevenLabs Conversational AI handler for fastrtc Stream.

This module provides integration between ElevenLabs Conversational AI SDK (v2.x)
and fastrtc for real-time audio streaming in the Reachy Mini conversation app.
"""

import json
import base64
import asyncio
import logging
from queue import Queue, Empty
from typing import Any, Final, Tuple, Literal, Optional, Callable
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import gradio as gr
from elevenlabs import ElevenLabs
from elevenlabs.conversational_ai.conversation import Conversation, AudioInterface, ClientTools
from fastrtc import AdditionalOutputs, AsyncStreamHandler, wait_for_item, audio_to_int16
from numpy.typing import NDArray
from scipy.signal import resample

from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.prompts import get_session_voice, get_session_instructions
from reachy_mini_conversation_app.tools.core_tools import (
    ToolDependencies,
    get_tool_specs,
    dispatch_tool_call,
)


logger = logging.getLogger(__name__)

# ElevenLabs audio format - 16kHz 16-bit PCM mono
ELEVENLABS_SAMPLE_RATE: Final[Literal[16000]] = 16000


class FastRTCAudioInterface(AudioInterface):
    """Custom AudioInterface that bridges ElevenLabs SDK with fastrtc.

    This interface:
    - Stores the input_callback from start() to be called when receiving audio
    - Queues output audio for the async handler to emit
    - Handles interrupts by clearing the output queue
    """

    def __init__(
        self,
        output_queue: Queue,
        on_interrupt: Optional[Callable[[], None]] = None,
        on_user_speaking: Optional[Callable[[bool], None]] = None,
    ):
        """Initialize the audio interface.

        Args:
            output_queue: Thread-safe queue for output audio chunks
            on_interrupt: Callback when audio should be interrupted
            on_user_speaking: Callback for user speaking state changes
        """
        self._output_queue = output_queue
        self._input_callback: Optional[Callable[[bytes], None]] = None
        self._on_interrupt = on_interrupt
        self._on_user_speaking = on_user_speaking
        self._is_speaking = False
        self._speech_timeout_s = 0.3
        self._last_audio_time = 0.0
        self._running = False

    def start(self, input_callback: Callable[[bytes], None]) -> None:
        """Start the audio interface.

        Args:
            input_callback: Callback to send audio input to ElevenLabs (16kHz PCM16 mono)
        """
        self._input_callback = input_callback
        self._running = True
        logger.info("FastRTCAudioInterface started")

    def stop(self) -> None:
        """Stop the audio interface and clean up resources."""
        self._running = False
        self._input_callback = None
        # Clear output queue
        while not self._output_queue.empty():
            try:
                self._output_queue.get_nowait()
            except Empty:
                break
        logger.info("FastRTCAudioInterface stopped")

    def output(self, audio: bytes) -> None:
        """Queue audio output for playback.

        Args:
            audio: Audio data in 16-bit PCM mono format at 16kHz
        """
        if self._running:
            self._output_queue.put(audio)

    def interrupt(self) -> None:
        """Handle interruption - clear queued audio."""
        # Clear the output queue
        while not self._output_queue.empty():
            try:
                self._output_queue.get_nowait()
            except Empty:
                break
        if self._on_interrupt:
            self._on_interrupt()
        logger.debug("Audio interrupted")

    def send_audio_input(self, audio_bytes: bytes) -> None:
        """Send audio input to ElevenLabs.

        Called by the handler when receiving audio from the microphone.

        Args:
            audio_bytes: Audio data in 16-bit PCM mono format at 16kHz
        """
        if self._input_callback and self._running:
            self._input_callback(audio_bytes)

            # Track user speaking state
            import time
            now = time.monotonic()
            if not self._is_speaking:
                self._is_speaking = True
                if self._on_user_speaking:
                    self._on_user_speaking(True)
            self._last_audio_time = now


class ElevenLabsRealtimeHandler(AsyncStreamHandler):
    """An ElevenLabs Conversational AI handler for fastrtc Stream.

    Uses the new ElevenLabs SDK v2.x Conversation API with custom AudioInterface.
    """

    def __init__(self, deps: ToolDependencies, gradio_mode: bool = False, instance_path: Optional[str] = None):
        """Initialize the handler."""
        super().__init__(
            expected_layout="mono",
            output_sample_rate=ELEVENLABS_SAMPLE_RATE,
            input_sample_rate=ELEVENLABS_SAMPLE_RATE,
        )

        # Override typing of the sample rates
        self.output_sample_rate: Literal[16000] = self.output_sample_rate
        self.input_sample_rate: Literal[16000] = self.input_sample_rate

        self.deps = deps
        self.client: Optional[ElevenLabs] = None
        self.conversation: Optional[Conversation] = None

        # Thread-safe queue for output audio (sync Queue, not asyncio.Queue)
        self._audio_output_queue: Queue = Queue()

        # Async queue for UI outputs (transcripts, tool results, etc.)
        self.output_queue: "asyncio.Queue[Tuple[int, NDArray[np.int16]] | AdditionalOutputs]" = asyncio.Queue()

        self.last_activity_time = 0.0
        self.start_time = 0.0
        self.is_idle_tool_call = False
        self.gradio_mode = gradio_mode
        self.instance_path = instance_path

        # Track how the API key was provided
        self._key_source: Literal["env", "textbox"] = "env"
        self._provided_api_key: str | None = None

        # Debouncing for partial transcripts
        self.partial_transcript_task: asyncio.Task[None] | None = None
        self.partial_transcript_sequence: int = 0
        self.partial_debounce_delay = 0.5  # seconds

        # Internal lifecycle flags
        self._shutdown_requested: bool = False
        self._connected_event: asyncio.Event = asyncio.Event()

        # Audio interface for bridging
        self._audio_interface: Optional[FastRTCAudioInterface] = None

        # Event loop reference for cross-thread callbacks
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Client tools for function calling
        self._client_tools: Optional[ClientTools] = None

    def copy(self) -> "ElevenLabsRealtimeHandler":
        """Create a copy of the handler."""
        return ElevenLabsRealtimeHandler(self.deps, self.gradio_mode, self.instance_path)

    async def apply_personality(self, profile: str | None) -> str:
        """Apply a new personality (profile) at runtime if possible.

        Updates the global config's selected profile and reinitializes the conversation
        with fresh instructions.

        Returns a short status message for UI feedback.
        """
        try:
            from reachy_mini_conversation_app.config import set_custom_profile

            set_custom_profile(profile)
            logger.info("Set custom profile to %r", profile)

            try:
                instructions = get_session_instructions()
                logger.info("Loaded instructions for profile %r (length=%d)", profile, len(instructions))
            except Exception as e:
                return f"Failed to load profile: {e}"

            # Restart the conversation with new instructions
            if self.conversation and not self._shutdown_requested:
                try:
                    await self._restart_conversation()
                    return "Applied personality and restarted conversation."
                except Exception as e:
                    logger.warning("Restart failed: %s", e)
                    return f"Profile updated but restart failed: {e}"

            return "Profile updated (will apply on next connection)."

        except Exception as e:
            logger.exception("apply_personality failed")
            return f"Failed to apply personality: {e}"

    async def _restart_conversation(self) -> None:
        """Restart the conversation session."""
        try:
            if self.conversation:
                self.conversation.end_session()
                self.conversation.wait_for_session_end()
            await asyncio.sleep(0.5)
            self._connected_event.clear()
            await self._run_conversation_session()
            logger.info("Conversation session restarted.")
        except Exception as e:
            logger.warning("_restart_conversation failed: %s", e)

    def _schedule_coroutine(self, coro: Any) -> None:
        """Schedule a coroutine to run on the main event loop from a callback thread."""
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(coro, self._loop)

    async def _emit_debounced_partial(self, text: str, sequence: int) -> None:
        """Emit partial transcript after debounce delay if sequence still matches."""
        await asyncio.sleep(self.partial_debounce_delay)
        if self.partial_transcript_sequence == sequence and not self._shutdown_requested:
            await self.output_queue.put(
                AdditionalOutputs({"role": "user", "content": text, "metadata": {"partial": True}})
            )

    def set_api_key(self, key: str, source: Literal["env", "textbox"] = "textbox") -> None:
        """Store API key and track its source for persistence logic."""
        self._provided_api_key = key.strip() if key else None
        self._key_source = source

    async def start_up(self) -> None:
        """Start the handler, resolving API key from env or Gradio textbox."""
        self._loop = asyncio.get_event_loop()
        self.start_time = self._loop.time()
        self.last_activity_time = self.start_time

        elevenlabs_api_key = getattr(config, "ELEVENLABS_API_KEY", None)
        if self.gradio_mode and not elevenlabs_api_key:
            # API key was not found in .env or environment variables
            try:
                await self.wait_for_args()  # type: ignore[no-untyped-call]
                args = list(self.latest_args)
                # Keep index consistent with existing Stream wiring (textbox typically at position 3)
                textbox_api_key = args[3] if len(args) > 3 and len(args[3]) > 0 else None
            except Exception:
                textbox_api_key = None
            if textbox_api_key is not None:
                elevenlabs_api_key = textbox_api_key
                self._key_source = "textbox"
                self._provided_api_key = textbox_api_key
            else:
                elevenlabs_api_key = getattr(config, "ELEVENLABS_API_KEY", None)
        else:
            if not elevenlabs_api_key or not str(elevenlabs_api_key).strip():
                logger.warning("ELEVENLABS_API_KEY missing. Proceeding with placeholder (tests/offline).")
                elevenlabs_api_key = "DUMMY"

        await self.start(elevenlabs_api_key)

    async def start(self, elevenlabs_api_key: str | None = None) -> None:
        """Start the handler with the ElevenLabs conversation."""
        if self._shutdown_requested:
            logger.info("Shutdown already requested; skipping start.")
            return

        # Resolve API key
        if elevenlabs_api_key:
            self.set_api_key(elevenlabs_api_key, source="textbox" if self.gradio_mode else "env")
        else:
            key_from_env = config.ELEVENLABS_API_KEY
            if key_from_env:
                self.set_api_key(key_from_env, source="env")
            elevenlabs_api_key = key_from_env

        if not elevenlabs_api_key:
            logger.error("No ElevenLabs API key provided")
            raise ValueError("ELEVENLABS_API_KEY is required")

        # Initialize ElevenLabs client
        self.client = ElevenLabs(api_key=elevenlabs_api_key)

        # Start the conversation session
        await self._run_conversation_session()

    def _on_interrupt_callback(self) -> None:
        """Handle interrupt from ElevenLabs (user started speaking)."""
        if self.deps.head_wobbler is not None:
            self.deps.head_wobbler.reset()

    def _on_user_speaking_callback(self, is_speaking: bool) -> None:
        """Handle user speaking state changes."""
        self.deps.movement_manager.set_listening(is_speaking)

    def _on_agent_response(self, response: str) -> None:
        """Callback for agent text responses."""
        logger.debug(f"Agent response: {response}")

        async def emit_response() -> None:
            await self.output_queue.put(
                AdditionalOutputs({"role": "assistant", "content": response})
            )

        self._schedule_coroutine(emit_response())

    def _on_user_transcript(self, transcript: str) -> None:
        """Callback for user speech transcription."""
        logger.debug(f"User transcript: {transcript}")
        # Stop listening mode when we get a transcript
        self.deps.movement_manager.set_listening(False)

        async def emit_transcript() -> None:
            await self.output_queue.put(
                AdditionalOutputs({"role": "user", "content": transcript})
            )

        self._schedule_coroutine(emit_transcript())

    def _setup_client_tools(self) -> ClientTools:
        """Set up client tools for function calling."""
        client_tools = ClientTools()

        # Get tool specs and register each tool
        tool_specs = get_tool_specs()
        for spec in tool_specs:
            tool_name = spec.get("name")
            if not tool_name:
                continue

            # Create a closure to capture tool_name
            def make_tool_handler(name: str) -> Callable:
                async def tool_handler(parameters: dict) -> str:
                    try:
                        args_json = json.dumps(parameters)
                        result = await dispatch_tool_call(name, args_json, self.deps)
                        logger.debug(f"Tool '{name}' result: {result}")

                        # Emit tool usage to UI
                        await self.output_queue.put(
                            AdditionalOutputs({
                                "role": "assistant",
                                "content": json.dumps(result),
                                "metadata": {"title": f"🛠️ Used tool {name}", "status": "done"},
                            })
                        )

                        # Handle camera tool image display
                        if name == "camera" and "b64_im" in result:
                            if self.deps.camera_worker is not None:
                                np_img = self.deps.camera_worker.get_latest_frame()
                                if np_img is not None:
                                    rgb_frame = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
                                else:
                                    rgb_frame = None
                                img = gr.Image(value=rgb_frame)
                                await self.output_queue.put(
                                    AdditionalOutputs({"role": "assistant", "content": img})
                                )

                        # Reset head wobbler after tool call
                        if self.deps.head_wobbler is not None:
                            self.deps.head_wobbler.reset()

                        return json.dumps(result)
                    except Exception as e:
                        logger.error(f"Tool '{name}' failed: {e}")
                        return json.dumps({"error": str(e)})

                return tool_handler

            client_tools.register(tool_name, make_tool_handler(tool_name), is_async=True)
            logger.debug(f"Registered tool: {tool_name}")

        return client_tools

    async def _run_conversation_session(self) -> None:
        """Establish and manage a conversation session using the new SDK API."""
        if not self.client:
            logger.error("ElevenLabs client not initialized")
            return

        try:
            # Get session configuration
            voice = get_session_voice()

            # Create audio interface
            self._audio_interface = FastRTCAudioInterface(
                output_queue=self._audio_output_queue,
                on_interrupt=self._on_interrupt_callback,
                on_user_speaking=self._on_user_speaking_callback,
            )

            # Setup client tools
            self._client_tools = self._setup_client_tools()

            # Get agent ID - required for the new SDK
            agent_id = config.ELEVENLABS_AGENT_ID
            if not agent_id:
                logger.error("ELEVENLABS_AGENT_ID is required for the new SDK. "
                           "Please create an agent at https://elevenlabs.io/agents and set ELEVENLABS_AGENT_ID in .env")
                raise ValueError("ELEVENLABS_AGENT_ID is required")

            logger.info("Creating conversation with agent_id=%s, voice=%s", agent_id, voice)

            # Create conversation with new SDK API
            self.conversation = Conversation(
                client=self.client,
                agent_id=agent_id,
                requires_auth=True,
                audio_interface=self._audio_interface,
                client_tools=self._client_tools,
                callback_agent_response=self._on_agent_response,
                callback_user_transcript=self._on_user_transcript,
            )

            # Persist API key if needed
            self._persist_api_key_if_needed()

            logger.info("Starting ElevenLabs conversation session with profile=%r voice=%r",
                       config.REACHY_MINI_CUSTOM_PROFILE, voice)

            # Start the conversation (runs in background thread)
            self.conversation.start_session()

            # Mark as connected
            self._connected_event.set()

            logger.info("ElevenLabs conversation session started")

            # Start background task to process audio output
            asyncio.create_task(self._process_audio_output())

        except Exception as e:
            logger.exception("Conversation session failed: %s", e)

    async def _process_audio_output(self) -> None:
        """Process audio output from the sync queue to async queue."""
        while not self._shutdown_requested:
            try:
                # Check for audio in the sync queue (non-blocking)
                try:
                    audio_chunk = self._audio_output_queue.get_nowait()
                except Empty:
                    await asyncio.sleep(0.01)  # Small sleep to avoid busy loop
                    continue

                # Convert to numpy array for fastrtc
                audio_array = np.frombuffer(audio_chunk, dtype=np.int16).reshape(1, -1)

                # Feed to head wobbler if available
                if self.deps.head_wobbler is not None:
                    audio_b64 = base64.b64encode(audio_chunk).decode("utf-8")
                    self.deps.head_wobbler.feed(audio_b64)

                if self._loop:
                    self.last_activity_time = self._loop.time()

                # Queue audio for playback
                await self.output_queue.put((self.output_sample_rate, audio_array))

            except Exception as e:
                if not self._shutdown_requested:
                    logger.error("Audio output processing error: %s", e)
                    await asyncio.sleep(0.1)

    async def receive(self, frame: Tuple[int, NDArray[np.int16]]) -> None:
        """Receive audio frame from the microphone and send it to ElevenLabs.

        Args:
            frame: A tuple containing (sample_rate, audio_data).
        """
        if not self._audio_interface:
            return

        input_sample_rate, audio_frame = frame

        # Reshape if needed
        if audio_frame.ndim == 2:
            if audio_frame.shape[1] > audio_frame.shape[0]:
                audio_frame = audio_frame.T
            if audio_frame.shape[1] > 1:
                audio_frame = audio_frame[:, 0]

        # Flatten if needed
        audio_frame = audio_frame.flatten()

        # Resample if needed
        if self.input_sample_rate != input_sample_rate:
            audio_frame = resample(audio_frame, int(len(audio_frame) * self.input_sample_rate / input_sample_rate))

        # Cast to int16
        audio_frame = audio_to_int16(audio_frame)

        # Send to ElevenLabs via the audio interface
        try:
            audio_bytes = audio_frame.tobytes()
            self._audio_interface.send_audio_input(audio_bytes)
        except Exception as e:
            logger.debug("Dropping audio frame: connection not ready (%s)", e)

    async def emit(self) -> Tuple[int, NDArray[np.int16]] | AdditionalOutputs | None:
        """Emit audio frame to be played by the speaker."""
        # Handle idle detection
        if self._loop:
            idle_duration = self._loop.time() - self.last_activity_time
            if idle_duration > 15.0 and self.deps.movement_manager.is_idle():
                try:
                    await self.send_idle_signal(idle_duration)
                except Exception as e:
                    logger.warning("Idle signal skipped: %s", e)
                    return None
                self.last_activity_time = self._loop.time()

        return await wait_for_item(self.output_queue)  # type: ignore[no-any-return]

    async def shutdown(self) -> None:
        """Shutdown the handler."""
        self._shutdown_requested = True

        # Cancel debounce task
        if self.partial_transcript_task and not self.partial_transcript_task.done():
            self.partial_transcript_task.cancel()

        # End conversation session
        if self.conversation:
            try:
                self.conversation.end_session()
                # Wait briefly for session to end
                self.conversation.wait_for_session_end()
            except Exception as e:
                logger.debug(f"Conversation end_session error: {e}")
            finally:
                self.conversation = None

        # Stop audio interface
        if self._audio_interface:
            self._audio_interface.stop()
            self._audio_interface = None

        # Clear output queues
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    def format_timestamp(self) -> str:
        """Format current timestamp with date, time, and elapsed seconds."""
        if self._loop:
            loop_time = self._loop.time()
            elapsed_seconds = loop_time - self.start_time
        else:
            elapsed_seconds = 0.0
        dt = datetime.now()
        return f"[{dt.strftime('%Y-%m-%d %H:%M:%S')} | +{elapsed_seconds:.1f}s]"

    async def get_available_voices(self) -> list[str]:
        """Get available voices from ElevenLabs API."""
        if not self.client:
            # Return common ElevenLabs voices
            return [
                "rachel", "clyde", "domi", "dave", "fin", "sarah",
                "antoni", "thomas", "charlie", "emily", "elli",
                "callum", "charlotte", "matilda", "alice"
            ]

        try:
            voices = self.client.voices.get_all()
            return [voice.name.lower() for voice in voices.voices]
        except Exception as e:
            logger.warning(f"Failed to fetch voices: {e}")
            return ["rachel", "clyde", "domi", "dave", "fin", "sarah"]

    async def send_idle_signal(self, idle_duration: float) -> None:
        """Send an idle signal to trigger idle behavior.

        Note: The new SDK may not support sending text directly.
        This is kept for compatibility but may need adjustment.
        """
        logger.debug("Idle detected for %.1fs - the agent may initiate conversation", idle_duration)
        self.is_idle_tool_call = True
        # Note: The new SDK doesn't have a direct send_text method
        # The agent may have its own idle behavior configured

    def _persist_api_key_if_needed(self) -> None:
        """Persist the API key into `.env` when appropriate.

        Only runs in Gradio mode when key came from the textbox and is non-empty.
        """
        try:
            if not self.gradio_mode:
                return

            if self._key_source != "textbox":
                logger.info("API key not provided via textbox; skipping persistence.")
                return

            key = (self._provided_api_key or "").strip()
            if not key:
                logger.warning("No API key provided via textbox; skipping persistence.")
                return

            if self.instance_path is None:
                logger.warning("Instance path is None; cannot persist API key.")
                return

            # Update environment variable
            import os
            os.environ["ELEVENLABS_API_KEY"] = key

            target_dir = Path(self.instance_path)
            env_path = target_dir / ".env"

            if env_path.exists():
                logger.info(".env already exists; not overwriting.")
                return

            example_path = target_dir / ".env.example"
            content_lines: list[str] = []

            if example_path.exists():
                try:
                    content = example_path.read_text(encoding="utf-8")
                    content_lines = content.splitlines()
                except Exception as e:
                    logger.warning(f"Failed to read .env.example: {e}")

            # Replace or append ELEVENLABS_API_KEY
            replaced = False
            for i, line in enumerate(content_lines):
                if line.strip().startswith("ELEVENLABS_API_KEY="):
                    content_lines[i] = f"ELEVENLABS_API_KEY={key}"
                    replaced = True
                    break

            if not replaced:
                content_lines.append(f"ELEVENLABS_API_KEY={key}")

            final_text = "\n".join(content_lines) + "\n"
            env_path.write_text(final_text, encoding="utf-8")
            logger.info(f"Created {env_path} and stored ELEVENLABS_API_KEY")

        except Exception as e:
            logger.warning(f"Could not persist ELEVENLABS_API_KEY: {e}")
