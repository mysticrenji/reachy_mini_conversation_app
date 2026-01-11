import json
import base64
import asyncio
import logging
from typing import Any, Final, Tuple, Literal, Optional
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import gradio as gr
from elevenlabs import ElevenLabs
from elevenlabs.conversational_ai.conversation import Conversation
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

# ElevenLabs audio format - typically 16kHz for conversational AI
ELEVENLABS_SAMPLE_RATE: Final[Literal[16000]] = 16000


class ElevenLabsRealtimeHandler(AsyncStreamHandler):
    """An ElevenLabs Conversational AI handler for fastrtc Stream."""

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
        self.output_queue: "asyncio.Queue[Tuple[int, NDArray[np.int16]] | AdditionalOutputs]" = asyncio.Queue()

        self.last_activity_time = asyncio.get_event_loop().time()
        self.start_time = asyncio.get_event_loop().time()
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
        self._audio_task: Optional[asyncio.Task] = None
        self._event_task: Optional[asyncio.Task] = None

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
                await self.conversation.end_session()
            await asyncio.sleep(0.5)
            asyncio.create_task(self._run_conversation_session(), name="elevenlabs-conversation-restart")
            await asyncio.wait_for(self._connected_event.wait(), timeout=5.0)
            logger.info("Conversation session restarted and connected.")
        except asyncio.TimeoutError:
            logger.warning("Conversation session restart timed out; continuing in background.")
        except Exception as e:
            logger.warning("_restart_conversation failed: %s", e)

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

    async def _run_conversation_session(self) -> None:
        """Establish and manage a single conversation session."""
        if not self.client:
            logger.error("ElevenLabs client not initialized")
            return

        try:
            # Get session configuration
            instructions = get_session_instructions()
            voice = get_session_voice()
            
            # Convert tools to ElevenLabs format
            tools_specs = get_tool_specs()
            elevenlabs_tools = self._convert_tools_to_elevenlabs_format(tools_specs)

            # Create conversation configuration
            conversation_config = {
                "agent": {
                    "prompt": {
                        "prompt": instructions,
                    },
                    "first_message": "Hello! I'm ready to chat with you.",
                    "language": "en",
                },
            }

            # If agent_id is configured, use it; otherwise use dynamic config
            if config.ELEVENLABS_AGENT_ID:
                logger.info("Using pre-configured agent: %s", config.ELEVENLABS_AGENT_ID)
                self.conversation = await self.client.conversational_ai.conversations.start_session(
                    agent_id=config.ELEVENLABS_AGENT_ID
                )
            else:
                logger.info("Creating conversation with voice=%s", voice)
                self.conversation = await self.client.conversational_ai.conversations.start_session(
                    agent_override=conversation_config
                )

            # Persist API key if needed
            self._persist_api_key_if_needed()
            
            logger.info("ElevenLabs conversation session initialized with profile=%r voice=%r",
                       config.REACHY_MINI_CUSTOM_PROFILE, voice)

            # Mark as connected
            self._connected_event.set()

            # Start audio and event processing tasks
            self._audio_task = asyncio.create_task(self._process_audio_stream())
            self._event_task = asyncio.create_task(self._process_events())

            # Wait for tasks to complete
            await asyncio.gather(self._audio_task, self._event_task, return_exceptions=True)

        except Exception as e:
            logger.exception("Conversation session failed: %s", e)

    def _convert_tools_to_elevenlabs_format(self, tools_specs: list) -> list:
        """Convert OpenAI-style tool specs to ElevenLabs format."""
        elevenlabs_tools = []
        for tool in tools_specs:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                elevenlabs_tools.append({
                    "name": func.get("name"),
                    "description": func.get("description"),
                    "parameters": func.get("parameters", {}),
                })
        return elevenlabs_tools

    async def _process_audio_stream(self) -> None:
        """Process incoming audio from ElevenLabs."""
        if not self.conversation:
            return

        try:
            async for audio_chunk in self.conversation.audio_output:
                if self._shutdown_requested:
                    break

                # ElevenLabs provides PCM16 audio
                audio_array = np.frombuffer(audio_chunk, dtype=np.int16).reshape(1, -1)
                
                # Feed to head wobbler if available
                if self.deps.head_wobbler is not None:
                    # Convert to base64 for compatibility
                    audio_b64 = base64.b64encode(audio_chunk).decode("utf-8")
                    self.deps.head_wobbler.feed(audio_b64)
                
                self.last_activity_time = asyncio.get_event_loop().time()
                
                # Queue audio for playback
                await self.output_queue.put((self.output_sample_rate, audio_array))

        except Exception as e:
            if not self._shutdown_requested:
                logger.error("Audio processing error: %s", e)

    async def _process_events(self) -> None:
        """Process events from ElevenLabs conversation."""
        if not self.conversation:
            return

        try:
            async for event in self.conversation.events:
                if self._shutdown_requested:
                    break

                logger.debug(f"ElevenLabs event: {event.get('type')}")
                event_type = event.get("type")

                # User speaking events
                if event_type == "user_speaking_started":
                    if hasattr(self, "_clear_queue") and callable(self._clear_queue):
                        self._clear_queue()
                    if self.deps.head_wobbler is not None:
                        self.deps.head_wobbler.reset()
                    self.deps.movement_manager.set_listening(True)
                    logger.debug("User speech started")

                elif event_type == "user_speaking_stopped":
                    self.deps.movement_manager.set_listening(False)
                    logger.debug("User speech stopped")

                # Transcription events
                elif event_type == "user_transcript":
                    transcript = event.get("transcript", "")
                    is_final = event.get("is_final", False)
                    
                    if is_final:
                        # Cancel any pending partial emission
                        if self.partial_transcript_task and not self.partial_transcript_task.done():
                            self.partial_transcript_task.cancel()
                        
                        await self.output_queue.put(
                            AdditionalOutputs({"role": "user", "content": transcript})
                        )
                        logger.debug(f"User transcript (final): {transcript}")
                    else:
                        # Handle partial transcript with debouncing
                        self.partial_transcript_sequence += 1
                        current_sequence = self.partial_transcript_sequence
                        
                        if self.partial_transcript_task and not self.partial_transcript_task.done():
                            self.partial_transcript_task.cancel()
                        
                        self.partial_transcript_task = asyncio.create_task(
                            self._emit_debounced_partial(transcript, current_sequence)
                        )
                        logger.debug(f"User partial transcript: {transcript}")

                elif event_type == "agent_response":
                    transcript = event.get("transcript", "")
                    await self.output_queue.put(
                        AdditionalOutputs({"role": "assistant", "content": transcript})
                    )
                    logger.debug(f"Assistant transcript: {transcript}")

                # Tool/Function calling
                elif event_type == "tool_call":
                    tool_name = event.get("tool_name")
                    tool_args = event.get("parameters", {})
                    call_id = event.get("tool_call_id")

                    if not tool_name:
                        logger.error("Invalid tool call: missing tool_name")
                        continue

                    try:
                        # Convert args dict to JSON string for dispatch
                        args_json_str = json.dumps(tool_args)
                        tool_result = await dispatch_tool_call(tool_name, args_json_str, self.deps)
                        logger.debug("Tool '%s' executed successfully", tool_name)
                        logger.debug("Tool result: %s", tool_result)
                    except Exception as e:
                        logger.error("Tool '%s' failed: %s", tool_name, e)
                        tool_result = {"error": str(e)}

                    # Send tool result back to ElevenLabs
                    if call_id and self.conversation:
                        await self.conversation.send_tool_result(
                            tool_call_id=call_id,
                            result=json.dumps(tool_result)
                        )

                    # Display tool usage in UI
                    await self.output_queue.put(
                        AdditionalOutputs({
                            "role": "assistant",
                            "content": json.dumps(tool_result),
                            "metadata": {"title": f"🛠️ Used tool {tool_name}", "status": "done"},
                        })
                    )

                    # Handle camera tool image display
                    if tool_name == "camera" and "b64_im" in tool_result:
                        # Send image back to the conversation for vision context
                        b64_im = tool_result["b64_im"]
                        if self.conversation:
                            # ElevenLabs may support sending context - adapt as needed
                            logger.info("Camera image captured for context")

                        # Display image in Gradio
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

                    # Handle idle tool calls
                    if self.is_idle_tool_call:
                        self.is_idle_tool_call = False

                    # Resync head wobble after tool call
                    if self.deps.head_wobbler is not None:
                        self.deps.head_wobbler.reset()

                # Error handling
                elif event_type == "error":
                    error_msg = event.get("message", "Unknown error")
                    logger.error("ElevenLabs error: %s", error_msg)
                    await self.output_queue.put(
                        AdditionalOutputs({"role": "assistant", "content": f"[error] {error_msg}"})
                    )

        except Exception as e:
            if not self._shutdown_requested:
                logger.error("Event processing error: %s", e)

    async def receive(self, frame: Tuple[int, NDArray[np.int16]]) -> None:
        """Receive audio frame from the microphone and send it to ElevenLabs.

        Handles both mono and stereo audio formats, resampling as needed.

        Args:
            frame: A tuple containing (sample_rate, audio_data).
        """
        if not self.conversation:
            return

        input_sample_rate, audio_frame = frame

        # Reshape if needed
        if audio_frame.ndim == 2:
            if audio_frame.shape[1] > audio_frame.shape[0]:
                audio_frame = audio_frame.T
            if audio_frame.shape[1] > 1:
                audio_frame = audio_frame[:, 0]

        # Resample if needed
        if self.input_sample_rate != input_sample_rate:
            audio_frame = resample(audio_frame, int(len(audio_frame) * self.input_sample_rate / input_sample_rate))

        # Cast to int16
        audio_frame = audio_to_int16(audio_frame)

        # Send to ElevenLabs
        try:
            audio_bytes = audio_frame.tobytes()
            await self.conversation.send_audio(audio_bytes)
        except Exception as e:
            logger.debug("Dropping audio frame: connection not ready (%s)", e)

    async def emit(self) -> Tuple[int, NDArray[np.int16]] | AdditionalOutputs | None:
        """Emit audio frame to be played by the speaker."""
        # Handle idle detection
        idle_duration = asyncio.get_event_loop().time() - self.last_activity_time
        if idle_duration > 15.0 and self.deps.movement_manager.is_idle():
            try:
                await self.send_idle_signal(idle_duration)
            except Exception as e:
                logger.warning("Idle signal skipped: %s", e)
                return None
            self.last_activity_time = asyncio.get_event_loop().time()

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
                await self.conversation.end_session()
            except Exception as e:
                logger.debug(f"Conversation end_session error: {e}")
            finally:
                self.conversation = None

        # Cancel processing tasks
        for task in [self._audio_task, self._event_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Clear output queue
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    def format_timestamp(self) -> str:
        """Format current timestamp with date, time, and elapsed seconds."""
        loop_time = asyncio.get_event_loop().time()
        elapsed_seconds = loop_time - self.start_time
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
            voices = await self.client.voices.get_all()
            return [voice.name.lower() for voice in voices.voices]
        except Exception as e:
            logger.warning(f"Failed to fetch voices: {e}")
            return ["rachel", "clyde", "domi", "dave", "fin", "sarah"]

    async def send_idle_signal(self, idle_duration: float) -> None:
        """Send an idle signal to trigger idle behavior."""
        logger.debug("Sending idle signal")
        self.is_idle_tool_call = True
        
        timestamp_msg = (
            f"[Idle time update: {self.format_timestamp()} - No activity for {idle_duration:.1f}s] "
            f"You've been idle for a while. Feel free to get creative - dance, show an emotion, "
            f"look around, do nothing, or just be yourself!"
        )
        
        if not self.conversation:
            logger.debug("No conversation, cannot send idle signal")
            return
        
        # Send text input to trigger idle behavior
        try:
            await self.conversation.send_text(timestamp_msg)
        except Exception as e:
            logger.warning(f"Failed to send idle signal: {e}")

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
