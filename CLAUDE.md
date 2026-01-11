# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Conversational app for the Reachy Mini robot combining ElevenLabs Conversational AI (realtime), vision pipelines, and choreographed motion libraries.

## Common Commands

```bash
# Install dependencies
uv sync                              # Basic install
uv sync --extra all_vision --group dev  # Full dev environment with all vision extras

# Run the application
reachy-mini-conversation-app         # Console mode (requires physical robot)
reachy-mini-conversation-app --gradio  # Gradio web UI (required for simulation)
reachy-mini-conversation-app --head-tracker mediapipe  # With face tracking

# Linting and formatting
ruff check .                         # Run linter
ruff check --fix .                   # Auto-fix issues
ruff format .                        # Format code

# Type checking
mypy src/

# Testing
pytest                               # Run all tests
pytest tests/test_openai_realtime.py  # Run specific test file
pytest -v -k "test_name"             # Run specific test by name
```

## Architecture

### Runtime Flow
```
User Audio → ElevenLabsRealtimeHandler → ElevenLabs API → TTS Audio → Speaker
                    ↓ (tool calls)
              dispatch_tool_call() → Robot actions (camera, dance, emotions, head movement)
```

### Core Components

**`elevenlabs_realtime.py`** - Main integration handler (`ElevenLabsRealtimeHandler`). Manages bidirectional 16kHz PCM16 audio streaming with ElevenLabs Conversational AI. Handles transcripts, tool calls, and audio output.

**`moves.py`** - Movement system with `MovementManager` running a 100Hz control loop. Primary moves (dances, emotions, breathing) are sequential and mutually exclusive. Secondary moves (speech wobble, face tracking) are additive offsets. Single control point via `ReachyMini.set_target`.

**`tools/core_tools.py`** - Tool registry and dispatcher. Tools subclass `Tool` base class with `name`, `description`, `parameters_schema`, and async `__call__`. Tools are loaded from profile's `tools.txt` file, first checking profile-local implementations then shared library.

**`prompts.py`** - Profile and instruction loading. Expands `[placeholder]` syntax to include shared prompt templates from `prompts/` directory.

### Threading Model
- `MovementManager` owns a dedicated worker thread for the 100Hz control loop
- External threads communicate via command queue
- Secondary offset producers set pending values guarded by locks
- Each async service (movement_manager, head_wobbler, camera_worker, vision_manager) runs in its own thread

### Two Deployment Modes
- **Gradio mode** (`--gradio`): Web UI with `fastrtc.Stream`, chat visualization, personality editor
- **Console mode**: Direct audio I/O with `LocalStream`, FastAPI settings endpoints

### Profile System
Profiles live in `src/reachy_mini_conversation_app/profiles/<name>/`:
- `instructions.txt` - System prompt (supports `[template_name]` includes)
- `tools.txt` - Enabled tools (one per line, `#` for comments)
- `voice.txt` - Voice selection (optional)
- `*.py` - Custom tool implementations (optional)

## Configuration

Copy `.env.example` to `.env`. Key variables:
- `ELEVENLABS_API_KEY` - Required
- `ELEVENLABS_AGENT_ID` - Optional pre-configured agent
- `REACHY_MINI_CUSTOM_PROFILE` - Profile name (default: "default")
- `HF_TOKEN` - For emotion datasets and local vision

## Code Style

- Line length: 119 characters
- Ruff handles formatting and linting (E, F, W, I, C4, D rules)
- isort with length-sort and specific first-party packages
- mypy strict mode enabled