# Technology Stack

**Analysis Date:** 2026-01-21

## Languages

**Primary:**
- Python 3.10+ - Main application language; supports 3.10, 3.11, 3.12

## Runtime

**Environment:**
- Python 3.10, 3.11, or 3.12 runtime
- Package requires `>=3.10` per `pyproject.toml`

**Package Manager:**
- pip (via setuptools)
- Lockfile: Not detected (no poetry.lock, requirements.txt, or similar)

## Frameworks

**Core:**
- Click 8.0+ - CLI framework for command-line interface at `splatworld_agent/cli.py`
- Pydantic 2.0+ - Data validation and model serialization in `splatworld_agent/models.py`

**Logging & Output:**
- Rich 13.0+ - Terminal UI, tables, progress bars, and styled output (used throughout `cli.py`)

**HTTP Client:**
- httpx 0.25+ - Async HTTP client for API calls (used in generators and Marble client)

**Configuration:**
- PyYAML 6.0+ - YAML file parsing for config loading/saving in `splatworld_agent/config.py`
- python-dotenv 1.0+ - Environment variable management

**Testing:**
- pytest 7.0+ - Test runner
- pytest-asyncio 0.21+ - Async test support

**Code Quality:**
- Black 23.0+ - Code formatting (line-length: 100)
- Ruff 0.1+ - Linting (enforces E, F, I, N, W rules)

## Key Dependencies

**Critical:**
- anthropic 0.27+ - Claude API client for taste profile learning via `splatworld_agent/learning.py`
- httpx 0.25+ - HTTP requests for all external APIs (image generators and Marble)

**Infrastructure:**
- pydantic 2.0+ - Strongly typed data models for profiles, generations, feedback
- click 8.0+ - CLI command routing and argument parsing
- pyyaml 6.0+ - Configuration file storage at `~/.splatworld_agent/config.yaml`
- rich 13.0+ - User-facing terminal output and interactivity

## Configuration

**Environment:**
- Configured via environment variables and YAML file at `~/.splatworld_agent/config.yaml`
- Global config directory: `~/.splatworld_agent/`
- Project-level directory: `.splatworld/` in each project

**Environment Variables Required:**
- `ANTHROPIC_API_KEY` - Required for taste profile learning
- `WORLDLABS_API_KEY` - Required for 3D Gaussian splat conversion (Marble API)
- `GOOGLE_API_KEY` - Required for Google Gemini 2.0 Flash image generation
- `NANOBANANA_API_KEY` or `GOOGLE_API_KEY` - For Nano Banana Pro (Gemini 3 Pro Image)

**Build:**
- setuptools 61.0+ - Package building
- wheel - Binary distribution format
- Configuration in `pyproject.toml` with setuptools backend

## Package Structure

**Entry Point:**
- `splatworld-agent` CLI command defined in `pyproject.toml`: `splatworld_agent.cli:main`

**Core Modules:**
- `splatworld_agent/cli.py` - Command-line interface (1,651 lines)
- `splatworld_agent/config.py` - Configuration management
- `splatworld_agent/models.py` - Data models (Pydantic)
- `splatworld_agent/learning.py` - Claude-powered preference synthesis
- `splatworld_agent/profile.py` - Taste profile persistence
- `splatworld_agent/core/marble.py` - World Labs Marble API client
- `splatworld_agent/generators/nano.py` - Nano Banana Pro (Gemini 3 Pro Image)
- `splatworld_agent/generators/gemini.py` - Gemini 2.0 Flash image generator

## Platform Requirements

**Development:**
- Python 3.10+ with pip
- Tested on 3.10, 3.11, 3.12
- No OS-specific dependencies (platform-independent)

**Production/Deployment:**
- Python 3.10+ runtime
- Network access to external APIs:
  - Google Generative AI (Gemini APIs)
  - World Labs (Marble API)
  - Anthropic (Claude API)
- Local filesystem for `.splatworld/` project directory and `~/.splatworld_agent/` config

## Optional Dependencies

**Nano Banana Pro:**
- Placeholder for future Nano Banana Pro SDK when available
- Currently uses direct HTTP calls to Google Gemini API

**Gemini:**
- `google-genai 1.0+` - Optional for future Google SDK integration (not currently used)

**Development:**
- `pytest 7.0+` - Testing framework
- `pytest-asyncio 0.21+` - Async test support
- `black 23.0+` - Code formatting
- `ruff 0.1+` - Linting

---

*Stack analysis: 2026-01-21*
