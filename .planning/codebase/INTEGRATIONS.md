# External Integrations

**Analysis Date:** 2026-01-21

## APIs & External Services

**Image Generation:**
- **Nano Banana Pro (Gemini 3 Pro Image)** - High-quality panoramic image generation
  - SDK/Client: HTTP via httpx
  - API: `https://generativelanguage.googleapis.com/v1beta/models/gemini-3-pro-image-preview:generateContent`
  - Auth: `GOOGLE_API_KEY` environment variable (or `NANOBANANA_API_KEY`)
  - Implementation: `splatworld_agent/generators/nano.py`
  - Features: 4K panoramic images (21:9 aspect ratio), equirectangular format
  - Timeout: 120 seconds

- **Gemini 2.0 Flash** - Alternative image generator (free tier available)
  - SDK/Client: HTTP via httpx
  - API: `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent`
  - Auth: `GOOGLE_API_KEY` environment variable
  - Implementation: `splatworld_agent/generators/gemini.py`
  - Features: Standard image generation, free tier available
  - Timeout: 120 seconds

**3D Conversion:**
- **World Labs Marble API** - Converts 2D images to 3D Gaussian splats
  - SDK/Client: HTTP via httpx (custom `MarbleClient` wrapper)
  - API Base: `https://api.worldlabs.ai/marble/v1`
  - Auth: `WORLDLABS_API_KEY` environment variable (header: `WLT-Api-Key`)
  - Implementation: `splatworld_agent/core/marble.py`
  - Endpoints:
    - `POST /worlds:generate` - Start splat generation from image
    - `GET /operations/{operation_id}` - Poll operation status
  - Output formats:
    - `.spz` file (3D Gaussian splat)
    - `.glb` file (collision mesh)
    - Web viewer URL: `https://worldlabs.ai/viewer/{world_id}`
  - Cost: $1.50 per conversion
  - Timeout: 600 seconds (10 minutes) for generation
  - Polling: 10-second intervals with exponential backoff on rate limits
  - Error handling: Retry logic with exponential backoff for 500+ errors and 429 rate limits

**Preference Learning:**
- **Anthropic Claude API** - Synthesizes user feedback into taste profile preferences
  - SDK/Client: `anthropic` Python library (0.27+)
  - Auth: `ANTHROPIC_API_KEY` environment variable
  - Implementation: `splatworld_agent/learning.py`
  - Model: Claude (via Anthropic client, version flexible)
  - Purpose: Analyzes generation + feedback pairs to extract visual preferences
  - System prompt: Structured JSON analysis for visual style, composition, domain, quality preferences
  - Confidence delta: 0.05 (weak), 0.1 (moderate), 0.2 (strong) signals

## Data Storage

**Databases:**
- None - File-based storage only

**File Storage:**
- **Local filesystem**
  - Configuration: `~/.splatworld_agent/config.yaml` - Global API keys and defaults
  - Project directory: `.splatworld/` in each project
    - `profile.json` - Taste profile (preferences, stats, calibration status)
    - `generations/{gen_id}/` - Per-generation storage
      - `metadata.json` - Generation metadata, paths, costs
      - `source.png` - Generated image
      - `scene.spz` - 3D Gaussian splat (optional, downloaded from Marble)
      - `collision.glb` - Collision mesh (optional, downloaded from Marble)
    - `feedback/` - Feedback history as JSON files
    - `sessions/` - Session history and activity tracking

**Caching:**
- None detected - All API responses processed immediately

## Authentication & Identity

**Auth Provider:**
- None - All integrations use API key authentication

**API Key Management:**
- Configuration file: `splatworld_agent/config.py`
- Location: `~/.splatworld_agent/config.yaml` (YAML format)
- Environment precedence: Environment variables override config file values
- Setup command: `splatworld-agent setup-keys` CLI command

## Monitoring & Observability

**Error Tracking:**
- None - Errors printed to console via Rich output

**Logs:**
- **Rich console output** - Styled terminal logging in `splatworld_agent/cli.py`
- **No persistent logs** - All logging is console-based with Rich formatting
- Progress tracking: Rich Progress bars for long operations (image generation, conversions)

## CI/CD & Deployment

**Hosting:**
- Self-hosted - Runs as CLI tool on user's machine
- Distributed via pip package installation

**CI Pipeline:**
- Not detected - No GitHub Actions, CircleCI, or similar in codebase

**Testing Infrastructure:**
- pytest configuration in `pyproject.toml`:
  - asyncio_mode: auto
  - testpaths: tests/
- Test file: `tests/test_models.py`

## Environment Configuration

**Required env vars:**
```
ANTHROPIC_API_KEY        # For taste profile learning
WORLDLABS_API_KEY        # For 3D splat conversion
GOOGLE_API_KEY           # For Gemini image generation
```

**Optional env vars:**
```
NANOBANANA_API_KEY       # Alternative to GOOGLE_API_KEY for Nano Banana Pro
```

**Secrets location:**
- File: `~/.splatworld_agent/config.yaml` (for persisted keys)
- Environment: Direct environment variables (recommended)

## Webhooks & Callbacks

**Incoming:**
- None detected

**Outgoing:**
- Marble API supports `on_progress` callback for polling updates
  - Called during `marble.generate_and_wait()` with (status, description) tuples
  - Used in `splatworld_agent/cli.py` to update progress UI

## File Delivery

**Download Support:**
- `MarbleClient.download_file()` - Downloads `.spz` and `.glb` files from generated URLs
- Uses httpx with follow redirects, retries on network errors
- Saves to local filesystem at user-specified path

## API Response Patterns

**Image Generators (Nano/Gemini):**
- Request: POST with JSON payload containing prompt in "parts"
- Response: JSON with base64-encoded inline image data
- Error handling: Check response status codes and error field in JSON

**Marble API:**
- Request: POST/GET with JSON body and `WLT-Api-Key` header
- Response: JSON with operation metadata, completion status, resource URLs
- Long-running: Async operations requiring polling for completion
- Error responses: Error field in JSON, HTTP status codes 401 (auth), 429 (rate limit), 5xx (server)

---

*Integration audit: 2026-01-21*
