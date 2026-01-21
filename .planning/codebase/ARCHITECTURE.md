# Architecture

**Analysis Date:** 2026-01-21

## Pattern Overview

**Overall:** Command-driven CLI with pluggable backends and learned preference synthesis

**Key Characteristics:**
- CLI-first entry point via Click framework, designed for Claude Code integration via slash commands
- Learnable taste profile system that evolves with user feedback
- Pluggable image generation backends (Nano Banana Pro/Gemini default, extensible)
- Multi-step generation pipeline: prompt → image → 3D splat conversion
- File-based, per-project storage in `.splatworld/` directory
- Claude-powered learning to extract preferences from feedback patterns

## Layers

**CLI Command Layer:**
- Purpose: Parse user commands and orchestrate workflows
- Location: `splatworld_agent/cli.py`
- Contains: Click command definitions (@main.command) for all operations
- Depends on: ProfileManager, Config, LearningEngine, generators, MarbleClient
- Used by: User invocations via `splatworld-agent` command

**Profile & State Management Layer:**
- Purpose: Persist and manage taste profiles, generations, feedback, sessions
- Location: `splatworld_agent/profile.py`, `splatworld_agent/models.py`
- Contains: ProfileManager (file I/O orchestration), dataclasses for all entities
- Depends on: File system, json module
- Used by: CLI commands, LearningEngine

**Learning & Preference Extraction:**
- Purpose: Analyze feedback and synthesize preference updates via Claude
- Location: `splatworld_agent/learning.py`
- Contains: LearningEngine (calls Claude to analyze patterns)
- Depends on: Anthropic API, models, ProfileManager
- Used by: `train`, `learn`, `batch`, `feedback` commands

**Generation Pipeline:**
- Purpose: Generate images and convert to 3D splats
- Location: `splatworld_agent/generators/`, `splatworld_agent/core/marble.py`
- Contains: ImageGenerator (abstract), NanoGenerator, GeminiGenerator, MarbleClient
- Depends on: httpx for API calls, external image/splat APIs
- Used by: `generate`, `batch`, `train`, `convert` commands

**Configuration & Discovery:**
- Purpose: Manage API keys, defaults, and project initialization
- Location: `splatworld_agent/config.py`
- Contains: Config, APIKeys, Defaults dataclasses, get_project_dir()
- Depends on: File system, environment variables, YAML
- Used by: All commands to validate setup and locate projects

## Data Flow

**Generation Flow (single image):**
1. User invokes `generate "prompt"` command
2. ProfileManager loads taste profile from `.splatworld/profile.json`
3. LearningEngine enhances prompt with learned preferences (if calibrated)
4. ImageGenerator (Nano/Gemini) generates image from enhanced prompt
5. Image bytes saved to `.splatworld/generations/YYYY-MM-DD/gen-{id}/source.png`
6. Generation metadata saved to `.splatworld/generations/YYYY-MM-DD/gen-{id}/metadata.json`
7. (Optional) MarbleClient converts image → 3D splat, saves to scene.spz
8. ProfileManager updates profile stats (total_generations++)

**Feedback & Learning Flow:**
1. User rates generation: `feedback ++` or `review` (interactive)
2. Feedback saved to `.splatworld/feedback.jsonl` (append-only log)
3. ProfileManager updates profile stats (feedback_count++, love_count++, etc.)
4. When threshold reached (auto_learn_threshold=10), or user runs `learn`:
5. LearningEngine loads profile + generations + feedback history
6. Claude analyzes patterns and returns suggested updates (JSON)
7. Updates applied to profile: visual_style, composition, domain, quality preferences
8. Calibration checked: if feedback_count >= 20 AND positive/negative distribution OK, mark calibrated
9. Updated profile saved

**Batch Workflow:**
1. `batch "prompt" -n 5 -c 2` generates 5 images, user reviews, learns, repeats for 2 cycles
2. Each image saved with batch_id metadata
3. After each cycle: review → learn → (next cycle or stop)
4. Learned preferences applied to subsequent cycles' enhanced prompts

**Training (Guided Calibration):**
1. `train "prompt"` runs generate-review-learn loop until profile is calibrated
2. Each round: generates N images → user rates → learns
3. Continues until: feedback_count >= 20 AND positive_ratio >= 10% AND negative_ratio >= 10%
4. Session tracked for resume capability

**State Management:**
- Profiles: loaded at command start, saved after modifications
- Generations: stored per-date in nested directories, loaded on-demand
- Feedback: append-only JSONL for audit trail
- Sessions: track work periods and allow resumption

## Key Abstractions

**TasteProfile (models.py):**
- Purpose: Represents learned aesthetic preferences
- Examples: `visual_style.lighting.preference`, `composition.density.avoid`, `domain.environments`
- Pattern: Nested dataclass hierarchy with to_dict/from_dict for persistence
- Properties: is_calibrated (bool), training_progress (str), to_prompt_context() → enhanced prompt

**ImageGenerator (generators/__init__.py):**
- Purpose: Abstract interface for image generation backends
- Examples: `NanoGenerator`, `GeminiGenerator`
- Pattern: ABC with generate(prompt, seed) → bytes
- Used by: CLI selects generator via config or --generator flag

**Generation & Feedback (models.py):**
- Purpose: Represent outputs and their ratings
- Generation: id, prompt, enhanced_prompt, timestamp, metadata, paths (image/splat/mesh)
- Feedback: generation_id, rating (++/+/-/--/text), text, extracted_preferences
- Pattern: Dataclasses with to_dict/from_dict, optional linked feedback on Generation

**ProfileManager (profile.py):**
- Purpose: File I/O abstraction for all profile state
- Examples: load_profile(), save_generation(), add_feedback(), get_recent_generations()
- Pattern: Centralized manager, knows about `.splatworld/` directory structure
- Key: Separates file handling from business logic

**MarbleClient (core/marble.py):**
- Purpose: Wraps World Labs Marble API for image→splat conversion
- Methods: generate_and_wait(image_base64) → MarbleResult, download_file()
- Pattern: HTTP client with polling loop for async job completion
- Cost tracking: $1.50 per conversion

## Entry Points

**CLI Main Entry:**
- Location: `splatworld_agent/cli.py:main()`
- Triggers: `splatworld-agent COMMAND [options]`
- Responsibilities: Parse CLI options, delegate to command handlers

**Key Command Entry Points:**
- `init`: Initialize new project with empty `.splatworld/` structure
- `generate`: Single image generation with optional splat conversion
- `train`: Guided calibration loop (most important for onboarding)
- `batch`: Multi-image review workflow
- `review`: Interactive rating interface
- `convert`: Batch splat conversion for loved images
- `learn`: Explicit preference synthesis from feedback
- `profile`: View/edit taste profile
- `feedback`: Rate a generation
- `setup-keys`: Configure API credentials (global)
- `resume-work`: Show session history and start new session

## Error Handling

**Strategy:** Fail-fast with clear error messages, no silent failures

**Patterns:**
- Config validation before operations: `config.validate()` returns list of issues
- Generation API errors → wrapped in MarbleError, MarbleAuthError, MarbleTimeoutError
- File I/O errors → let exceptions propagate, caught by CLI with try/except
- Missing project: checked via `get_project_dir()`, exits if not found
- Preference synthesis: graceful degradation if Claude returns partial updates

## Cross-Cutting Concerns

**Logging:** Rich Console for formatted output, progress spinners for long operations

**Validation:**
- Config validation: check for required API keys before operations
- Profile calibration: stats.can_calibrate() checks minimum feedback + distribution
- Input validation: Click handles argument/option validation

**Authentication:**
- API keys loaded from environment (priority) or ~/.splatworld_agent/config.yaml
- Per-service: Marble (WORLDLABS_API_KEY), Nano (GOOGLE_API_KEY), Anthropic (ANTHROPIC_API_KEY)
- Validation: check_keys command verifies setup

**Session Management:**
- Start session: `resume-work` command creates new Session in current_session.json
- Track activity: generations, feedback, conversions, learns during session
- End session: `exit` command records to sessions.jsonl and calculates duration
- Resumption: `resume-work` shows recent session history and current status

---

*Architecture analysis: 2026-01-21*
