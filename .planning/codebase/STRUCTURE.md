# Codebase Structure

**Analysis Date:** 2026-01-21

## Directory Layout

```
splatworld_agent/
├── splatworld_agent/          # Main package
│   ├── __init__.py            # Package init, exports public API
│   ├── cli.py                 # Click CLI commands (1650 lines)
│   ├── config.py              # Config/API key management
│   ├── models.py              # Dataclass definitions for all entities
│   ├── profile.py             # ProfileManager for file I/O
│   ├── learning.py            # LearningEngine for preference synthesis
│   ├── core/                  # External integrations
│   │   ├── __init__.py
│   │   └── marble.py          # Marble API client for splat conversion
│   └── generators/            # Image generation backends
│       ├── __init__.py        # ImageGenerator base class
│       ├── nano.py            # Nano Banana Pro (Gemini 3)
│       └── gemini.py          # Gemini fallback generator
├── tests/                     # Test suite
│   ├── __init__.py
│   └── test_models.py         # Model serialization tests
├── prompts/                   # Claude Code slash command templates
│   ├── init.md
│   ├── generate.md
│   ├── batch.md
│   ├── review.md
│   ├── feedback.md
│   ├── learn.md
│   ├── profile.md
│   ├── history.md
│   ├── exemplar.md
│   └── help.md
├── commands/                  # Command documentation
│   ├── init.md
│   ├── generate.md
│   ├── train.md
│   ├── batch.md
│   ├── review.md
│   ├── convert.md
│   ├── feedback.md
│   ├── learn.md
│   ├── profile.md
│   ├── exit.md
│   ├── resume-work.md
│   ├── update.md
│   ├── help.md
│   └── ... (other command docs)
├── pyproject.toml             # Package metadata and dependencies
├── setup.py                   # Setup script
├── PROJECT.md                 # Project vision and requirements
└── README.md                  # User-facing documentation
```

## Directory Purposes

**splatworld_agent/:**
- Purpose: Main package containing all source code
- Contains: CLI, models, managers, integrations, generators
- Key files: `cli.py` (massive, all commands), `models.py` (data structures)

**splatworld_agent/core/:**
- Purpose: External API integrations
- Contains: Marble API client for 3D conversion
- Key files: `marble.py` (image→splat conversion)

**splatworld_agent/generators/:**
- Purpose: Pluggable image generation backends
- Contains: Abstract base, Nano Banana Pro (Google Gemini 3), Gemini alternative
- Key files: `nano.py` (default, best quality), `gemini.py` (fallback)

**tests/:**
- Purpose: Test suite
- Contains: Model serialization tests (basic coverage)
- Key files: `test_models.py` (TasteProfile, Generation, Feedback serialization)

**prompts/:**
- Purpose: Claude Code slash command templates
- Contains: `.md` files that Claude Code reads for command help/context
- Key files: One per major command (generate.md, batch.md, train.md, etc.)

**commands/:**
- Purpose: User-facing command documentation
- Contains: Markdown docs for all CLI commands
- Key files: One per command (init.md, generate.md, batch.md, etc.)

## Key File Locations

**Entry Points:**
- `splatworld_agent/cli.py`: CLI main(@click.group), all command handlers
- `pyproject.toml [project.scripts]`: Entry point mapped to `splatworld_agent.cli:main`
- User runs: `splatworld-agent init`, `splatworld-agent generate`, etc.

**Configuration:**
- `splatworld_agent/config.py`: Config, APIKeys, Defaults classes
- Global config: `~/.splatworld_agent/config.yaml` (user API keys, defaults)
- Project config: `.splatworld/` directory (per-project, created by `init`)

**Core Logic:**
- `splatworld_agent/models.py`: All dataclasses (TasteProfile, Generation, Feedback, etc.)
- `splatworld_agent/profile.py`: ProfileManager handles all file I/O to `.splatworld/`
- `splatworld_agent/learning.py`: LearningEngine calls Claude for preference synthesis
- `splatworld_agent/generators/nano.py`: Image generation via Gemini 3 Pro Image

**Testing:**
- `tests/test_models.py`: Serialization roundtrip tests for models
- Run: `pytest tests/`

**Runtime Data Storage (per-project):**
- `.splatworld/profile.json`: Main taste profile (JSON)
- `.splatworld/feedback.jsonl`: Append-only feedback log (one JSON object per line)
- `.splatworld/sessions.jsonl`: Session history (one per line)
- `.splatworld/current_session.json`: Active session tracking
- `.splatworld/generations/YYYY-MM-DD/gen-{id}/metadata.json`: Generation metadata
- `.splatworld/generations/YYYY-MM-DD/gen-{id}/source.png`: Generated image
- `.splatworld/generations/YYYY-MM-DD/gen-{id}/scene.spz`: 3D splat (if converted)
- `.splatworld/exemplars/`: Reference images user loves
- `.splatworld/anti-exemplars/`: Reference images user hates

## Naming Conventions

**Files:**
- `cli.py`: Entry point and command definitions
- `models.py`: Data structures
- `profile.py`: State management
- `learning.py`: AI/ML features
- `config.py`: Configuration
- `marble.py`: Third-party integration (Marble API)
- `nano.py`: Generator implementation (Nano)
- `gemini.py`: Generator implementation (Gemini)
- `test_*.py`: Test files (pytest discovery)

**Directories:**
- `splatworld_agent/`: Main package (matches pip package name, converted underscores)
- `generators/`: Pluggable backends
- `core/`: Core integrations
- `tests/`: Test suite
- `prompts/`: Claude Code command templates
- `commands/`: User documentation

**Python Identifiers:**
- Classes: PascalCase (TasteProfile, ProfileManager, MarbleClient)
- Functions: snake_case (enhance_prompt, get_project_dir)
- Constants: UPPER_SNAKE_CASE (API_BASE, PROJECT_DIR_NAME, MARBLE_COST_PER_GENERATION)
- Private: _leading_underscore for internal methods

## Where to Add New Code

**New Feature (e.g., new workflow):**
- Primary code: Add new @main.command() to `splatworld_agent/cli.py`
- Models: Add new dataclass to `splatworld_agent/models.py` if needed
- Manager methods: Add to ProfileManager in `splatworld_agent/profile.py` for file I/O
- Tests: Add to `tests/test_models.py` or create `tests/test_feature.py`

**New Generator Backend (e.g., DALL-E):**
- Implementation: Create `splatworld_agent/generators/dalle.py`
- Inherit from: `ImageGenerator` base class (implement generate(), name())
- Register: Add lazy import to `splatworld_agent/generators/__init__.py`
- CLI: Update `generate` command to accept `--generator dalle` option

**New API Integration:**
- Create module: `splatworld_agent/integrations/service.py` or under `core/`
- Follow MarbleClient pattern: class with context manager support
- Import in CLI command and call via config validation check

**Utilities/Helpers:**
- Shared functions: Add to new `splatworld_agent/utils.py`
- Import in cli.py and other modules as needed
- Keep functions small and testable

## Special Directories

**`.splatworld/` (Runtime Project Directory):**
- Purpose: Per-project data storage (created on `init`)
- Generated: Yes, created by `splatworld-agent init`
- Committed: No, user adds to `.gitignore`
- Contains: Profile, feedback log, generations, exemplars, sessions
- Lifecycle: Created once per project, persists across sessions

**`~/.splatworld_agent/` (Global Config Directory):**
- Purpose: User's global API key configuration
- Generated: Yes, created on first `setup-keys` or `install-prompts`
- Committed: No, user-specific
- Contains: `config.yaml` (API keys, defaults)
- Lifecycle: Created once per user, persists across projects

**`.git/` (Repository):**
- Purpose: Git version control
- Generated: Yes, repo structure
- Committed: Yes (except .gitignore)
- Key files: `.gitignore` excludes `.splatworld/` and `~/.splatworld_agent/`

## File Organization Patterns

**Command-Driven Architecture:**
- All user-facing operations are Click commands in `cli.py`
- Commands are organized by workflow (init, generate, batch, train, review, convert, learn)
- Each command imports specialized managers/engines as needed (lazy loading)

**Manager Pattern:**
- ProfileManager: single class handling all `.splatworld/` I/O
- Config: handles config file and env var loading
- ImageGenerator: abstract base with concrete implementations

**Data Model Separation:**
- Models in `models.py`: pure data structures with serialization
- Managers in `profile.py`: file I/O and business logic
- CLI in `cli.py`: user interaction and workflow orchestration

**Integration Pattern:**
- External APIs (Marble, image generators) isolated in `core/` or `generators/`
- Each has its own class (MarbleClient, NanoGenerator)
- CLI commands instantiate and manage lifecycle (with context managers where possible)

---

*Structure analysis: 2026-01-21*
