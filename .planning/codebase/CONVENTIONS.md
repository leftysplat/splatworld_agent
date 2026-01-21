# Coding Conventions

**Analysis Date:** 2026-01-21

## Naming Patterns

**Files:**
- Lowercase with underscores: `models.py`, `cli.py`, `learning.py`
- Module files match their primary class/purpose: `config.py` contains `Config` class, `learning.py` contains `LearningEngine`

**Functions:**
- Snake case: `load_profile()`, `add_feedback()`, `enhance_prompt()`
- Private methods prefixed with underscore: `_update_style_pref()`, `_rating_description()`, `_update_list()`
- Command functions use action verbs: `generate()`, `feedback()`, `convert()`, `learn()`

**Variables:**
- Snake case: `api_key`, `project_dir`, `feedback_count`, `generation_id`
- Constants in UPPER_CASE: `MIN_FEEDBACK_FOR_CALIBRATION`, `PROJECT_DIR_NAME`, `PROFILE_FILE`
- Abbreviations preserved: `gen_id`, `fb`, `vm` for variables when unambiguous in context

**Types:**
- Classes use PascalCase: `TasteProfile`, `StylePreference`, `Generation`, `Feedback`, `ProfileManager`, `LearningEngine`
- Dataclass models define types on fields: `confidence: float`, `exemplars: list[Exemplar]`

**Imports:**
- Organized into groups:
  1. Standard library (datetime, json, pathlib, etc.)
  2. Third-party packages (anthropic, httpx, click, pydantic, etc.)
  3. Local imports (relative imports from same package)
- Import statement pattern: `from module import ClassName, function`
- Environment setup in docstrings at top of files

## Code Style

**Formatting:**
- Line length: 100 characters (configured in `pyproject.toml`)
- Black formatter configured for consistency
- Indentation: 4 spaces (Python standard)

**Linting:**
- Ruff configured for Python 3.10+
- Rule selection: E, F, I, N, W (errors, warnings, imports, naming, whitespace)
- E501 (line too long) ignored - Black handles this
- Configuration in `pyproject.toml`:
  ```
  [tool.ruff]
  line-length = 100
  select = ["E", "F", "I", "N", "W"]
  ignore = ["E501"]
  ```

**Docstrings:**
- Module-level docstrings describe purpose: `"""Configuration management for SplatWorld Agent."""`
- Class docstrings: One-line summary of responsibility
- Method docstrings include Args/Returns sections when non-obvious:
  ```python
  def load_profile(self) -> TasteProfile:
      """Load the taste profile."""
      if not self.profile_path.exists():
          raise FileNotFoundError(...)
  ```
- Inline comments rare; code clarity preferred over commentary

## Import Organization

**Order:**
1. Standard library: `os`, `json`, `uuid`, `base64`, `from datetime import ...`, `from pathlib import ...`, `from typing import ...`
2. Third-party: `click`, `anthropic`, `httpx`, `pydantic`, `pyyaml`, `rich`, `dataclasses`
3. Local imports: `from splatworld_agent import ...`, `from .models import ...`

**Path Aliases:**
- No path aliases configured
- Absolute imports from package root: `from splatworld_agent.models import TasteProfile`
- Relative imports within modules: `from .models import ...`

## Error Handling

**Patterns:**
- Custom exception classes inherit from base Exception: `class MarbleError(Exception):`
- Exceptions defined at module top: `MarbleAuthError`, `MarbleConversionError`, `MarbleTimeoutError` in `core/marble.py`
- CLI commands catch broad Exception with console error display:
  ```python
  except Exception as e:
      console.print(f"[red]Generation failed:[/red] {e}")
      sys.exit(1)
  ```
- User-facing keyboard interrupts caught separately:
  ```python
  except (KeyboardInterrupt, EOFError):
      console.print("\n[yellow]Review stopped.[/yellow]")
  ```
- FileNotFoundError raised with descriptive message:
  ```python
  if not self.profile_path.exists():
      raise FileNotFoundError(f"Profile not found. Run 'splatworld-agent init' first.")
  ```
- API errors provide specific error codes and retry context:
  ```python
  if response.status_code == 401:
      raise MarbleAuthError("Invalid Marble API key")
  ```

## Logging

**Framework:** Rich console for CLI output (not standard logging)

**Patterns:**
- Rich console instance: `console = Console()` at module level in `cli.py`
- Color-coded messages:
  - `[green]` for success
  - `[red]` for errors
  - `[yellow]` for warnings
  - `[cyan]` for informational
  - `[dim]` for secondary info
  - `[bold]` for emphasis
- Panels for major operations: `console.print(Panel.fit(...))`
- Tables for data display: `Table(title=...)` with `add_column()` and `add_row()`
- Progress indicators with spinners and text: `Progress(SpinnerColumn(), TextColumn(...))`
- No structured logging; stdout-based display only

## Comments

**When to Comment:**
- Comments rare; used only for non-obvious logic
- Prompt constants include detailed documentation:
  ```python
  SYNTHESIS_SYSTEM_PROMPT = """You are a taste profile analyzer..."""
  ```
- Magic numbers explained inline:
  ```python
  MIN_FEEDBACK_FOR_CALIBRATION = 20  # Minimum rated images
  MIN_POSITIVE_RATIO = 0.1  # Need at least 10% positive feedback
  ```
- No commented-out code; dead code removed

**JSDoc/TSDoc:**
- Python docstrings on classes and public methods
- Not exhaustive - relies on clear naming and type hints
- Example:
  ```python
  def analyze_feedback(
      self,
      generation: Generation,
      feedback: Feedback,
      profile: TasteProfile,
  ) -> dict:
      """
      Analyze feedback and suggest profile updates.

      Args:
          generation: The generation that received feedback
          feedback: The user's feedback
          profile: Current taste profile

      Returns:
          Analysis results with suggested updates
      """
  ```

## Function Design

**Size:**
- Preference for focused, single-responsibility functions
- CLI command functions often 50-100+ lines due to logging and UI orchestration
- Helper methods kept small (<30 lines)

**Parameters:**
- Explicit parameters preferred; dataclass objects group related data
- Optional parameters with defaults at end: `api_key: Optional[str] = None`
- Type hints on all parameters
- Click decorators handle CLI argument parsing

**Return Values:**
- Explicit returns; no implicit None returns
- Dataclass instances returned from loaders: `def load_profile(self) -> TasteProfile:`
- Tuple returns used for multi-value results with validation:
  ```python
  def can_calibrate(self) -> tuple[bool, str]:
      """Check if there's enough feedback to calibrate.

      Returns (can_calibrate, reason_if_not).
      """
  ```
- Dicts for flexible response structures (especially from Claude API):
  ```python
  def synthesize_from_history(...) -> dict:
      return {"analysis": "...", "updates": {...}}
  ```

## Module Design

**Exports:**
- All public classes/functions exportable from module
- No `__all__` directive used
- Base class at top of module, then helper dataclasses, then main implementation class

**Barrel Files:**
- Limited use of `__init__.py` files
- `generators/__init__.py` defines `ImageGenerator` base interface
- `core/__init__.py` minimal (just imports)
- Main package `__init__.py` exposes version only

**Class Organization Pattern:**
```python
@dataclass
class StylePreference:
    """Docstring."""

    # Fields
    preference: str = ""
    avoid: str = ""
    confidence: float = 0.0

    # Instance methods
    def to_dict(self) -> dict: ...

    # Class methods
    @classmethod
    def from_dict(cls, data: dict) -> "StylePreference": ...

    # Properties
    @property
    def is_preference_set(self) -> bool: ...
```

**Dataclasses:**
- Extensively used for data models: `@dataclass`
- Default factory for mutable defaults: `field(default_factory=dict)`
- Frozen not used (mutable profiles)
- Always include `to_dict()` and `from_dict()` for serialization

---

*Convention analysis: 2026-01-21*
