# Testing Patterns

**Analysis Date:** 2026-01-21

## Test Framework

**Runner:**
- pytest 7.0.0+
- pytest-asyncio 0.21.0+ for async support
- Config: `[tool.pytest.ini_options]` in `pyproject.toml`
- Async mode: `asyncio_mode = "auto"`
- Test paths: `tests/` directory
- Python versions: 3.10, 3.11, 3.12

**Assertion Library:**
- pytest built-in assertions (no special library)
- Simple `assert` statements with implicit message capture

**Run Commands:**
```bash
pytest tests/                 # Run all tests
pytest tests/ -v              # Verbose output
pytest tests/ --co            # Collect tests only
pytest tests/ -k "test_"      # Run specific pattern
pytest tests/ --tb=short      # Shorter traceback
```

## Test File Organization

**Location:**
- Test files co-located in `tests/` directory alongside source
- One test module per main module: `test_models.py` for `models.py`

**Naming:**
- Test files: `test_*.py` pattern
- Test classes: `Test*` naming convention (PascalCase)
- Test methods: `test_*` naming convention (snake_case)
- Test fixtures/helpers: descriptive function names

**Structure:**
```
tests/
├── __init__.py
└── test_models.py          # Tests for splatworld_agent/models.py
```

## Test Structure

**Suite Organization:**
Current test structure (from `tests/test_models.py`):

```python
class TestStylePreference:
    def test_to_dict(self):
        """Test serialization to dict."""
        pref = StylePreference(
            preference="moody lighting",
            avoid="flat lighting",
            confidence=0.8,
        )
        d = pref.to_dict()
        assert d["preference"] == "moody lighting"
        assert d["avoid"] == "flat lighting"
        assert d["confidence"] == 0.8

    def test_from_dict(self):
        """Test deserialization from dict."""
        d = {"preference": "warm", "avoid": "cold", "confidence": 0.5}
        pref = StylePreference.from_dict(d)
        assert pref.preference == "warm"
        assert pref.avoid == "cold"
        assert pref.confidence == 0.5


class TestTasteProfile:
    def test_empty_profile(self):
        """Verify empty profile initialization."""
        profile = TasteProfile()
        assert profile.version == "1.0"
        assert profile.stats.total_generations == 0

    def test_to_prompt_context_empty(self):
        """Empty profile produces empty context."""
        profile = TasteProfile()
        assert profile.to_prompt_context() == ""

    def test_to_prompt_context_with_preferences(self):
        """Profile with preferences generates context string."""
        profile = TasteProfile()
        profile.visual_style.lighting.preference = "moody, dramatic"
        profile.visual_style.color_palette.preference = "warm earth tones"
        profile.quality.must_have = ["realistic lighting"]

        context = profile.to_prompt_context()
        assert "moody, dramatic" in context
        assert "warm earth tones" in context
        assert "realistic lighting" in context

    def test_serialization_roundtrip(self):
        """Data survives to_dict/from_dict roundtrip."""
        profile = TasteProfile()
        profile.visual_style.lighting.preference = "test"
        profile.stats.total_generations = 5

        d = profile.to_dict()
        loaded = TasteProfile.from_dict(d)

        assert loaded.visual_style.lighting.preference == "test"
        assert loaded.stats.total_generations == 5
```

**Patterns:**
- Class-based organization (test classes group related tests)
- One logical test per method (clear test names)
- Arrange-act-assert pattern implicit (setup, execute, verify)
- Test docstrings describe what is being verified
- Setup done in method bodies (no fixtures yet)

## Mocking

**Framework:**
- No mocking library imported (unittest.mock available but not used)
- Real objects used in current tests

**What to Mock:**
- External APIs (Claude, Marble, Gemini, etc.) - NOT currently tested
- File I/O operations - NOT currently tested
- HTTP requests - Should use `httpx` mocking when added
- System time - `datetime.now()` should be mockable

**What NOT to Mock:**
- Internal dataclass models (test real serialization)
- Core logic in `TasteProfile`, `Feedback`, `Generation`
- Configuration loading from files (test real loading)

## Fixtures and Factories

**Test Data:**
Current pattern uses inline object construction:

```python
# In test method body
profile = TasteProfile()
profile.visual_style.lighting.preference = "moody, dramatic"
profile.stats.total_generations = 5

pref = StylePreference(
    preference="warm",
    avoid="cold",
    confidence=0.5,
)

gen = Generation(
    id="test-001",
    prompt="modern kitchen",
    enhanced_prompt="modern kitchen with warm lighting",
    timestamp=datetime.now(),
)
```

**Location:**
- No separate fixture file yet
- Fixtures should be added to `conftest.py` as coverage expands
- Factories can be added as helper functions in test modules

**Recommended Fixtures to Add:**
```python
# conftest.py
@pytest.fixture
def taste_profile():
    """Minimal empty taste profile."""
    return TasteProfile()

@pytest.fixture
def profile_with_prefs():
    """Profile with some preferences set."""
    profile = TasteProfile()
    profile.visual_style.lighting.preference = "moody"
    profile.stats.total_generations = 5
    return profile

@pytest.fixture
def sample_feedback():
    """Positive feedback example."""
    return Feedback(
        generation_id="gen-001",
        timestamp=datetime.now(),
        rating="++",
    )
```

## Coverage

**Requirements:**
- Not enforced in config
- No coverage target specified
- Should be added as codebase matures

**View Coverage:**
```bash
pip install pytest-cov
pytest tests/ --cov=splatworld_agent --cov-report=html
# Open htmlcov/index.html in browser
```

## Test Types

**Unit Tests:**
- Scope: Individual dataclass methods and properties
- Approach: Test serialization, validation, property calculations
- Examples in `test_models.py`:
  - `test_to_dict()` - serialization
  - `test_from_dict()` - deserialization
  - `test_quick_ratings()` - property boolean checks
  - `test_can_calibrate()` - business logic validation

**Integration Tests:**
- Scope: Profile manager with file I/O, learning engine with models
- Not yet implemented
- Should test:
  - `ProfileManager.load_profile()` + `save_profile()` roundtrip
  - `LearningEngine.apply_updates()` on real profile
  - CLI command workflows end-to-end

**E2E Tests:**
- Framework: Not yet implemented
- Would test: Full generate-feedback-learn cycle
- Blocked by: External API integration (Anthropic, image generators)
- Approach: Use mocks for external services, test orchestration logic

## Common Patterns

**Async Testing:**
Currently not needed - codebase uses async support in `learning.py` but tests don't require async yet.

When adding async tests:
```python
import pytest

class TestAsyncLearning:
    @pytest.mark.asyncio
    async def test_analyze_feedback_async(self):
        """Test async feedback analysis."""
        engine = LearningEngine(api_key="test-key")
        gen = Generation(...)
        fb = Feedback(...)
        profile = TasteProfile()

        result = await engine.analyze_feedback(gen, fb, profile)
        assert "analysis" in result
```

**Error Testing:**
Pattern for testing exceptions:

```python
def test_missing_profile_raises_error(tmp_path):
    """Load from non-existent path raises FileNotFoundError."""
    manager = ProfileManager(tmp_path)
    manager.splatworld_dir.mkdir()  # Dir exists, but no profile

    with pytest.raises(FileNotFoundError):
        manager.load_profile()
```

Example for testing API errors:

```python
def test_marble_auth_error(monkeypatch):
    """Invalid API key raises MarbleAuthError."""
    marble = MarbleClient(api_key="invalid")

    # Mock httpx response
    def mock_post(*args, **kwargs):
        response = Mock()
        response.status_code = 401
        response.json.return_value = {"error": "Invalid API key"}
        return response

    monkeypatch.setattr("httpx.Client.post", mock_post)

    with pytest.raises(MarbleAuthError):
        marble.make_request("POST", "/generate", {})
```

## Test Data Patterns

**Minimal Valid Data:**
```python
# Minimal generation
gen = Generation(
    id="test-001",
    prompt="test prompt",
    enhanced_prompt="test prompt",
    timestamp=datetime.now(),
)

# Minimal profile
profile = TasteProfile()

# Minimal feedback
fb = Feedback(
    generation_id="test-001",
    timestamp=datetime.now(),
    rating="++",
)
```

**Complete Data Example:**
```python
def create_profile_with_full_feedback():
    """Create profile with extensive preferences and history."""
    profile = TasteProfile()

    # Visual preferences
    profile.visual_style.lighting.preference = "moody, dramatic"
    profile.visual_style.lighting.avoid = "flat, washed out"
    profile.visual_style.color_palette.preference = "warm earth tones"
    profile.visual_style.mood.preference = "melancholic"

    # Composition
    profile.composition.density.preference = "sparse, minimal"
    profile.composition.perspective.preference = "wide angle"

    # Quality
    profile.quality.must_have = ["realistic", "high detail"]
    profile.quality.never = ["cartoonish", "low poly"]

    # Domain
    profile.domain.environments = ["indoor", "architectural"]
    profile.domain.avoid_environments = ["cartoon", "fantasy"]

    # Stats
    profile.stats.total_generations = 25
    profile.stats.feedback_count = 20
    profile.stats.love_count = 8
    profile.stats.like_count = 4
    profile.stats.dislike_count = 3
    profile.stats.hate_count = 5

    # Calibration
    profile.calibration.is_calibrated = True
    profile.calibration.learn_count = 3

    return profile
```

## Gaps to Address

**Not Yet Tested:**
- `cli.py` commands (requires Click testing + mock APIs)
- `profile.py` file operations (requires temporary directories)
- `learning.py` Claude API interaction (requires API mocking)
- `config.py` file loading/parsing (requires test files)
- Generator implementations (image generation APIs)
- Marble 3D conversion API (external service)

**Testing Strategy Moving Forward:**
1. Add `conftest.py` with common fixtures
2. Add integration tests for file operations with `tmp_path` fixture
3. Mock external APIs with `responses` or `httpx` mocking
4. Add CLI command tests with Click's testing utilities
5. Establish coverage baseline and target (e.g., >80%)

---

*Testing analysis: 2026-01-21*
