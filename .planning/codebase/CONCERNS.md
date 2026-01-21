# Codebase Concerns

**Analysis Date:** 2026-01-21

## Tech Debt

**Monolithic CLI module:**
- Issue: `cli.py` is 1,651 lines with 30+ commands, mixing concerns (setup, generation, learning, review, conversion, session management)
- Files: `splatworld_agent/cli.py`
- Impact: Difficult to maintain, test, and extend; unclear separation of responsibilities
- Fix approach: Refactor into separate modules (e.g., `commands/generate.py`, `commands/learn.py`) with shared utilities, or implement Click command groups with submodules

**JSON parsing from Claude responses with brittle string splitting:**
- Issue: `learning.py` uses regex string splitting to extract JSON from markdown code blocks instead of using robust JSON parsing
- Files: `splatworld_agent/learning.py` lines 115-120, 176-179
- Impact: Fragile to Claude response format changes; could fail silently if Claude includes code blocks in unexpected ways
- Fix approach: Use `json.JSONDecodeError` exception handling with fallback, or add a utility function to safely extract JSON from various markdown formats

**Unhandled API key fallback chain:**
- Issue: Multiple generators and clients fall back between different API key environment variable names without clear precedence
- Files: `splatworld_agent/generators/nano.py` line 38, `splatworld_agent/config.py` lines 45-53
- Impact: Confusion about which key is actually used; potential silent failures
- Fix approach: Document explicitly or use a single unified key resolution method

**Potential data loss on profile updates:**
- Issue: Profile is reloaded from disk multiple times during session without locking mechanism
- Files: `splatworld_agent/profile.py` (multiple calls to `load_profile()` in `cli.py`)
- Impact: Concurrent operations could cause feedback to be lost or statistics to be inconsistent
- Fix approach: Implement file locking or transaction-like semantics for profile updates

## Known Bugs

**API key embedded in URL query parameters:**
- Symptoms: API keys passed as URL query parameters which may be logged in HTTP client libraries or server logs
- Files: `splatworld_agent/generators/nano.py` line 76, `splatworld_agent/generators/gemini.py` line 68
- Trigger: Any call to `NanoGenerator.generate()` or `GeminiGenerator.generate()`
- Workaround: Keys logged to httpx internals; use httpx logging level configuration to suppress

**Missing null check after JSON parsing:**
- Symptoms: Code assumes `response.content[0].text` exists without checking
- Files: `splatworld_agent/learning.py` lines 112, 174
- Trigger: If Claude API returns response without text content (unlikely but possible)
- Workaround: Add explicit index bounds checking

**Metadata file read without error handling:**
- Symptoms: `cli.py` line 255 opens metadata.json with `open()` but catches generic `Exception`
- Files: `splatworld_agent/cli.py` lines 254-265, 441-444, 743-744
- Trigger: Metadata file corruption or deletion after generation saved
- Workaround: File is written immediately after saving, so window is small; but should use explicit `FileNotFoundError` handling

## Security Considerations

**API keys stored in plaintext YAML config:**
- Risk: User's API keys stored unencrypted in `~/.splatworld_agent/config.yaml`
- Files: `splatworld_agent/config.py` lines 102-123
- Current mitigation: File permissions (rely on OS file permissions), config file in user home directory
- Recommendations:
  1. Use system keychain integration (macOS Keychain, Linux Secret Service, Windows Credential Manager)
  2. Document security implications clearly in README
  3. Add permission check on config file (warn if readable by others)
  4. Support reading keys only from environment variables (skip file config) for secure deployments

**API keys in subprocess environment:**
- Risk: Keys passed to subprocess calls visible in process listings
- Files: `splatworld_agent/cli.py` lines 1361-1366 (subprocess.run for git operations)
- Current mitigation: Keys not used in git subprocess, so low risk
- Recommendations: Maintain practice of not passing secrets to subprocesses; document this pattern

**Image content not validated before sending to APIs:**
- Risk: Arbitrary files could be sent to external APIs if user path argument is manipulated
- Files: `splatworld_agent/cli.py` lines 768, 710
- Current mitigation: Click's `type=click.Path(exists=True)` validates existence
- Recommendations: Add MIME type validation to ensure only image files are processed

## Performance Bottlenecks

**Linear scan of generations directory for every operation:**
- Problem: Finding generations requires iterating through all date directories and files
- Files: `splatworld_agent/profile.py` lines 211-235, 198-209
- Cause: Flat JSON file structure; no indexing; generation IDs scattered across dated directories
- Improvement path:
  1. Build in-memory index of generation IDs on ProfileManager init (cache)
  2. Consider SQLite database for larger projects
  3. Add generation metadata cache file listing all IDs with timestamps

**Claude API calls not batched:**
- Problem: `learn` command calls Claude for every feedback when analyzing history would be more efficient
- Files: `splatworld_agent/learning.py` lines 122-181
- Cause: Design uses single-feedback analysis in some paths
- Improvement path: Batch feedback analysis by collecting 10+ items before calling Claude

**Marble API polling without exponential backoff:**
- Problem: `generate_and_wait()` may poll at fixed intervals burning API quota
- Files: `splatworld_agent/core/marble.py` (not shown in excerpt but referenced in cli.py)
- Cause: Fixed 2-3 second polling interval while waiting for conversion
- Improvement path: Implement exponential backoff with jitter for polling operations

## Fragile Areas

**Session management state machine:**
- Files: `splatworld_agent/profile.py` lines 244-318, `splatworld_agent/cli.py` lines 1460-1647
- Why fragile: Session state tracked in separate JSON file that can be orphaned; no guarantees of atomic transitions
- Safe modification: Add validation that current_session file exists before operations; implement cleanup of stale sessions on startup
- Test coverage: Minimal testing of session lifecycle; `test_models.py` only tests model serialization

**Feedback-to-generation mapping assumptions:**
- Files: `splatworld_agent/learning.py` lines 140-151, `splatworld_agent/cli.py` lines 642-657
- Why fragile: Code assumes all feedbacks have matching generations; orphaned feedback would cause key errors
- Safe modification: Always check `if gen: ...` before accessing generation properties; log warnings for unmatched feedback
- Test coverage: No tests for missing generations; no tests for feedback without corresponding generation

**Profile stats manual increment:**
- Files: `splatworld_agent/profile.py` lines 98-111, `splatworld_agent/cli.py` lines 193-194
- Why fragile: Stats incremented separately from feedback addition; could become inconsistent if add_feedback fails partway through
- Safe modification: Move all stat updates into atomic transaction; consider making ProfileManager the single source of truth for stats
- Test coverage: No tests of stat consistency; no tests of concurrent profile access

## Scaling Limits

**Feedback and generation files are append-only:**
- Current capacity: Tested up to reasonable session size (50-100 generations); untested beyond
- Limit: Reading entire feedback.jsonl into memory every operation; O(n) performance; potential memory issues with >10k feedback entries
- Scaling path: Migrate to SQLite database or implement pagination/chunking in ProfileManager

**Marble API rate limiting not exposed:**
- Current capacity: Unknown; API key limits apply but not surfaced to user
- Limit: Cost tracking added but no quota checking; user might hit API limits unexpectedly
- Scaling path: Add rate limiting awareness to config; track API calls; warn user of quota exhaustion

## Dependencies at Risk

**Anthropic SDK version pinning:**
- Risk: `pyproject.toml` uses `anthropic>=0.27.0` which is loose; major version bumps could break due to API changes
- Impact: Claude model name might change (currently hardcoded `claude-sonnet-4-20250514`)
- Migration plan: Pin to major version range (e.g., `>=0.27.0,<1.0`); test upgrade path before release; add deprecation handling for model versions

**World Labs Marble API stability:**
- Risk: External API with no uptime SLA documented; format could change
- Impact: Conversions could fail; if format changes, parsing breaks
- Migration plan: Add feature flag for API format versions; implement versioned response parsers; monitor API status page

**Python 3.10 only for type hints:**
- Risk: Uses `list[Type]` syntax requiring Python 3.10+; blocks Python 3.9 compatibility
- Impact: Some users on older Python versions cannot use this tool
- Migration plan: Either drop 3.9 support officially or use `from __future__ import annotations` and `List` from typing for compatibility

## Missing Critical Features

**No network error recovery in main workflows:**
- Problem: Long-running commands (batch generation, train, convert) have no pause/resume capability if network fails
- Blocks: Cannot reliably use over poor connections; no idempotency for failed operations
- Gap: Session management exists but not integrated with recovery

**No local image preview in interactive mode:**
- Problem: `review` command tries to `open()` files but on headless systems this fails silently
- Blocks: Users on SSH/headless environments cannot review images interactively
- Gap: No fallback to ANSI art or terminal rendering; no check for X11 availability

**No validation of prompt inputs:**
- Problem: User prompts sent directly to APIs without content moderation
- Blocks: Could send harmful content; no guardrails in place
- Gap: Should add content policy enforcement or warnings before API calls

## Test Coverage Gaps

**No integration tests for CLI commands:**
- What's not tested: Full workflows (generate → feedback → learn → convert); CLI argument parsing; error handling paths
- Files: `splatworld_agent/cli.py` (entire module), `tests/test_models.py` (only 99 lines covering models)
- Risk: Regressions in command behavior; broken CLI flags discovered only after release
- Priority: High - CLI is primary user interface

**No tests for API client error handling:**
- What's not tested: Marble API retry logic; rate limiting; timeout scenarios; malformed responses
- Files: `splatworld_agent/core/marble.py`, `splatworld_agent/generators/*.py`
- Risk: Silent failures; error messages might be unhelpful to users; retry logic never validated
- Priority: High - API failures are common in production

**No tests for profile data integrity:**
- What's not tested: Concurrent access; stats consistency; orphaned feedback handling; session cleanup
- Files: `splatworld_agent/profile.py`, `splatworld_agent/models.py`
- Risk: Silent data corruption; inconsistent stats; orphaned sessions
- Priority: Medium - affects reliability over time

**No tests for learning engine:**
- What's not tested: Claude integration; JSON parsing from responses; update application; profile convergence
- Files: `splatworld_agent/learning.py` (316 lines, 0% coverage)
- Risk: Learning might produce malformed profiles; updates could corrupt preferences
- Priority: High - core feature for taste learning

**No tests for exemplar/anti-exemplar features:**
- What's not tested: Image file copying; profile serialization with exemplars; reference inclusion in prompts
- Files: `splatworld_agent/profile.py` lines 136-174, `splatworld_agent/cli.py` lines 770-800
- Risk: Exemplars might not be used in prompt enhancement; file operations could fail
- Priority: Medium - feature partially implemented

---

*Concerns audit: 2026-01-21*
