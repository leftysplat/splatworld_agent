"""Custom TUI widgets for progress display.

This module provides Textual widgets for multi-stage progress visualization
with elapsed time tracking and status icons.
"""
from time import perf_counter
from typing import Optional

from textual.reactive import reactive
from textual.widgets import RichLog, Static


class StageProgress(Static):
    """Multi-stage progress display with elapsed time and status icons.

    Displays progress through multiple stages with:
    - Status icons (waiting, running, complete, error)
    - Stage number and description
    - Elapsed time per stage that updates every second

    Example output:
        ✓ Stage 1/3: Prompt enhanced (2s)
        ● Stage 2/3: Generating with Nano... (5s)
        ⏸ Stage 3/3:
    """

    # Status icons using Rich markup
    ICONS = {
        "waiting": "[dim]⏸[/dim]",      # pause symbol
        "running": "[cyan]●[/cyan]",     # dot (spinner effect with refresh)
        "complete": "[green]✓[/green]",  # checkmark
        "error": "[red]✗[/red]",         # X mark
    }

    def __init__(self, total_stages: int = 3, id: Optional[str] = None):
        """Initialize StageProgress widget.

        Args:
            total_stages: Number of stages to track (default 3)
            id: Widget ID for querying
        """
        super().__init__(id=id)
        self.total_stages = total_stages
        self._stages: dict[int, dict] = {}
        self._current_stage = 0
        self._elapsed_timer = None
        self._overall_start = None
        self._completed_stages = 0

        # Initialize all stages as waiting
        for i in range(1, total_stages + 1):
            self._stages[i] = {
                "status": "waiting",
                "description": "",
                "start_time": None,
                "end_time": None,
            }

    def on_mount(self):
        """Start elapsed time timer when widget mounts."""
        self._elapsed_timer = self.set_interval(1, self._refresh_display)

    def set_stage(self, stage: int, status: str, description: str = ""):
        """Update a stage's status.

        Args:
            stage: Stage number (1-indexed)
            status: One of "waiting", "running", "complete", "error"
            description: Optional description text for the stage
        """
        if stage not in self._stages:
            return

        stage_data = self._stages[stage]
        stage_data["status"] = status
        stage_data["description"] = description

        if status == "running":
            stage_data["start_time"] = perf_counter()
            self._current_stage = stage
            # Track overall start on first stage
            if self._overall_start is None:
                self._overall_start = perf_counter()
        elif status in ("complete", "error"):
            stage_data["end_time"] = perf_counter()
            # Track completion for ETA calculation
            if status == "complete":
                self._completed_stages = stage

        self.refresh()

    def stop_timer(self):
        """Stop the elapsed time timer."""
        if self._elapsed_timer:
            self._elapsed_timer.pause()

    def _refresh_display(self):
        """Called every second to update elapsed times."""
        self.refresh()

    def _calculate_eta(self) -> str:
        """Calculate estimated time remaining using linear extrapolation.

        Formula: remaining = (elapsed / completed_fraction) - elapsed

        Returns empty string until at least one stage completes to avoid
        division by zero and misleading early estimates.
        """
        # Guard: Don't calculate until we have completion data
        if self._overall_start is None or self._completed_stages == 0:
            return ""

        elapsed = perf_counter() - self._overall_start
        progress_fraction = self._completed_stages / self.total_stages

        # Linear extrapolation: total_time = elapsed / fraction_done
        total_estimated = elapsed / progress_fraction
        remaining = total_estimated - elapsed

        # Guard: Don't show negative or very small remaining time
        if remaining < 1:
            return ""

        # Format as minutes:seconds
        mins, secs = divmod(int(remaining), 60)
        if mins > 0:
            return f"ETA: {mins}m {secs}s"
        return f"ETA: {secs}s"

    def render(self) -> str:
        """Render all stages with status and elapsed time."""
        lines = []
        for stage_num in range(1, self.total_stages + 1):
            stage = self._stages[stage_num]
            icon = self.ICONS.get(stage["status"], " ")

            # Calculate elapsed time
            elapsed = ""
            if stage["start_time"]:
                end = stage["end_time"] or perf_counter()
                secs = int(end - stage["start_time"])
                elapsed = f" ({secs}s)"

            # Build line: [icon] Stage N/M: [description] (Xs)
            line = f"{icon} Stage {stage_num}/{self.total_stages}"
            if stage["description"]:
                line += f": {stage['description']}"
            line += elapsed

            lines.append(line)

        # Add ETA if available (only after first stage completes)
        eta = self._calculate_eta()
        if eta:
            lines.append(f"[dim]{eta}[/dim]")

        return "\n".join(lines)


class ResourcePanel(Static):
    """Live resource counter display with credit warning.

    Displays:
    - API calls counter: "API calls: 3"
    - Credit usage: "Credits: 5/10 (50%)" or warning at 75%+

    Requirements: RES-01, RES-02
    """

    api_calls = reactive(0)
    credits_used = reactive(0)
    credits_limit: reactive[Optional[int]] = reactive(None)

    WARNING_THRESHOLD = 75.0

    def render(self) -> str:
        parts = [f"API calls: {self.api_calls}"]

        # Credit display with optional warning
        if self.credits_limit is not None and self.credits_limit > 0:
            pct = (self.credits_used / self.credits_limit) * 100
            credit_str = f"{self.credits_used}/{self.credits_limit} ({pct:.0f}%)"

            if pct >= self.WARNING_THRESHOLD:
                parts.append(f"[yellow bold]Credits: {credit_str} - Consider Gemini[/yellow bold]")
            else:
                parts.append(f"Credits: {credit_str}")
        elif self.credits_used > 0:
            parts.append(f"Credits: {self.credits_used}")

        return " | ".join(parts)


class OutputLog(RichLog):
    """Scrolling log for file paths and URLs as they become available.

    Shows entries like:
    - Saved: generated_images/42.png
    - Viewer: https://marble.worldlabs.ai/world/abc123

    Requirements: OUT-01, OUT-02
    """

    def __init__(self, **kwargs):
        # Set defaults but allow override
        kwargs.setdefault("markup", True)
        kwargs.setdefault("highlight", True)
        kwargs.setdefault("auto_scroll", True)
        kwargs.setdefault("max_lines", 10)
        kwargs.setdefault("min_width", 40)
        super().__init__(**kwargs)

    def log_saved(self, path: str) -> None:
        """Log a saved file path with green prefix."""
        self.write(f"[green]Saved:[/green] {path}")

    def log_url(self, label: str, url: str) -> None:
        """Log a URL with blue label prefix."""
        self.write(f"[blue]{label}:[/blue] {url}")
