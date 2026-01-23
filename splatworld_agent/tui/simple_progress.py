"""Simple line-by-line progress display for non-interactive environments.

This module provides pretty progress output that works in Claude Code's bash
environment by printing new lines instead of doing in-place terminal updates.

All output goes to stderr to not interfere with JSON output on stdout.
"""

from rich.console import Console

# Progress console writes to stderr (visible to user, doesn't interfere with JSON)
_console = Console(stderr=True, force_terminal=True)


def print_header(prompt: str, provider: str = "nano"):
    """Print the generation header with prompt preview."""
    _console.print()
    _console.print("━" * 55, style="cyan")
    _console.print(" 🎨 SPLATWORLD DIRECT", style="bold cyan")
    _console.print("━" * 55, style="cyan")
    _console.print()

    # Truncate long prompts
    display_prompt = prompt[:50] + "..." if len(prompt) > 50 else prompt
    _console.print(f" [dim]Prompt:[/dim] {display_prompt}")
    _console.print(f" [dim]Provider:[/dim] {provider}")
    _console.print()


def print_stage(stage: str, icon: str, progress: int, status: str):
    """Print a single stage progress line.

    Args:
        stage: Stage name (e.g., "Enhance", "Generate", "Convert")
        icon: Emoji icon for the stage
        progress: Progress percentage (0-100)
        status: Status text to display
    """
    # Create progress bar (20 chars wide)
    filled = int(progress / 5)  # 100% = 20 blocks
    empty = 20 - filled
    bar = "█" * filled + "░" * empty

    # Status indicator
    if progress >= 100:
        indicator = "[green]✓[/green]"
    elif progress > 0:
        indicator = "[yellow]⏳[/yellow]"
    else:
        indicator = "[dim]○[/dim]"

    _console.print(f" {icon} {stage:<10} [cyan]{bar}[/cyan]  {progress:>3}%  {indicator} {status}")


def print_stage_start(stage: str, icon: str, status: str):
    """Print stage starting (0% progress)."""
    print_stage(stage, icon, 0, status)


def print_stage_progress(stage: str, icon: str, progress: int, status: str):
    """Print stage in progress."""
    print_stage(stage, icon, progress, status)


def print_stage_complete(stage: str, icon: str, status: str):
    """Print stage completion (100% progress)."""
    print_stage(stage, icon, 100, status)
    _console.print()  # Blank line between stages


def print_error(message: str):
    """Print an error message."""
    _console.print()
    _console.print("━" * 55, style="red")
    _console.print(" ❌ ERROR", style="bold red")
    _console.print("━" * 55, style="red")
    _console.print()
    _console.print(f" {message}", style="red")
    _console.print()


def print_partial_success(image_path: str, error: str):
    """Print partial success (image saved but 3D conversion failed)."""
    _console.print()
    _console.print("━" * 55, style="yellow")
    _console.print(" ⚠️  PARTIAL SUCCESS", style="bold yellow")
    _console.print("━" * 55, style="yellow")
    _console.print()
    _console.print(f" [green]✓[/green] Image saved: {image_path}")
    _console.print(f" [red]✗[/red] 3D conversion failed: {error}")
    _console.print()
    _console.print(" [dim]Run /splatworld:convert to retry 3D conversion[/dim]")
    _console.print()


def print_result(viewer_url: str, image_path: str, splat_path: str = None,
                 enhanced_prompt: str = None, reasoning: str = None):
    """Print the final success result.

    Args:
        viewer_url: The Marble viewer URL (main output)
        image_path: Path to saved image
        splat_path: Path to saved splat file (optional)
        enhanced_prompt: The enhanced prompt used (optional)
        reasoning: Reasoning for prompt modifications (optional)
    """
    _console.print()
    _console.print("━" * 55, style="green")
    _console.print(" ✨ GENERATION COMPLETE", style="bold green")
    _console.print("━" * 55, style="green")
    _console.print()

    # Viewer URL is the main output - make it prominent
    _console.print(f" [bold cyan]🌐 Viewer:[/bold cyan] {viewer_url}")
    _console.print()

    # File paths
    _console.print(f" [dim]📁 Image:[/dim]  {image_path}")
    if splat_path:
        _console.print(f" [dim]📁 Splat:[/dim]  {splat_path}")

    # Enhancement info
    if enhanced_prompt or reasoning:
        _console.print()
        _console.print(" [dim]─── Enhancement ───[/dim]")
        if reasoning:
            _console.print(f" [dim]{reasoning}[/dim]")

    _console.print()
