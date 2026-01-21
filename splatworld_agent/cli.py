"""
CLI for SplatWorld Agent.

This CLI is called by Claude Code slash commands to perform operations.
"""

import click
from datetime import datetime
from pathlib import Path
import json
import sys

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax

from splatworld_agent import __version__
from splatworld_agent.config import Config, get_project_dir, GLOBAL_CONFIG_DIR
from splatworld_agent.profile import ProfileManager
from splatworld_agent.models import TasteProfile, Feedback

console = Console()


@click.group()
@click.version_option(version=__version__)
def main():
    """SplatWorld Agent - Iterative 3D splat generation with taste learning."""
    pass


@main.command()
@click.option("--path", type=click.Path(), default=".", help="Project path to initialize")
def init(path: str):
    """Initialize .splatworld/ in a project directory."""
    project_dir = Path(path).resolve()

    manager = ProfileManager(project_dir)

    if manager.is_initialized():
        console.print(f"[yellow]Project already initialized at {manager.splatworld_dir}[/yellow]")
        return

    profile = manager.initialize()

    console.print(Panel.fit(
        f"[green]Initialized SplatWorld Agent[/green]\n\n"
        f"Created: {manager.splatworld_dir}\n"
        f"Profile: {manager.profile_path}\n\n"
        f"Your taste profile is empty. Generate some content and provide feedback to teach it your preferences.",
        title="SplatWorld Agent",
    ))


@main.command()
@click.argument("prompt", nargs=-1, required=True)
@click.option("--seed", type=int, help="Random seed for reproducibility")
@click.option("--no-enhance", is_flag=True, help="Don't enhance prompt with taste profile")
def generate(prompt: tuple, seed: int, no_enhance: bool):
    """Generate image and splat from a prompt."""
    prompt_text = " ".join(prompt)

    project_dir = get_project_dir()
    if not project_dir:
        console.print("[red]Error: Not in a SplatWorld project. Run 'splatworld-agent init' first.[/red]")
        sys.exit(1)

    manager = ProfileManager(project_dir.parent)
    config = Config.load()

    # Validate config
    issues = config.validate()
    if issues:
        console.print("[red]Configuration issues:[/red]")
        for issue in issues:
            console.print(f"  - {issue}")
        sys.exit(1)

    profile = manager.load_profile()

    # Enhance prompt with taste profile
    enhanced_prompt = prompt_text
    if not no_enhance:
        taste_context = profile.to_prompt_context()
        if taste_context:
            enhanced_prompt = f"{prompt_text}. {taste_context}"
            console.print(f"[dim]Original prompt:[/dim] {prompt_text}")
            console.print(f"[dim]Enhanced with taste:[/dim] {taste_context}")

    console.print(f"\n[bold]Generating:[/bold] {enhanced_prompt}")
    console.print(f"[dim]Seed: {seed or 'random'}[/dim]")

    # TODO: Implement actual generation
    # 1. Call image generator (Nano/Gemini)
    # 2. Call Marble API for 3D conversion
    # 3. Download and save assets
    # 4. Create Generation record

    console.print("\n[yellow]Generation not yet implemented. Core structure ready.[/yellow]")


@main.command()
@click.argument("feedback_text", nargs=-1)
@click.option("--generation", "-g", help="Generation ID (defaults to last)")
def feedback(feedback_text: tuple, generation: str):
    """Provide feedback on a generation."""
    text = " ".join(feedback_text) if feedback_text else ""

    project_dir = get_project_dir()
    if not project_dir:
        console.print("[red]Error: Not in a SplatWorld project.[/red]")
        sys.exit(1)

    manager = ProfileManager(project_dir.parent)

    # Determine rating from text
    rating = "text"
    if text in ("++", "love", "love it", "perfect"):
        rating = "++"
    elif text in ("+", "good", "nice", "like"):
        rating = "+"
    elif text in ("-", "meh", "okay", "not great"):
        rating = "-"
    elif text in ("--", "hate", "hate it", "bad", "terrible"):
        rating = "--"

    # Get generation ID
    if not generation:
        last_gen = manager.get_last_generation()
        if not last_gen:
            console.print("[red]No generations found. Generate something first.[/red]")
            sys.exit(1)
        generation = last_gen.id

    # Create and save feedback
    fb = Feedback(
        generation_id=generation,
        timestamp=datetime.now(),
        rating=rating,
        text=text if rating == "text" else "",
    )

    manager.add_feedback(fb)

    rating_display = {
        "++": "[green]Love it![/green]",
        "+": "[green]Good[/green]",
        "-": "[yellow]Not great[/yellow]",
        "--": "[red]Hate it[/red]",
        "text": f"[blue]{text}[/blue]",
    }

    console.print(f"Feedback recorded: {rating_display[rating]}")
    console.print(f"[dim]Generation: {generation}[/dim]")

    # Check if we should suggest learning
    profile = manager.load_profile()
    config = Config.load()
    unprocessed = len(manager.get_unprocessed_feedback())
    if unprocessed >= config.defaults.auto_learn_threshold:
        console.print(f"\n[cyan]You have {unprocessed} feedback entries. Consider running 'splatworld-agent learn' to update your taste profile.[/cyan]")


@main.command()
@click.argument("image_path", type=click.Path(exists=True))
@click.option("--notes", "-n", default="", help="Notes about why you like this")
def exemplar(image_path: str, notes: str):
    """Add an exemplar image to your taste profile."""
    project_dir = get_project_dir()
    if not project_dir:
        console.print("[red]Error: Not in a SplatWorld project.[/red]")
        sys.exit(1)

    manager = ProfileManager(project_dir.parent)
    ex = manager.add_exemplar(Path(image_path), notes)

    console.print(f"[green]Added exemplar:[/green] {ex.path}")
    if notes:
        console.print(f"[dim]Notes: {notes}[/dim]")


@main.command("anti-exemplar")
@click.argument("image_path", type=click.Path(exists=True))
@click.option("--notes", "-n", default="", help="Notes about why you dislike this")
def anti_exemplar(image_path: str, notes: str):
    """Add an anti-exemplar image (things you never want)."""
    project_dir = get_project_dir()
    if not project_dir:
        console.print("[red]Error: Not in a SplatWorld project.[/red]")
        sys.exit(1)

    manager = ProfileManager(project_dir.parent)
    ex = manager.add_anti_exemplar(Path(image_path), notes)

    console.print(f"[green]Added anti-exemplar:[/green] {ex.path}")
    if notes:
        console.print(f"[dim]Notes: {notes}[/dim]")


@main.command()
@click.option("--edit", is_flag=True, help="Open profile for editing")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def profile(edit: bool, as_json: bool):
    """View or edit your taste profile."""
    project_dir = get_project_dir()
    if not project_dir:
        console.print("[red]Error: Not in a SplatWorld project.[/red]")
        sys.exit(1)

    manager = ProfileManager(project_dir.parent)
    prof = manager.load_profile()

    if as_json:
        console.print(json.dumps(prof.to_dict(), indent=2))
        return

    if edit:
        import subprocess
        editor = "code" if Path("/usr/local/bin/code").exists() else "nano"
        subprocess.run([editor, str(manager.profile_path)])
        return

    # Display profile nicely
    console.print(Panel.fit(
        f"[bold]Taste Profile[/bold]\n"
        f"Created: {prof.created.strftime('%Y-%m-%d')}\n"
        f"Updated: {prof.updated.strftime('%Y-%m-%d %H:%M')}\n"
        f"Generations: {prof.stats.total_generations}\n"
        f"Feedback: {prof.stats.feedback_count} ({prof.stats.love_count} loves, {prof.stats.hate_count} hates)",
        title="SplatWorld Agent",
    ))

    # Visual style
    if any([prof.visual_style.lighting.preference,
            prof.visual_style.color_palette.preference,
            prof.visual_style.mood.preference]):
        console.print("\n[bold]Visual Style[/bold]")
        if prof.visual_style.lighting.preference:
            console.print(f"  Lighting: {prof.visual_style.lighting.preference}")
        if prof.visual_style.color_palette.preference:
            console.print(f"  Colors: {prof.visual_style.color_palette.preference}")
        if prof.visual_style.mood.preference:
            console.print(f"  Mood: {prof.visual_style.mood.preference}")

    # Composition
    if any([prof.composition.density.preference,
            prof.composition.perspective.preference,
            prof.composition.foreground.preference]):
        console.print("\n[bold]Composition[/bold]")
        if prof.composition.density.preference:
            console.print(f"  Density: {prof.composition.density.preference}")
        if prof.composition.perspective.preference:
            console.print(f"  Perspective: {prof.composition.perspective.preference}")
        if prof.composition.foreground.preference:
            console.print(f"  Foreground: {prof.composition.foreground.preference}")

    # Domain
    if prof.domain.environments:
        console.print(f"\n[bold]Preferred Environments[/bold]: {', '.join(prof.domain.environments)}")
    if prof.domain.avoid_environments:
        console.print(f"[bold]Avoid Environments[/bold]: {', '.join(prof.domain.avoid_environments)}")

    # Quality
    if prof.quality.must_have:
        console.print(f"\n[bold]Must Have[/bold]: {', '.join(prof.quality.must_have)}")
    if prof.quality.never:
        console.print(f"[bold]Never[/bold]: {', '.join(prof.quality.never)}")

    # Exemplars
    if prof.exemplars:
        console.print(f"\n[bold]Exemplars[/bold]: {len(prof.exemplars)} images")
    if prof.anti_exemplars:
        console.print(f"[bold]Anti-Exemplars[/bold]: {len(prof.anti_exemplars)} images")

    # Prompt context preview
    context = prof.to_prompt_context()
    if context:
        console.print(f"\n[dim]Prompt enhancement preview:[/dim]")
        console.print(f"[cyan]{context}[/cyan]")
    else:
        console.print("\n[dim]Profile is empty. Generate content and provide feedback to build preferences.[/dim]")


@main.command()
@click.option("--limit", "-n", default=10, help="Number of generations to show")
def history(limit: int):
    """Browse past generations."""
    project_dir = get_project_dir()
    if not project_dir:
        console.print("[red]Error: Not in a SplatWorld project.[/red]")
        sys.exit(1)

    manager = ProfileManager(project_dir.parent)
    generations = manager.get_recent_generations(limit=limit)

    if not generations:
        console.print("[dim]No generations yet.[/dim]")
        return

    table = Table(title=f"Recent Generations (last {limit})")
    table.add_column("ID", style="cyan")
    table.add_column("Date")
    table.add_column("Prompt")
    table.add_column("Feedback")

    for gen in generations:
        feedback_str = ""
        if gen.feedback:
            feedback_str = gen.feedback.rating
            if gen.feedback.rating == "text":
                feedback_str = gen.feedback.text[:20] + "..."

        table.add_row(
            gen.id,
            gen.timestamp.strftime("%m-%d %H:%M"),
            gen.prompt[:40] + ("..." if len(gen.prompt) > 40 else ""),
            feedback_str,
        )

    console.print(table)


@main.command()
def learn():
    """Synthesize feedback into updated preferences."""
    project_dir = get_project_dir()
    if not project_dir:
        console.print("[red]Error: Not in a SplatWorld project.[/red]")
        sys.exit(1)

    manager = ProfileManager(project_dir.parent)
    feedback_history = manager.get_feedback_history()

    if len(feedback_history) < 3:
        console.print("[yellow]Need at least 3 feedback entries to learn patterns.[/yellow]")
        return

    console.print(f"Analyzing {len(feedback_history)} feedback entries...")

    # TODO: Implement actual learning
    # 1. Load Claude API
    # 2. Send feedback history for analysis
    # 3. Extract preference patterns
    # 4. Update profile

    console.print("\n[yellow]Learning not yet implemented. Core structure ready.[/yellow]")


@main.command("install-prompts")
def install_prompts():
    """Install Claude Code slash command prompts."""
    prompts_dir = Path(__file__).parent.parent / "prompts"
    target_dir = Path.home() / ".claude" / "splatworld-agent"

    if not prompts_dir.exists():
        console.print("[red]Prompts directory not found in package.[/red]")
        sys.exit(1)

    target_dir.mkdir(parents=True, exist_ok=True)

    import shutil
    for prompt_file in prompts_dir.glob("*.md"):
        shutil.copy2(prompt_file, target_dir / prompt_file.name)
        console.print(f"Installed: {prompt_file.name}")

    console.print(f"\n[green]Prompts installed to {target_dir}[/green]")
    console.print("You can now use /splatworld-agent:* commands in Claude Code.")


@main.command()
def config():
    """View or edit configuration."""
    cfg = Config.load()

    console.print(Panel.fit(
        f"[bold]Configuration[/bold]\n\n"
        f"Config file: {GLOBAL_CONFIG_DIR / 'config.yaml'}\n\n"
        f"[bold]API Keys[/bold]\n"
        f"  Marble: {'[green]configured[/green]' if cfg.api_keys.marble else '[red]missing[/red]'}\n"
        f"  Nano: {'[green]configured[/green]' if cfg.api_keys.nano else '[yellow]missing[/yellow]'}\n"
        f"  Google: {'[green]configured[/green]' if cfg.api_keys.google else '[yellow]missing[/yellow]'}\n"
        f"  Anthropic: {'[green]configured[/green]' if cfg.api_keys.anthropic else '[red]missing[/red]'}\n\n"
        f"[bold]Defaults[/bold]\n"
        f"  Image generator: {cfg.defaults.image_generator}\n"
        f"  Auto-learn threshold: {cfg.defaults.auto_learn_threshold}\n"
        f"  Download splats: {cfg.defaults.download_splats}\n"
        f"  Download meshes: {cfg.defaults.download_meshes}",
        title="SplatWorld Agent Config",
    ))

    issues = cfg.validate()
    if issues:
        console.print("\n[red]Issues:[/red]")
        for issue in issues:
            console.print(f"  - {issue}")


@main.command()
def help():
    """Show help and available commands."""
    console.print(Panel.fit(
        "[bold]SplatWorld Agent[/bold]\n"
        "Iterative 3D splat generation with taste learning.\n\n"
        "[bold]Commands:[/bold]\n"
        "  init           Initialize .splatworld/ in current project\n"
        "  generate       Generate image + splat from prompt\n"
        "  feedback       Rate/critique a generation\n"
        "  exemplar       Add reference image you love\n"
        "  anti-exemplar  Add reference image you hate\n"
        "  profile        View/edit taste profile\n"
        "  history        Browse past generations\n"
        "  learn          Synthesize feedback into preferences\n"
        "  config         View/edit configuration\n"
        "  install-prompts  Install Claude Code slash commands\n\n"
        "[dim]Use 'splatworld-agent COMMAND --help' for command details.[/dim]",
        title="SplatWorld Agent",
    ))


if __name__ == "__main__":
    main()
