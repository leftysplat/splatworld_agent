"""
CLI for SplatWorld Agent.

This CLI is called by Claude Code slash commands to perform operations.
"""

import base64
import click
from datetime import datetime
from pathlib import Path
import json
import sys
import uuid

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich.progress import Progress, SpinnerColumn, TextColumn

from splatworld_agent import __version__
from splatworld_agent.config import Config, get_project_dir, GLOBAL_CONFIG_DIR
from splatworld_agent.profile import ProfileManager
from splatworld_agent.models import TasteProfile, Feedback, Generation
from splatworld_agent.learning import LearningEngine, enhance_prompt

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
@click.option("--no-splat", is_flag=True, help="Skip 3D splat generation")
@click.option("--generator", type=click.Choice(["nano", "gemini"]), default=None, help="Image generator to use")
def generate(prompt: tuple, seed: int, no_enhance: bool, no_splat: bool, generator: str):
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
        enhanced_prompt = enhance_prompt(prompt_text, profile)
        if enhanced_prompt != prompt_text:
            console.print(f"[dim]Original prompt:[/dim] {prompt_text}")
            console.print(f"[dim]Enhanced:[/dim] {enhanced_prompt}")

    console.print(f"\n[bold]Generating:[/bold] {enhanced_prompt}")

    # Create generation ID
    gen_id = f"gen-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"

    # Select image generator
    gen_name = generator or config.defaults.image_generator

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Step 1: Generate image
            task = progress.add_task(f"Generating image with {gen_name}...", total=None)

            if gen_name == "nano":
                from splatworld_agent.generators.nano import NanoGenerator
                img_gen = NanoGenerator(api_key=config.api_keys.nano or config.api_keys.google)
            else:
                from splatworld_agent.generators.gemini import GeminiGenerator
                img_gen = GeminiGenerator(api_key=config.api_keys.google)

            image_bytes = img_gen.generate(enhanced_prompt, seed=seed)
            img_gen.close()

            progress.update(task, description="[green]Image generated!")

            # Save the generation
            gen_dir = manager.save_generation(Generation(
                id=gen_id,
                prompt=prompt_text,
                enhanced_prompt=enhanced_prompt,
                timestamp=datetime.now(),
                metadata={"generator": gen_name, "seed": seed},
            ))

            # Save image
            image_path = gen_dir / "source.png"
            with open(image_path, "wb") as f:
                f.write(image_bytes)

            console.print(f"\n[green]Image saved:[/green] {image_path}")

            # Step 2: Convert to 3D splat (if requested)
            splat_path = None
            mesh_path = None

            if not no_splat and config.api_keys.marble:
                progress.update(task, description="Converting to 3D splat...")

                from splatworld_agent.core.marble import MarbleClient

                marble = MarbleClient(api_key=config.api_keys.marble)

                def on_progress(status: str, description: str):
                    progress.update(task, description=f"Marble: {description or status}")

                image_b64 = base64.b64encode(image_bytes).decode()
                result = marble.generate_and_wait(
                    image_base64=image_b64,
                    mime_type="image/png",
                    display_name=gen_id,
                    is_panorama=True,
                    on_progress=on_progress,
                )

                # Download splat file
                if result.splat_url and config.defaults.download_splats:
                    splat_path = gen_dir / "scene.spz"
                    marble.download_file(result.splat_url, splat_path)
                    console.print(f"[green]Splat saved:[/green] {splat_path}")

                # Download mesh file
                if result.mesh_url and config.defaults.download_meshes:
                    mesh_path = gen_dir / "collision.glb"
                    marble.download_file(result.mesh_url, mesh_path)
                    console.print(f"[green]Mesh saved:[/green] {mesh_path}")

                marble.close()

                console.print(f"[blue]Viewer:[/blue] {result.viewer_url}")
                console.print(f"[dim]Cost: ${result.cost_usd:.2f}[/dim]")

                progress.update(task, description="[green]3D conversion complete!")

            elif not no_splat and not config.api_keys.marble:
                console.print("[yellow]Skipping 3D conversion (no Marble API key configured)[/yellow]")

            # Update generation metadata with paths
            metadata_path = gen_dir / "metadata.json"
            with open(metadata_path) as f:
                gen_data = json.load(f)

            gen_data["source_image_path"] = str(image_path)
            if splat_path:
                gen_data["splat_path"] = str(splat_path)
            if mesh_path:
                gen_data["mesh_path"] = str(mesh_path)

            with open(metadata_path, "w") as f:
                json.dump(gen_data, f, indent=2)

        console.print(f"\n[bold green]Generation complete![/bold green]")
        console.print(f"[dim]ID: {gen_id}[/dim]")
        console.print("\nUse [cyan]splatworld-agent feedback ++[/cyan] to love it, or [cyan]-- [/cyan] to hate it.")

    except Exception as e:
        console.print(f"\n[red]Generation failed:[/red] {e}")
        sys.exit(1)


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
@click.option("--dry-run", is_flag=True, help="Show what would be learned without saving")
def learn(dry_run: bool):
    """Synthesize feedback into updated preferences."""
    project_dir = get_project_dir()
    if not project_dir:
        console.print("[red]Error: Not in a SplatWorld project.[/red]")
        sys.exit(1)

    manager = ProfileManager(project_dir.parent)
    config = Config.load()

    if not config.api_keys.anthropic:
        console.print("[red]Error: Anthropic API key required for learning. Set ANTHROPIC_API_KEY.[/red]")
        sys.exit(1)

    feedback_history = manager.get_feedback_history()

    if len(feedback_history) < 3:
        console.print("[yellow]Need at least 3 feedback entries to learn patterns.[/yellow]")
        console.print(f"[dim]Current feedback count: {len(feedback_history)}[/dim]")
        return

    # Get generations that have feedback
    generations = []
    for fb in feedback_history:
        gen = manager.get_generation(fb.generation_id)
        if gen:
            generations.append(gen)

    if not generations:
        console.print("[yellow]No generations found with feedback.[/yellow]")
        return

    console.print(f"Analyzing {len(feedback_history)} feedback entries from {len(generations)} generations...")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Analyzing patterns with Claude...", total=None)

            engine = LearningEngine(api_key=config.api_keys.anthropic)
            profile = manager.load_profile()

            result = engine.synthesize_from_history(generations, feedback_history, profile)

            progress.update(task, description="[green]Analysis complete!")

        # Show analysis
        console.print(f"\n[bold]Analysis:[/bold]")
        console.print(result.get("analysis", "No analysis provided"))

        # Show updates
        updates = result.get("updates", {})
        if not updates:
            console.print("\n[yellow]No preference updates identified.[/yellow]")
            return

        console.print(f"\n[bold]Suggested Updates:[/bold]")

        if "visual_style" in updates:
            console.print("\n[cyan]Visual Style:[/cyan]")
            for key, val in updates["visual_style"].items():
                if val.get("preference"):
                    console.print(f"  {key}: prefer '{val['preference']}'")
                if val.get("avoid"):
                    console.print(f"  {key}: avoid '{val['avoid']}'")

        if "composition" in updates:
            console.print("\n[cyan]Composition:[/cyan]")
            for key, val in updates["composition"].items():
                if val.get("preference"):
                    console.print(f"  {key}: prefer '{val['preference']}'")
                if val.get("avoid"):
                    console.print(f"  {key}: avoid '{val['avoid']}'")

        if "domain" in updates:
            console.print("\n[cyan]Domain:[/cyan]")
            dom = updates["domain"]
            if dom.get("environments", {}).get("add"):
                console.print(f"  Add environments: {dom['environments']['add']}")
            if dom.get("avoid_environments", {}).get("add"):
                console.print(f"  Avoid environments: {dom['avoid_environments']['add']}")

        if "quality" in updates:
            console.print("\n[cyan]Quality:[/cyan]")
            qual = updates["quality"]
            if qual.get("must_have", {}).get("add"):
                console.print(f"  Must have: {qual['must_have']['add']}")
            if qual.get("never", {}).get("add"):
                console.print(f"  Never: {qual['never']['add']}")

        if dry_run:
            console.print("\n[yellow]Dry run - no changes saved.[/yellow]")
        else:
            # Apply updates
            updated_profile = engine.apply_updates(profile, updates)
            manager.save_profile(updated_profile)
            console.print("\n[bold green]Profile updated![/bold green]")
            console.print("[dim]Use 'splatworld-agent profile' to view your updated taste profile.[/dim]")

    except Exception as e:
        console.print(f"\n[red]Learning failed:[/red] {e}")
        sys.exit(1)


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
