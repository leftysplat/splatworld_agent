"""
CLI for SplatWorld Agent.

This CLI is called by Claude Code slash commands to perform operations.
"""

import base64
import click
from datetime import datetime
from pathlib import Path
import json
import re
import sys
import uuid

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich.progress import Progress, SpinnerColumn, TextColumn

from splatworld_agent import __version__
from splatworld_agent.config import Config, get_project_dir, GLOBAL_CONFIG_DIR, GLOBAL_CONFIG_FILE
from splatworld_agent.profile import ProfileManager
from splatworld_agent.models import TasteProfile, Feedback, Generation
from splatworld_agent.learning import LearningEngine, enhance_prompt
from splatworld_agent.display import display

console = Console()


def parse_batch_ratings(input_str: str) -> list[tuple[int, str]]:
    """Parse '1 ++ 2 - 3 -- 4 +' into [(1, '++'), (2, '-'), (3, '--'), (4, '+')].

    Handles:
    - Spaces between number and rating: "1 ++"
    - No spaces: "1++"
    - Mixed: "1++ 2 -"

    Returns list of (image_number, rating) tuples.
    """
    # CRITICAL: Check ++ and -- before + and - (longer match first)
    pattern = r'(\d+)\s*(\+\+|--|\+|-)'
    matches = re.findall(pattern, input_str)
    return [(int(num), rating) for num, rating in matches]


def format_batch_context(batch_context: dict) -> str:
    """Format batch context for display in review."""
    if not batch_context.get("batch_id"):
        return "[dim]Single generation (no batch)[/dim]"

    batch_id = batch_context["batch_id"]

    # Extract date from batch_id: "batch-20260121-143522" -> "Jan 21 14:35"
    try:
        parts = batch_id.split("-")
        if len(parts) >= 3:
            date_part = parts[1]  # "20260121"
            time_part = parts[2]  # "143522"
            batch_dt = datetime.strptime(date_part + time_part, "%Y%m%d%H%M%S")
            readable = batch_dt.strftime("%b %d %H:%M")
        else:
            readable = batch_id
    except (ValueError, IndexError):
        readable = batch_id

    img_num = batch_context["batch_index"] + 1  # 1-indexed for display
    batch_size = batch_context["batch_size"]

    return f"[cyan]Batch {readable}[/cyan] - Image [bold]{img_num}[/bold] of {batch_size}"


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


@main.command("setup-keys")
@click.option("--anthropic", "anthropic_key", help="Anthropic API key")
@click.option("--google", "google_key", help="Google API key")
@click.option("--worldlabs", "worldlabs_key", help="World Labs (Marble) API key")
@click.option("--nano", "nano_key", help="Nano Banana API key")
def setup_keys(anthropic_key: str, google_key: str, worldlabs_key: str, nano_key: str):
    """Configure API keys for SplatWorld Agent."""
    cfg = Config.load()

    # Update keys if provided
    if anthropic_key:
        cfg.api_keys.anthropic = anthropic_key
    if google_key:
        cfg.api_keys.google = google_key
        # Google key is also used for Nano Banana Pro
        cfg.api_keys.nano = google_key
    if worldlabs_key:
        cfg.api_keys.marble = worldlabs_key
    if nano_key:
        cfg.api_keys.nano = nano_key

    # Set default image generator to nano (Nano Banana Pro)
    cfg.defaults.image_generator = "nano"

    # Save config
    cfg.save()

    console.print(f"[green]API keys saved to {GLOBAL_CONFIG_FILE}[/green]")

    # Show status
    console.print("\n[bold]API Key Status:[/bold]")
    console.print(f"  Anthropic: {'[green]configured[/green]' if cfg.api_keys.anthropic else '[red]missing[/red]'}")
    console.print(f"  Google/Nano Banana Pro: {'[green]configured[/green]' if cfg.api_keys.nano else '[red]missing[/red]'}")
    console.print(f"  World Labs (Marble): {'[green]configured[/green]' if cfg.api_keys.marble else '[red]missing[/red]'}")


@main.command("check-keys")
def check_keys():
    """Check API key configuration status."""
    cfg = Config.load()
    issues = cfg.validate()

    console.print("[bold]API Key Status:[/bold]")
    console.print(f"  Anthropic: {'[green]configured[/green]' if cfg.api_keys.anthropic else '[red]missing[/red]'}")
    console.print(f"  Google/Nano Banana Pro: {'[green]configured[/green]' if cfg.api_keys.nano else '[red]missing[/red]'}")
    console.print(f"  World Labs (Marble): {'[green]configured[/green]' if cfg.api_keys.marble else '[red]missing[/red]'}")

    if issues:
        console.print("\n[red]Missing required keys:[/red]")
        for issue in issues:
            console.print(f"  - {issue}")
        sys.exit(1)
    else:
        console.print("\n[green]All required keys configured![/green]")


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

    # Warn if not calibrated
    if not profile.is_calibrated:
        console.print(Panel.fit(
            f"[yellow]Profile not calibrated[/yellow]\n\n"
            f"{profile.training_progress}\n\n"
            f"For best results, run [cyan]splatworld-agent train \"{prompt_text}\"[/cyan]\n"
            f"to calibrate your taste profile first (20 rated images).\n\n"
            f"[dim]Continuing with uncalibrated profile...[/dim]",
            title="Training Recommended",
        ))

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
@click.argument("image_nums", nargs=-1, type=int, required=True)
@click.argument("rating", type=click.Choice(["++", "+", "-", "--"]))
def rate(image_nums: tuple, rating: str):
    """Rate images by number from the current batch.

    Examples:
        rate 1 ++      Rate image 1 as love
        rate 3 -       Rate image 3 as not great
        rate 2 5 +     Rate images 2 and 5 as good

    Rating scale:
        ++  love it (will be converted to splat)
        +   like it
        -   not great
        --  hate it
    """
    project_dir = get_project_dir()
    if not project_dir:
        console.print("[red]Error: Not in a SplatWorld project.[/red]")
        sys.exit(1)

    manager = ProfileManager(project_dir.parent)

    # Get current batch info for validation
    if not manager.current_session_path.exists():
        console.print("[red]No batch context found.[/red]")
        console.print("[yellow]Generate a batch first: splatworld-agent batch \"your prompt\"[/yellow]")
        sys.exit(1)

    # Read batch size for validation
    try:
        with open(manager.current_session_path) as f:
            session = json.load(f)
        batch_size = session.get("batch_size", 0)
    except (json.JSONDecodeError, IOError):
        console.print("[red]Error reading batch context.[/red]")
        sys.exit(1)

    if batch_size == 0:
        console.print("[red]No images in current batch.[/red]")
        sys.exit(1)

    # Validate all image numbers first
    invalid_nums = [n for n in image_nums if n < 1 or n > batch_size]
    if invalid_nums:
        console.print(f"[red]Invalid image number(s): {invalid_nums}[/red]")
        console.print(f"[yellow]Valid range: 1-{batch_size}[/yellow]")
        sys.exit(1)

    # Rate each image
    rating_display = {
        "++": "[green]Love it![/green]",
        "+": "[green]Good[/green]",
        "-": "[yellow]Not great[/yellow]",
        "--": "[red]Hate it[/red]",
    }

    for image_num in image_nums:
        gen_id = manager.resolve_image_number(image_num)

        if not gen_id:
            console.print(f"[red]Could not resolve image {image_num}.[/red]")
            continue

        # Create and save feedback (replaces existing if re-rating)
        fb = Feedback(
            generation_id=gen_id,
            timestamp=datetime.now(),
            rating=rating,
        )
        was_replaced, old_rating = manager.add_or_replace_feedback(fb)

        if was_replaced:
            console.print(f"Image {image_num}: {rating_display[old_rating]} -> {rating_display[rating]} [dim](updated)[/dim]")
        else:
            console.print(f"Image {image_num}: {rating_display[rating]}")

    # Show quick stats
    if len(image_nums) > 1:
        console.print(f"\n[dim]Rated {len(image_nums)} images as {rating}[/dim]")

    # Suggest next steps
    profile = manager.load_profile()
    unprocessed = len(manager.get_unprocessed_feedback())
    config = Config.load()
    if unprocessed >= config.defaults.auto_learn_threshold:
        console.print(f"\n[cyan]You have {unprocessed} feedback entries. Consider running 'splatworld-agent learn'.[/cyan]")


@main.command("brate")
@click.argument("ratings_input", nargs=-1, required=True)
def batch_rate(ratings_input: tuple):
    """Rate multiple images with different ratings in one command.

    Examples:
        brate 1 ++ 2 - 3 -- 4 +
        brate 1++ 2- 3-- 4+     (spaces optional)

    Rating scale:
        ++  love it (will be converted to splat)
        +   like it
        -   not great
        --  hate it
    """
    project_dir = get_project_dir()
    if not project_dir:
        console.print("[red]Error: Not in a SplatWorld project.[/red]")
        sys.exit(1)

    manager = ProfileManager(project_dir.parent)

    # Check for batch context
    if not manager.current_session_path.exists():
        console.print("[red]No batch context found.[/red]")
        console.print("[yellow]Generate a batch first: splatworld-agent batch \"your prompt\"[/yellow]")
        sys.exit(1)

    # Get batch info
    try:
        with open(manager.current_session_path) as f:
            session = json.load(f)
        batch_size = session.get("batch_size", 0)
        batch_gen_ids = session.get("batch_generation_ids", [])
    except (json.JSONDecodeError, IOError):
        console.print("[red]Error reading batch context.[/red]")
        sys.exit(1)

    if batch_size == 0:
        console.print("[red]No images in current batch.[/red]")
        sys.exit(1)

    # Parse the input
    input_str = " ".join(ratings_input)
    pairs = parse_batch_ratings(input_str)

    if not pairs:
        console.print("[red]Could not parse ratings.[/red]")
        console.print("[yellow]Format: 1 ++ 2 - 3 -- 4 +[/yellow]")
        sys.exit(1)

    # Validate all image numbers
    invalid_nums = [n for n, _ in pairs if n < 1 or n > batch_size]
    if invalid_nums:
        console.print(f"[red]Invalid image number(s): {invalid_nums}[/red]")
        console.print(f"[yellow]Valid range: 1-{batch_size}[/yellow]")
        sys.exit(1)

    # Process each rating
    rating_display = {
        "++": "[green]Love it![/green]",
        "+": "[green]Good[/green]",
        "-": "[yellow]Not great[/yellow]",
        "--": "[red]Hate it[/red]",
    }

    for image_num, rating in pairs:
        gen_id = manager.resolve_image_number(image_num)

        if not gen_id:
            console.print(f"[red]Could not resolve image {image_num}.[/red]")
            continue

        fb = Feedback(
            generation_id=gen_id,
            timestamp=datetime.now(),
            rating=rating,
        )

        was_replaced, old_rating = manager.add_or_replace_feedback(fb)

        if was_replaced:
            console.print(f"Image {image_num}: {rating_display[old_rating]} -> {rating_display[rating]} [dim](updated)[/dim]")
        else:
            console.print(f"Image {image_num}: {rating_display[rating]}")

    # Check for missing ratings (RATE-02)
    unrated = manager.get_unrated_in_batch(batch_gen_ids)
    if unrated:
        console.print(f"\n[yellow]Unrated images: {', '.join(map(str, unrated))}[/yellow]")
        if click.confirm("Rate remaining images?", default=True):
            for img_num in unrated:
                rating = click.prompt(
                    f"Rating for image {img_num}",
                    type=click.Choice(["++", "+", "-", "--", "s"]),
                    default="s"
                )
                if rating != "s":
                    gen_id = manager.resolve_image_number(img_num)
                    fb = Feedback(
                        generation_id=gen_id,
                        timestamp=datetime.now(),
                        rating=rating,
                    )
                    manager.add_or_replace_feedback(fb)
                    console.print(f"Image {img_num}: {rating_display[rating]}")

    # Summary
    console.print(f"\n[green]Rated {len(pairs)} images.[/green]")

    # Suggest next steps
    profile = manager.load_profile()
    unprocessed = len(manager.get_unprocessed_feedback())
    config = Config.load()
    if unprocessed >= config.defaults.auto_learn_threshold:
        console.print(f"\n[cyan]You have {unprocessed} feedback entries. Consider running 'splatworld-agent learn'.[/cyan]")


@main.command()
@click.argument("prompt", nargs=-1, required=True)
@click.option("--count", "-n", default=5, help="Number of images to generate per cycle")
@click.option("--cycles", "-c", default=1, help="Number of cycles (generate, review, learn)")
@click.option("--generator", type=click.Choice(["nano", "gemini"]), default=None, help="Image generator to use")
def batch(prompt: tuple, count: int, cycles: int, generator: str):
    """Generate a batch of images for review.

    Generates N images, then lets you review and rate them before
    optionally running more cycles with learned preferences.

    Example workflow:
        splatworld-agent batch "cozy cabin interior" -n 5 -c 2
        # Generates 5 images, you review them, learns preferences,
        # then generates 5 more with updated taste profile
    """
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

    gen_name = generator or config.defaults.image_generator

    # Check calibration status
    profile = manager.load_profile()
    if not profile.is_calibrated:
        console.print(Panel.fit(
            f"[yellow]Profile not calibrated[/yellow]\n\n"
            f"{profile.training_progress}\n\n"
            f"Consider running [cyan]splatworld-agent train \"{prompt_text}\"[/cyan]\n"
            f"for guided training (generates, reviews, learns until calibrated).\n\n"
            f"[dim]Continuing with batch generation...[/dim]",
            title="Training Recommended",
        ))

    for cycle_num in range(1, cycles + 1):
        console.print(Panel.fit(
            f"[bold]Cycle {cycle_num}/{cycles}[/bold]\n"
            f"Generating {count} images for: {prompt_text}",
            title="Batch Generation",
        ))

        profile = manager.load_profile()
        enhanced_prompt = enhance_prompt(prompt_text, profile)

        if enhanced_prompt != prompt_text:
            console.print(f"[dim]Enhanced with taste:[/dim] {enhanced_prompt}")

        # Track this batch
        batch_id = f"batch-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        batch_gen_ids = []

        try:
            if gen_name == "nano":
                from splatworld_agent.generators.nano import NanoGenerator
                img_gen = NanoGenerator(api_key=config.api_keys.nano or config.api_keys.google)
            else:
                from splatworld_agent.generators.gemini import GeminiGenerator
                img_gen = GeminiGenerator(api_key=config.api_keys.google)

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                for i in range(count):
                    task = progress.add_task(f"Generating image {i+1}/{count}...", total=None)

                    gen_id = f"gen-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
                    batch_gen_ids.append(gen_id)

                    image_bytes = img_gen.generate(enhanced_prompt, seed=None)

                    # Save generation
                    gen_dir = manager.save_generation(Generation(
                        id=gen_id,
                        prompt=prompt_text,
                        enhanced_prompt=enhanced_prompt,
                        timestamp=datetime.now(),
                        metadata={"generator": gen_name, "batch_id": batch_id, "batch_index": i},
                    ))

                    # Save image
                    image_path = gen_dir / "source.png"
                    with open(image_path, "wb") as f:
                        f.write(image_bytes)

                    # Update metadata with path
                    metadata_path = gen_dir / "metadata.json"
                    with open(metadata_path) as f:
                        gen_data = json.load(f)
                    gen_data["source_image_path"] = str(image_path)
                    with open(metadata_path, "w") as f:
                        json.dump(gen_data, f, indent=2)

                    progress.update(task, description=f"[green]Image {i+1}/{count} saved: {gen_id}")

            img_gen.close()

            # Store batch context for numbered references
            manager.set_current_batch(batch_id, batch_gen_ids)

            console.print(f"\n[bold green]Batch complete![/bold green] Generated {count} images.")
            console.print(f"[dim]Batch ID: {batch_id}[/dim]")

            # Show numbered images summary
            console.print("\n[bold]Your images:[/bold]")
            for i, gen_id in enumerate(batch_gen_ids, start=1):
                gen = manager.get_generation(gen_id)
                if gen and gen.source_image_path:
                    console.print(f"  [cyan]{i}[/cyan] - {gen.source_image_path}")

            # Show rating instructions with new syntax
            console.print("\n[bold]Rate your images:[/bold]")
            console.print("  [cyan]splatworld-agent rate 1 ++[/cyan]  - Love image 1")
            console.print("  [cyan]splatworld-agent rate 3 -[/cyan]   - Dislike image 3")
            console.print("  [cyan]splatworld-agent rate 2 5 +[/cyan] - Like images 2 and 5")

            # If more cycles, prompt for review
            if cycle_num < cycles:
                console.print(f"\n[yellow]Review images before cycle {cycle_num + 1}...[/yellow]")
                console.print("Press Enter after reviewing, or Ctrl+C to stop.")
                try:
                    input()
                    # Run learn to update profile
                    feedback_count = len(manager.get_feedback_history())
                    if feedback_count >= 3:
                        console.print("Learning from feedback...")
                        engine = LearningEngine(api_key=config.api_keys.anthropic)
                        generations = manager.get_recent_generations(limit=count * cycle_num)
                        feedbacks = manager.get_feedback_history()
                        result = engine.synthesize_from_history(generations, feedbacks, profile)
                        if result.get("updates"):
                            profile = engine.apply_updates(profile, result["updates"])
                            manager.save_profile(profile)
                            console.print("[green]Profile updated with learned preferences.[/green]")
                except KeyboardInterrupt:
                    console.print("\n[yellow]Stopping batch generation.[/yellow]")
                    break

        except Exception as e:
            console.print(f"\n[red]Batch generation failed:[/red] {e}")
            sys.exit(1)

    console.print("\n[bold]Batch workflow complete![/bold]")
    console.print("Next steps:")
    console.print("  1. [cyan]splatworld-agent rate 1 ++[/cyan] - Rate images by number")
    console.print("  2. [cyan]splatworld-agent learn[/cyan] - Learn from your feedback")
    console.print("  3. [cyan]splatworld-agent convert[/cyan] - Convert favorites to 3D splats")


@main.command()
@click.option("--batch", "-b", help="Review specific batch ID")
@click.option("--all", "all_unrated", is_flag=True, help="Review ALL unrated images across all batches")
@click.option("--current", "-c", is_flag=True, help="Review current batch (default if no batch specified)")
@click.option("--limit", "-n", default=10, help="Number of images to review")
@click.option("--unrated", is_flag=True, help="Only show unrated images")
def review(batch: str, current: bool, limit: int, unrated: bool, all_unrated: bool):
    """Interactively review and rate generated images.

    By default, reviews the current batch if one exists.
    Use --all to review ALL unrated images across all batches.

    Opens each image and prompts for quick feedback:
      ++ = love it
      +  = like it
      -  = not great
      -- = hate it
      s  = skip
      q  = quit review
    """
    project_dir = get_project_dir()
    if not project_dir:
        console.print("[red]Error: Not in a SplatWorld project.[/red]")
        sys.exit(1)

    manager = ProfileManager(project_dir.parent)

    # --all flag: Review ALL unrated images across all batches
    if all_unrated:
        unrated_with_context = manager.get_all_unrated_generations()
        unrated_with_context = unrated_with_context[:limit]

        if not unrated_with_context:
            console.print("[green]All images have been rated![/green]")
            return

        console.print(Panel.fit(
            f"[bold]All Unrated Images[/bold]\n"
            f"Found {len(unrated_with_context)} unrated images\n\n"
            "Rating options:\n"
            "  [green]++[/green] = love it (will be converted to splat)\n"
            "  [green]+[/green]  = like it\n"
            "  [yellow]-[/yellow]  = not great\n"
            "  [red]--[/red] = hate it\n"
            "  [dim]s[/dim]  = skip\n"
            "  [dim]q[/dim]  = quit review",
            title="Review All Unrated",
        ))

        reviewed = 0
        loved = 0
        total = len(unrated_with_context)

        for i, (gen, batch_ctx) in enumerate(unrated_with_context, start=1):
            console.print(f"\n[bold cyan]Image {i}[/bold cyan] of {total}")
            console.print(format_batch_context(batch_ctx))
            console.print(f"Prompt: {gen.prompt}")

            if gen.source_image_path:
                # Try inline display first
                displayed = display.display_image(Path(gen.source_image_path), max_width=80)

                if displayed:
                    console.print("[dim]Preview shown above[/dim]")
                else:
                    # Fallback to file path + external viewer
                    console.print(f"[cyan]File: {gen.source_image_path}[/cyan]")
                    try:
                        import subprocess
                        subprocess.Popen(["open", gen.source_image_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    except Exception:
                        pass

            while True:
                try:
                    rating_input = input("\nRating (++/+/-/--/s/q): ").strip().lower()
                except (KeyboardInterrupt, EOFError):
                    rating_input = "q"

                if rating_input == "q":
                    console.print("[yellow]Review stopped.[/yellow]")
                    break
                elif rating_input == "s":
                    console.print("[dim]Skipped[/dim]")
                    break
                elif rating_input in ("++", "+", "-", "--"):
                    fb = Feedback(
                        generation_id=gen.id,
                        timestamp=datetime.now(),
                        rating=rating_input,
                    )
                    manager.add_or_replace_feedback(fb)

                    rating_display = {"++": "[green]Love it![/green]", "+": "[green]Good[/green]",
                                      "-": "[yellow]Not great[/yellow]", "--": "[red]Hate it[/red]"}
                    console.print(f"Recorded: {rating_display[rating_input]}")
                    reviewed += 1
                    if rating_input == "++":
                        loved += 1
                    break
                else:
                    console.print("[red]Invalid rating. Use ++, +, -, --, s, or q[/red]")

            if rating_input == "q":
                break

        console.print(f"\n[bold]Review complete![/bold]")
        console.print(f"Reviewed: {reviewed} images")
        console.print(f"Loved: {loved} images")

        # Show remaining unrated count
        remaining = manager.get_all_unrated_generations()
        if remaining:
            console.print(f"\n[yellow]Remaining unrated: {len(remaining)} images[/yellow]")

        if loved > 0:
            console.print(f"\n[cyan]Run 'splatworld-agent convert' to turn your {loved} loved images into 3D splats.[/cyan]")

        if reviewed >= 3:
            console.print(f"[cyan]Run 'splatworld-agent learn' to update your taste profile.[/cyan]")

        return

    # Determine which generations to review
    generations = []

    if batch:
        # Review specific batch by ID
        all_gens = manager.get_recent_generations(limit=limit * 2)
        generations = [g for g in all_gens if g.metadata.get("batch_id") == batch]
    elif current or (not batch and not unrated):
        # Review current batch (default behavior)
        generations = manager.get_current_batch_generations()
        if not generations:
            console.print("[yellow]No current batch. Generate one with: splatworld-agent batch \"prompt\"[/yellow]")
            console.print("[dim]Or use --unrated to review all unrated images.[/dim]")
            return
    else:
        # Review recent generations
        generations = manager.get_recent_generations(limit=limit * 2)

    if unrated:
        # Filter to unrated only
        feedbacks = {f.generation_id: f for f in manager.get_feedback_history()}
        generations = [g for g in generations if g.id not in feedbacks]

    generations = generations[:limit]

    if not generations:
        console.print("[yellow]No images to review.[/yellow]")
        return

    console.print(Panel.fit(
        f"[bold]Reviewing {len(generations)} images[/bold]\n\n"
        "Rating options:\n"
        "  [green]++[/green] = love it (will be converted to splat)\n"
        "  [green]+[/green]  = like it\n"
        "  [yellow]-[/yellow]  = not great\n"
        "  [red]--[/red] = hate it\n"
        "  [dim]s[/dim]  = skip\n"
        "  [dim]q[/dim]  = quit review",
        title="Image Review",
    ))

    reviewed = 0
    loved = 0

    for i, gen in enumerate(generations, start=1):
        # Show image number prominently
        console.print(f"\n[bold cyan]Image {i}[/bold cyan] of {len(generations)}")

        # Show batch index if available
        batch_index = gen.metadata.get("batch_index")
        if batch_index is not None:
            console.print(f"[dim]Batch position: {batch_index + 1}[/dim]")

        console.print(f"Prompt: {gen.prompt}")

        # Show image path
        if gen.source_image_path:
            # Try inline display first
            displayed = display.display_image(Path(gen.source_image_path), max_width=80)

            if displayed:
                console.print("[dim]Preview shown above[/dim]")
            else:
                # Fallback to file path + external viewer
                console.print(f"[cyan]File: {gen.source_image_path}[/cyan]")
                # Try to open image in default viewer
                try:
                    import subprocess
                    subprocess.Popen(["open", gen.source_image_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except Exception:
                    pass

        # Get rating
        while True:
            try:
                rating_input = input("\nRating (++/+/-/--/s/q): ").strip().lower()
            except (KeyboardInterrupt, EOFError):
                rating_input = "q"

            if rating_input == "q":
                console.print("[yellow]Review stopped.[/yellow]")
                break
            elif rating_input == "s":
                console.print("[dim]Skipped[/dim]")
                break
            elif rating_input in ("++", "+", "-", "--"):
                fb = Feedback(
                    generation_id=gen.id,
                    timestamp=datetime.now(),
                    rating=rating_input,
                )
                manager.add_feedback(fb)

                rating_display = {"++": "[green]Love it![/green]", "+": "[green]Good[/green]",
                                  "-": "[yellow]Not great[/yellow]", "--": "[red]Hate it[/red]"}
                console.print(f"Recorded: {rating_display[rating_input]}")
                reviewed += 1
                if rating_input == "++":
                    loved += 1
                break
            else:
                console.print("[red]Invalid rating. Use ++, +, -, --, s, or q[/red]")

        if rating_input == "q":
            break

    console.print(f"\n[bold]Review complete![/bold]")
    console.print(f"Reviewed: {reviewed} images")
    console.print(f"Loved: {loved} images")

    if loved > 0:
        console.print(f"\n[cyan]Run 'splatworld-agent convert' to turn your {loved} loved images into 3D splats.[/cyan]")

    if reviewed >= 3:
        console.print(f"[cyan]Run 'splatworld-agent learn' to update your taste profile.[/cyan]")


@main.command()
@click.option("--all-positive", is_flag=True, help="Convert all positively rated (+ and ++)")
@click.option("--generation", "-g", multiple=True, help="Specific generation IDs to convert")
@click.option("--dry-run", is_flag=True, help="Show what would be converted without doing it")
def convert(all_positive: bool, generation: tuple, dry_run: bool):
    """Convert loved images to 3D splats.

    By default, converts all images rated '++' (love it) that don't
    already have splats.
    """
    project_dir = get_project_dir()
    if not project_dir:
        console.print("[red]Error: Not in a SplatWorld project.[/red]")
        sys.exit(1)

    manager = ProfileManager(project_dir.parent)
    config = Config.load()

    if not config.api_keys.marble:
        console.print("[red]Error: Marble API key required for 3D conversion.[/red]")
        console.print("Set WORLDLABS_API_KEY environment variable.")
        sys.exit(1)

    # Find generations to convert
    to_convert = []

    if generation:
        # Specific generations requested
        for gen_id in generation:
            gen = manager.get_generation(gen_id)
            if gen:
                to_convert.append(gen)
            else:
                console.print(f"[yellow]Generation not found: {gen_id}[/yellow]")
    else:
        # Find loved generations without splats
        generations = manager.get_recent_generations(limit=50)
        feedbacks = {f.generation_id: f for f in manager.get_feedback_history()}

        for gen in generations:
            fb = feedbacks.get(gen.id)
            if not fb:
                continue

            # Skip if already has splat
            if gen.splat_path:
                continue

            # Check rating
            if all_positive and fb.rating in ("++", "+"):
                to_convert.append(gen)
            elif fb.rating == "++":
                to_convert.append(gen)

    if not to_convert:
        console.print("[yellow]No images to convert.[/yellow]")
        console.print("[dim]Rate images with '++' to mark them for conversion.[/dim]")
        return

    # Estimate cost
    cost = len(to_convert) * 1.50
    console.print(Panel.fit(
        f"[bold]Converting {len(to_convert)} images to 3D splats[/bold]\n\n"
        f"Estimated cost: [yellow]${cost:.2f}[/yellow]\n\n"
        "Images to convert:\n" +
        "\n".join(f"  - {g.id}: {g.prompt[:40]}..." for g in to_convert[:5]) +
        (f"\n  ... and {len(to_convert) - 5} more" if len(to_convert) > 5 else ""),
        title="3D Conversion",
    ))

    if dry_run:
        console.print("\n[yellow]Dry run - no conversions performed.[/yellow]")
        return

    # Confirm
    try:
        confirm = input(f"\nProceed with conversion? (y/N): ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        confirm = "n"

    if confirm != "y":
        console.print("[yellow]Conversion cancelled.[/yellow]")
        return

    # Convert each image
    from splatworld_agent.core.marble import MarbleClient

    marble = MarbleClient(api_key=config.api_keys.marble)
    total_cost = 0.0
    converted = 0

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            for i, gen in enumerate(to_convert):
                task = progress.add_task(f"Converting {i+1}/{len(to_convert)}: {gen.id}...", total=None)

                # Load image
                if not gen.source_image_path or not Path(gen.source_image_path).exists():
                    progress.update(task, description=f"[red]Missing image: {gen.id}[/red]")
                    continue

                with open(gen.source_image_path, "rb") as f:
                    image_bytes = f.read()

                image_b64 = base64.b64encode(image_bytes).decode()

                def on_progress(status: str, description: str):
                    progress.update(task, description=f"{gen.id}: {description or status}")

                result = marble.generate_and_wait(
                    image_base64=image_b64,
                    mime_type="image/png",
                    display_name=gen.id,
                    is_panorama=True,
                    on_progress=on_progress,
                )

                # Get generation directory
                gen_dir = Path(gen.source_image_path).parent

                # Download splat file
                splat_path = None
                mesh_path = None

                if result.splat_url:
                    splat_path = gen_dir / "scene.spz"
                    marble.download_file(result.splat_url, splat_path)

                if result.mesh_url and config.defaults.download_meshes:
                    mesh_path = gen_dir / "collision.glb"
                    marble.download_file(result.mesh_url, mesh_path)

                # Update metadata
                metadata_path = gen_dir / "metadata.json"
                with open(metadata_path) as f:
                    gen_data = json.load(f)

                if splat_path:
                    gen_data["splat_path"] = str(splat_path)
                if mesh_path:
                    gen_data["mesh_path"] = str(mesh_path)
                gen_data["viewer_url"] = result.viewer_url

                with open(metadata_path, "w") as f:
                    json.dump(gen_data, f, indent=2)

                total_cost += result.cost_usd
                converted += 1
                progress.update(task, description=f"[green]Converted: {gen.id}[/green]")

    finally:
        marble.close()

    console.print(f"\n[bold green]Conversion complete![/bold green]")
    console.print(f"Converted: {converted}/{len(to_convert)} images")
    console.print(f"Total cost: [yellow]${total_cost:.2f}[/yellow]")


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
    calibration_status = "[green]CALIBRATED[/green]" if prof.is_calibrated else f"[yellow]{prof.training_progress}[/yellow]"

    console.print(Panel.fit(
        f"[bold]Taste Profile[/bold]\n"
        f"Status: {calibration_status}\n"
        f"Created: {prof.created.strftime('%Y-%m-%d')}\n"
        f"Updated: {prof.updated.strftime('%Y-%m-%d %H:%M')}\n"
        f"Generations: {prof.stats.total_generations}\n"
        f"Feedback: {prof.stats.feedback_count} "
        f"([green]{prof.stats.love_count}++[/green] [green]{prof.stats.like_count}+[/green] "
        f"[yellow]{prof.stats.dislike_count}-[/yellow] [red]{prof.stats.hate_count}--[/red])",
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

            # Update calibration status
            updated_profile.calibration.last_learn_at = datetime.now()
            updated_profile.calibration.learn_count += 1

            # Check if we've reached calibration threshold
            can_calibrate, reason = updated_profile.stats.can_calibrate()
            if can_calibrate and not updated_profile.calibration.is_calibrated:
                updated_profile.calibration.is_calibrated = True
                updated_profile.calibration.calibrated_at = datetime.now()
                console.print("\n[bold green]Profile CALIBRATED![/bold green]")
                console.print("[green]Your taste profile is now trained and ready for autonomous generation.[/green]")
            elif not updated_profile.calibration.is_calibrated:
                console.print(f"\n[yellow]Training progress:[/yellow] {reason}")

            manager.save_profile(updated_profile)
            console.print("\n[bold green]Profile updated![/bold green]")
            console.print("[dim]Use 'splatworld-agent profile' to view your updated taste profile.[/dim]")

    except Exception as e:
        console.print(f"\n[red]Learning failed:[/red] {e}")
        sys.exit(1)


@main.command()
@click.argument("prompt", nargs=-1, required=True)
@click.option("--images-per-round", "-n", default=5, help="Images to generate per round")
@click.option("--generator", type=click.Choice(["nano", "gemini"]), default=None, help="Image generator")
def train(prompt: tuple, images_per_round: int, generator: str):
    """Guided training mode to calibrate your taste profile.

    Runs generate-review-learn cycles until the profile is calibrated
    (minimum 20 rated images with good positive/negative distribution).

    Example:
        splatworld-agent train "cozy cabin interior"
        # Generates 5 images, you review them, learns, repeats until calibrated
    """
    from splatworld_agent.models import CalibrationStatus

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

    if profile.is_calibrated:
        console.print("[green]Profile is already calibrated![/green]")
        console.print(f"[dim]Calibrated at: {profile.calibration.calibrated_at}[/dim]")
        console.print(f"[dim]Total feedback: {profile.stats.feedback_count}[/dim]")
        return

    min_feedback = CalibrationStatus.MIN_FEEDBACK_FOR_CALIBRATION
    console.print(Panel.fit(
        f"[bold]Training Mode[/bold]\n\n"
        f"Training your taste profile with: {prompt_text}\n\n"
        f"Requirements for calibration:\n"
        f"  - Minimum {min_feedback} rated images\n"
        f"  - At least 10% positive feedback (++/+)\n"
        f"  - At least 10% negative feedback (--/-)\n\n"
        f"Current progress: {profile.stats.feedback_count}/{min_feedback} ratings\n"
        f"  Positive: {profile.stats.positive_count} ({profile.stats.love_count} loves, {profile.stats.like_count} likes)\n"
        f"  Negative: {profile.stats.negative_count} ({profile.stats.hate_count} hates, {profile.stats.dislike_count} dislikes)\n\n"
        f"[dim]Press Ctrl+C at any time to pause training.[/dim]",
        title="Taste Profile Training",
    ))

    gen_name = generator or config.defaults.image_generator
    round_num = 0

    try:
        while not profile.is_calibrated:
            round_num += 1
            remaining = min_feedback - profile.stats.feedback_count
            images_this_round = min(images_per_round, remaining + 2)  # Generate a few extra

            console.print(f"\n[bold cyan]━━━ Round {round_num} ━━━[/bold cyan]")
            console.print(f"Generating {images_this_round} images...")

            # Get current profile for enhancement
            profile = manager.load_profile()
            enhanced_prompt = enhance_prompt(prompt_text, profile)

            if enhanced_prompt != prompt_text and round_num > 1:
                console.print(f"[dim]Prompt enhanced with learned preferences[/dim]")

            # Generate images
            if gen_name == "nano":
                from splatworld_agent.generators.nano import NanoGenerator
                img_gen = NanoGenerator(api_key=config.api_keys.nano or config.api_keys.google)
            else:
                from splatworld_agent.generators.gemini import GeminiGenerator
                img_gen = GeminiGenerator(api_key=config.api_keys.google)

            batch_gen_ids = []

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                for i in range(images_this_round):
                    task = progress.add_task(f"Generating {i+1}/{images_this_round}...", total=None)

                    gen_id = f"gen-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
                    batch_gen_ids.append(gen_id)

                    image_bytes = img_gen.generate(enhanced_prompt, seed=None)

                    gen_dir = manager.save_generation(Generation(
                        id=gen_id,
                        prompt=prompt_text,
                        enhanced_prompt=enhanced_prompt,
                        timestamp=datetime.now(),
                        metadata={"generator": gen_name, "training_round": round_num},
                    ))

                    image_path = gen_dir / "source.png"
                    with open(image_path, "wb") as f:
                        f.write(image_bytes)

                    metadata_path = gen_dir / "metadata.json"
                    with open(metadata_path) as f:
                        gen_data = json.load(f)
                    gen_data["source_image_path"] = str(image_path)
                    with open(metadata_path, "w") as f:
                        json.dump(gen_data, f, indent=2)

                    progress.update(task, description=f"[green]{i+1}/{images_this_round} done[/green]")

            img_gen.close()

            # Review phase
            console.print(f"\n[bold]Review Phase[/bold]")
            console.print("Rate each image: [green]++[/green]=love [green]+[/green]=like [yellow]-[/yellow]=meh [red]--[/red]=hate [dim]s[/dim]=skip")

            reviewed = 0
            for gen_id in batch_gen_ids:
                gen = manager.get_generation(gen_id)
                if not gen or not gen.source_image_path:
                    continue

                console.print(f"\n[dim]Image {reviewed + 1}/{len(batch_gen_ids)}[/dim]")

                # Open image
                try:
                    import subprocess
                    subprocess.Popen(["open", gen.source_image_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except Exception:
                    console.print(f"[cyan]File: {gen.source_image_path}[/cyan]")

                while True:
                    try:
                        rating = input("Rating (++/+/-/--/s): ").strip().lower()
                    except (KeyboardInterrupt, EOFError):
                        raise KeyboardInterrupt

                    if rating == "s":
                        console.print("[dim]Skipped[/dim]")
                        break
                    elif rating in ("++", "+", "-", "--"):
                        fb = Feedback(
                            generation_id=gen_id,
                            timestamp=datetime.now(),
                            rating=rating,
                        )
                        manager.add_feedback(fb)

                        rating_display = {"++": "[green]Love![/green]", "+": "[green]Good[/green]",
                                          "-": "[yellow]Meh[/yellow]", "--": "[red]Hate[/red]"}
                        console.print(rating_display[rating])
                        reviewed += 1
                        break
                    else:
                        console.print("[red]Invalid. Use ++, +, -, --, or s[/red]")

            # Reload profile to get updated stats
            profile = manager.load_profile()

            # Learn phase (if enough feedback)
            if profile.stats.feedback_count >= 3:
                console.print(f"\n[bold]Learning Phase[/bold]")
                console.print(f"Analyzing {profile.stats.feedback_count} feedback entries...")

                try:
                    engine = LearningEngine(api_key=config.api_keys.anthropic)
                    generations = manager.get_recent_generations(limit=50)
                    feedbacks = manager.get_feedback_history()

                    result = engine.synthesize_from_history(generations, feedbacks, profile)

                    if result.get("updates"):
                        profile = engine.apply_updates(profile, result["updates"])
                        profile.calibration.last_learn_at = datetime.now()
                        profile.calibration.learn_count += 1

                        # Check calibration
                        can_calibrate, reason = profile.stats.can_calibrate()
                        if can_calibrate:
                            profile.calibration.is_calibrated = True
                            profile.calibration.calibrated_at = datetime.now()

                        manager.save_profile(profile)
                        console.print("[green]Preferences updated![/green]")
                    else:
                        console.print("[dim]No new patterns identified yet.[/dim]")

                except Exception as e:
                    console.print(f"[yellow]Learning skipped: {e}[/yellow]")

            # Show progress
            console.print(f"\n[bold]Progress:[/bold] {profile.stats.feedback_count}/{min_feedback} ratings")
            console.print(f"  [green]Positive: {profile.stats.positive_count}[/green] | [red]Negative: {profile.stats.negative_count}[/red]")

            can_calibrate, reason = profile.stats.can_calibrate()
            if not can_calibrate:
                console.print(f"  [yellow]{reason}[/yellow]")

        # Training complete!
        console.print(Panel.fit(
            f"[bold green]Training Complete![/bold green]\n\n"
            f"Your taste profile is now calibrated.\n\n"
            f"Total ratings: {profile.stats.feedback_count}\n"
            f"Training rounds: {round_num}\n"
            f"Learn cycles: {profile.calibration.learn_count}\n\n"
            f"[bold]Next steps:[/bold]\n"
            f"  [cyan]splatworld-agent batch \"{prompt_text}\"[/cyan] - Generate with learned preferences\n"
            f"  [cyan]splatworld-agent convert[/cyan] - Convert favorites to 3D splats",
            title="Calibration Complete",
        ))

    except KeyboardInterrupt:
        console.print("\n\n[yellow]Training paused.[/yellow]")
        console.print(f"Progress saved: {profile.stats.feedback_count}/{min_feedback} ratings")
        console.print("Run [cyan]splatworld-agent train[/cyan] again to continue.")


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
def update():
    """Update SplatWorld Agent to the latest version.

    Pulls the latest changes from the git repository.
    """
    import subprocess

    # Find the package directory
    package_dir = Path(__file__).parent.parent

    # Check if it's a git repo
    git_dir = package_dir / ".git"
    if not git_dir.exists():
        console.print("[red]Error: SplatWorld Agent is not installed from git.[/red]")
        console.print(f"[dim]Package location: {package_dir}[/dim]")
        sys.exit(1)

    console.print(f"[dim]Updating from: {package_dir}[/dim]")

    try:
        # Fetch first to see what's available
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Fetching updates...", total=None)

            # Fetch
            result = subprocess.run(
                ["git", "fetch"],
                cwd=package_dir,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                console.print(f"[red]Fetch failed:[/red] {result.stderr}")
                sys.exit(1)

            # Check for updates
            result = subprocess.run(
                ["git", "log", "HEAD..origin/main", "--oneline"],
                cwd=package_dir,
                capture_output=True,
                text=True,
            )
            new_commits = result.stdout.strip().split("\n") if result.stdout.strip() else []

            if not new_commits or new_commits == [""]:
                progress.update(task, description="[green]Already up to date!")
                console.print("\n[green]SplatWorld Agent is already up to date.[/green]")
                return

            progress.update(task, description=f"Found {len(new_commits)} new commits...")

            # Show what's coming
            console.print(f"\n[bold]New updates available:[/bold]")
            for commit in new_commits[:10]:
                console.print(f"  [cyan]{commit}[/cyan]")
            if len(new_commits) > 10:
                console.print(f"  [dim]... and {len(new_commits) - 10} more[/dim]")

            # Pull
            progress.update(task, description="Pulling updates...")
            result = subprocess.run(
                ["git", "pull", "--ff-only"],
                cwd=package_dir,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                console.print(f"[red]Pull failed:[/red] {result.stderr}")
                console.print("[yellow]You may have local changes. Try:[/yellow]")
                console.print(f"  cd {package_dir} && git stash && git pull && git stash pop")
                sys.exit(1)

            progress.update(task, description="[green]Update complete!")

        console.print(Panel.fit(
            f"[bold green]Updated Successfully![/bold green]\n\n"
            f"Pulled {len(new_commits)} new commit(s).\n\n"
            f"[dim]Run '/splatworld-agent:help' to see new commands.[/dim]",
            title="SplatWorld Agent",
        ))

    except FileNotFoundError:
        console.print("[red]Error: git not found. Please install git.[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Update failed:[/red] {e}")
        sys.exit(1)


@main.command()
def help():
    """Show help and available commands."""
    console.print(Panel.fit(
        "[bold]SplatWorld Agent[/bold]\n"
        "Iterative 3D splat generation with taste learning.\n\n"
        "[bold]Session Management:[/bold]\n"
        "  resume-work    Resume from previous session\n"
        "  exit           Save session and exit\n\n"
        "[bold]Training (Start Here):[/bold]\n"
        "  train          Guided training until calibrated (20 images)\n"
        "  learn          Manually run learning on feedback\n\n"
        "[bold]Batch Workflow (After Training):[/bold]\n"
        "  batch          Generate N images for review\n"
        "  review         Interactively rate images (++/+/-/--)\n"
        "  convert        Convert loved images to 3D splats\n\n"
        "[bold]Single Generation:[/bold]\n"
        "  generate       Generate one image + splat from prompt\n"
        "  feedback       Rate/critique a generation\n\n"
        "[bold]Profile Management:[/bold]\n"
        "  init           Initialize .splatworld/ in current project\n"
        "  profile        View/edit taste profile\n"
        "  exemplar       Add reference image you love\n"
        "  anti-exemplar  Add reference image you hate\n"
        "  history        Browse past generations\n\n"
        "[bold]Setup:[/bold]\n"
        "  config         View/edit configuration\n"
        "  update         Update to latest version from git\n"
        "  install-prompts  Install Claude Code slash commands\n\n"
        "[dim]Use 'splatworld-agent COMMAND --help' for command details.[/dim]",
        title="SplatWorld Agent",
    ))


@main.command("exit")
@click.option("--summary", "-s", default="", help="Summary of what was accomplished")
@click.option("--notes", "-n", default="", help="Notes for next session")
def exit_session(summary: str, notes: str):
    """Save session and exit SplatWorld Agent.

    Records session activity (generations, feedback, conversions, learns)
    and saves to session history for later resumption.
    """
    project_dir = get_project_dir()
    if not project_dir:
        console.print("[red]Error: Not in a SplatWorld project.[/red]")
        sys.exit(1)

    manager = ProfileManager(project_dir.parent)

    # Check for current session
    current = manager.get_current_session()
    if not current:
        console.print("[yellow]No active session to end.[/yellow]")
        console.print("[dim]Start a session with 'splatworld-agent resume-work' first.[/dim]")
        return

    # Calculate duration
    duration = datetime.now() - current.started
    hours, remainder = divmod(int(duration.total_seconds()), 3600)
    minutes, seconds = divmod(remainder, 60)
    duration_str = f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m {seconds}s"

    # Auto-generate summary if not provided
    if not summary:
        activity = manager.calculate_session_activity(current.started)
        parts = []
        if activity["generations"] > 0:
            parts.append(f"{activity['generations']} generations")
        if activity["feedback"] > 0:
            parts.append(f"{activity['feedback']} ratings")
        if activity["conversions"] > 0:
            parts.append(f"{activity['conversions']} conversions")
        if activity["learns"] > 0:
            parts.append(f"{activity['learns']} learn cycles")

        if parts:
            summary = ", ".join(parts)
        else:
            summary = "No activity recorded"

    # End the session
    session = manager.end_session(summary=summary, notes=notes)

    if not session:
        console.print("[red]Failed to end session.[/red]")
        sys.exit(1)

    # Display farewell
    console.print(Panel.fit(
        f"[bold green]Session Complete[/bold green]\n\n"
        f"[bold]Duration:[/bold] {duration_str}\n"
        f"[bold]Activity:[/bold] {summary}\n"
        + (f"[bold]Notes:[/bold] {notes}\n" if notes else "")
        + (f"\n[bold]Last prompt:[/bold] {session.last_prompt[:50]}..." if session.last_prompt and len(session.last_prompt) > 50 else
           f"\n[bold]Last prompt:[/bold] {session.last_prompt}" if session.last_prompt else "")
        + f"\n\n[dim]Session saved. Use 'splatworld-agent resume-work' to continue.[/dim]",
        title="Goodbye!",
    ))


@main.command("resume-work")
def resume_work():
    """Resume work from previous session.

    Shows recent session history, current status, and starts a new session
    for tracking your work.
    """
    project_dir = get_project_dir()
    if not project_dir:
        console.print("[red]Error: Not in a SplatWorld project.[/red]")
        sys.exit(1)

    manager = ProfileManager(project_dir.parent)
    profile = manager.load_profile()

    # Check for existing active session
    current = manager.get_current_session()
    if current:
        duration = datetime.now() - current.started
        hours, remainder = divmod(int(duration.total_seconds()), 3600)
        minutes, _ = divmod(remainder, 60)
        duration_str = f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"

        console.print(Panel.fit(
            f"[yellow]Active session found[/yellow]\n\n"
            f"Started: {current.started.strftime('%Y-%m-%d %H:%M')}\n"
            f"Duration: {duration_str}\n\n"
            f"[dim]Use 'splatworld-agent exit' to end this session first,[/dim]\n"
            f"[dim]or continue working in the current session.[/dim]",
            title="Session Already Active",
        ))
        return

    # Show recent sessions
    sessions = manager.get_sessions(limit=5)

    if sessions:
        console.print("\n[bold]Recent Sessions:[/bold]")
        table = Table(show_header=True, header_style="bold")
        table.add_column("Date", style="cyan")
        table.add_column("Duration")
        table.add_column("Activity")
        table.add_column("Last Prompt")

        for s in sessions:
            if s.ended:
                duration = s.ended - s.started
                hours, remainder = divmod(int(duration.total_seconds()), 3600)
                minutes, _ = divmod(remainder, 60)
                duration_str = f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"
            else:
                duration_str = "?"

            prompt_preview = (s.last_prompt[:30] + "...") if s.last_prompt and len(s.last_prompt) > 30 else (s.last_prompt or "-")

            table.add_row(
                s.started.strftime("%m-%d %H:%M"),
                duration_str,
                s.summary[:40] if s.summary else "-",
                prompt_preview,
            )

        console.print(table)

        # Show notes from last session if any
        last_session = sessions[0]
        if last_session.notes:
            console.print(f"\n[bold]Notes from last session:[/bold]")
            console.print(f"[cyan]{last_session.notes}[/cyan]")
    else:
        console.print("\n[dim]No previous sessions found.[/dim]")

    # Show current status
    console.print(f"\n[bold]Current Status:[/bold]")

    # Calibration status
    if profile.is_calibrated:
        console.print(f"  Profile: [green]CALIBRATED[/green]")
    else:
        console.print(f"  Profile: [yellow]{profile.training_progress}[/yellow]")

    # Stats
    console.print(f"  Generations: {profile.stats.total_generations}")
    console.print(f"  Feedback: {profile.stats.feedback_count} "
                  f"([green]{profile.stats.positive_count}+[/green] / "
                  f"[red]{profile.stats.negative_count}-[/red])")

    # Check for unrated generations
    recent_gens = manager.get_recent_generations(limit=20)
    feedbacks = {f.generation_id: f for f in manager.get_feedback_history()}
    unrated = [g for g in recent_gens if g.id not in feedbacks]

    if unrated:
        console.print(f"\n  [yellow]Unrated generations: {len(unrated)}[/yellow]")
        console.print(f"  [dim]Run 'splatworld-agent review --unrated' to rate them[/dim]")

    # Check for loved images without splats
    loved_without_splats = []
    for gen in recent_gens:
        fb = feedbacks.get(gen.id)
        if fb and fb.rating == "++" and not gen.splat_path:
            loved_without_splats.append(gen)

    if loved_without_splats:
        console.print(f"\n  [cyan]Loved images ready for conversion: {len(loved_without_splats)}[/cyan]")
        console.print(f"  [dim]Run 'splatworld-agent convert' to create 3D splats[/dim]")

    # Start new session
    session = manager.start_session()

    console.print(Panel.fit(
        f"[bold green]Welcome Back![/bold green]\n\n"
        f"Session started: {session.started.strftime('%Y-%m-%d %H:%M')}\n"
        f"Session ID: {session.session_id}\n\n"
        f"[bold]Quick Commands:[/bold]\n"
        f"  [cyan]generate[/cyan] \"prompt\"  - Generate an image\n"
        f"  [cyan]batch[/cyan] \"prompt\"     - Generate multiple images\n"
        f"  [cyan]review[/cyan]             - Rate recent images\n"
        f"  [cyan]exit[/cyan]               - Save session and exit",
        title="SplatWorld Agent",
    ))


if __name__ == "__main__":
    main()
