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
import shutil
import sys
from typing import Optional
import uuid

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from splatworld_agent import __version__
from splatworld_agent.config import Config, get_project_dir, GLOBAL_CONFIG_DIR, GLOBAL_CONFIG_FILE
from splatworld_agent.profile import ProfileManager
from splatworld_agent.models import TasteProfile, Feedback, Generation, PromptHistoryEntry, ExplorationMode
from splatworld_agent.learning import LearningEngine, enhance_prompt, PromptAdapter
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


@main.command("migrate-data")
@click.option("--from-dir", type=click.Path(exists=True), help="Source directory to migrate from")
@click.option("--dry-run", is_flag=True, help="Show what would be migrated without copying")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompts")
def migrate_data(from_dir: Optional[str], dry_run: bool, yes: bool):
    """Migrate user data from old .splatworld_agent/ folder.

    Detects old installation in current directory or specified path
    and migrates profile, feedback, and session data to .splatworld/

    Files migrated:
    - profile.json (taste profile)
    - feedback.jsonl (rating history)
    - sessions.jsonl (session history)
    - prompt_history.jsonl (prompt tracking)
    - image_metadata/ (per-image metadata)
    - metadata/image_registry.json (number registry)
    """
    # Find old directory
    if from_dir:
        old_path = Path(from_dir)
        if old_path.name != ".splatworld_agent":
            old_path = old_path / ".splatworld_agent"
    else:
        # Auto-detect in current directory
        old_path = Path.cwd() / ".splatworld_agent"

    if not old_path.exists():
        console.print(f"[yellow]No old data found at: {old_path}[/yellow]")
        console.print("[dim]Specify --from-dir to migrate from another location[/dim]")
        return

    new_path = Path.cwd() / ".splatworld"

    # Count files to migrate
    files_to_migrate = list(old_path.rglob("*"))
    file_count = len([f for f in files_to_migrate if f.is_file()])

    console.print(f"\n[bold]Found old SplatWorld data:[/bold] {old_path}")
    console.print(f"[dim]Files to migrate: {file_count}[/dim]\n")

    # Show key files
    key_files = ["profile.json", "feedback.jsonl", "sessions.jsonl", "prompt_history.jsonl"]
    for key_file in key_files:
        key_path = old_path / key_file
        if key_path.exists():
            size = key_path.stat().st_size
            console.print(f"  [green]✓[/green] {key_file} ({size} bytes)")
        else:
            console.print(f"  [dim]- {key_file} (not found)[/dim]")

    # Check for image metadata
    metadata_dir = old_path / "image_metadata"
    if metadata_dir.exists():
        meta_count = len(list(metadata_dir.glob("*.json")))
        console.print(f"  [green]✓[/green] image_metadata/ ({meta_count} files)")

    if dry_run:
        console.print("\n[yellow]Dry run - no changes made[/yellow]")
        console.print(f"[dim]Would migrate to: {new_path}[/dim]")
        return

    # Check destination
    if new_path.exists() and (new_path / "profile.json").exists():
        console.print(f"\n[yellow]Warning:[/yellow] {new_path} already has data")

        # Compare profile timestamps
        old_profile = old_path / "profile.json"
        new_profile = new_path / "profile.json"
        if old_profile.exists() and new_profile.exists():
            old_mtime = old_profile.stat().st_mtime
            new_mtime = new_profile.stat().st_mtime
            if old_mtime > new_mtime:
                console.print("[dim]Old data is NEWER - migration recommended[/dim]")
            else:
                console.print("[dim]Existing data is NEWER - migration may overwrite[/dim]")

        if not yes:
            from rich.prompt import Confirm
            if not Confirm.ask("Merge old data into existing? (backs up existing first)"):
                console.print("Migration cancelled")
                return

        # Backup existing
        import datetime as dt
        timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        backup_path = new_path.parent / f".splatworld.backup-{timestamp}"
        shutil.copytree(new_path, backup_path)
        console.print(f"[dim]Backed up existing to: {backup_path}[/dim]")

    # Confirm migration
    if not yes:
        from rich.prompt import Confirm
        if not Confirm.ask(f"Migrate from {old_path} to {new_path}?"):
            console.print("Migration cancelled")
            return

    # Perform migration - preserve timestamps with copy2
    console.print("\n[bold]Migrating files...[/bold]")
    new_path.mkdir(parents=True, exist_ok=True)

    # Use shutil.copytree with dirs_exist_ok for merging
    shutil.copytree(old_path, new_path, dirs_exist_ok=True, copy_function=shutil.copy2)

    console.print(f"\n[green]Migration complete![/green]")
    console.print(f"\n[dim]Old data preserved at: {old_path}[/dim]")
    console.print(f"[dim]To delete old folder: rm -rf \"{old_path}\"[/dim]")


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
        console.print("[red]Error: Not in a SplatWorld project. Run 'splatworld init' first.[/red]")
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
            f"For best results, run [cyan]splatworld train \"{prompt_text}\"[/cyan]\n"
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

    # Create generation ID and get image number for flat structure
    gen_id = f"gen-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    image_number = manager.get_next_image_number()

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

            # Save the generation (dual-write: nested for backward compat, flat for new structure)
            gen_timestamp = datetime.now()
            image_dir, metadata_dir = manager.save_generation(Generation(
                id=gen_id,
                prompt=prompt_text,
                enhanced_prompt=enhanced_prompt,
                timestamp=gen_timestamp,
                metadata={"generator": gen_name, "seed": seed, "image_number": image_number},
            ))

            # Save image to flat structure (N.png)
            flat_image_path = manager.get_flat_image_path(image_number)
            manager.images_dir.mkdir(exist_ok=True)
            with open(flat_image_path, "wb") as f:
                f.write(image_bytes)

            # Also save to nested structure for backward compat
            image_path = image_dir / "source.png"
            with open(image_path, "wb") as f:
                f.write(image_bytes)

            # Save flat metadata
            manager.save_image_metadata(image_number, {
                "id": gen_id,
                "image_number": image_number,
                "prompt": prompt_text,
                "enhanced_prompt": enhanced_prompt,
                "timestamp": gen_timestamp.isoformat(),
                "generator": gen_name,
                "seed": seed,
            })

            # Register mapping from gen_id to image_number
            manager.register_image(gen_id, image_number)

            console.print(f"\n[green]Image {image_number} saved:[/green] {flat_image_path}")

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

                # Download splat file to visible splats directory (flat: N.spz)
                if result.splat_url and config.defaults.download_splats:
                    manager.splats_dir.mkdir(exist_ok=True)
                    splat_path = manager.get_flat_splat_path(image_number)
                    try:
                        marble.download_file(result.splat_url, splat_path)
                        console.print(f"[green]Splat {image_number} saved:[/green] {splat_path}")
                    except Exception as e:
                        splat_path = None
                        error_msg = str(e)
                        if "403" in error_msg or "Forbidden" in error_msg:
                            console.print(f"[yellow]Splat download requires premium account. Viewer URL saved.[/yellow]")
                        else:
                            console.print(f"[yellow]Splat download failed: {error_msg}[/yellow]")
                elif result.splat_url and not config.defaults.download_splats:
                    console.print(f"[dim]Splat download skipped (download_splats=false). Viewer URL saved.[/dim]")

                # Download mesh file
                if result.mesh_url and config.defaults.download_meshes:
                    mesh_path = image_dir / "collision.glb"
                    try:
                        marble.download_file(result.mesh_url, mesh_path)
                        console.print(f"[green]Mesh saved:[/green] {mesh_path}")
                    except Exception as e:
                        mesh_path = None
                        console.print(f"[yellow]Mesh download failed: {e}[/yellow]")

                marble.close()

                console.print(f"[blue]Viewer:[/blue] {result.viewer_url}")
                console.print(f"[dim]Cost: ${result.cost_usd:.2f}[/dim]")

                progress.update(task, description="[green]3D conversion complete!")

            elif not no_splat and not config.api_keys.marble:
                console.print("[yellow]Skipping 3D conversion (no Marble API key configured)[/yellow]")

            # Update generation metadata with paths (nested structure for backward compat)
            metadata_path = metadata_dir / "metadata.json"
            with open(metadata_path) as f:
                gen_data = json.load(f)

            gen_data["source_image_path"] = str(flat_image_path)
            gen_data["image_number"] = image_number
            if splat_path:
                gen_data["splat_path"] = str(splat_path)
            if mesh_path:
                gen_data["mesh_path"] = str(mesh_path)
            # Store URLs for later download via download-splats command
            if not no_splat and config.api_keys.marble:
                gen_data["viewer_url"] = result.viewer_url
                if result.splat_url:
                    gen_data["splat_url"] = result.splat_url

            with open(metadata_path, "w") as f:
                json.dump(gen_data, f, indent=2)

        console.print(f"\n[bold green]Generation complete![/bold green]")
        console.print(f"[dim]Image {image_number} (ID: {gen_id})[/dim]")
        console.print(f"\nUse [cyan]splatworld rate {image_number} ++[/cyan] to love it, or [cyan]splatworld rate {image_number} --[/cyan] to hate it.")

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
        console.print(f"\n[cyan]You have {unprocessed} feedback entries. Consider running 'splatworld learn' to update your taste profile.[/cyan]")


@main.command()
@click.argument("image_nums", nargs=-1, type=int, required=True)
@click.argument("rating", type=click.Choice(["++", "+", "-", "--"]))
def rate(image_nums: tuple, rating: str):
    """Rate images by global image number.

    Image numbers are the sequential numbers assigned during generation
    (e.g., 1.png, 2.png in generated_images/).

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

    # Rate each image
    rating_display = {
        "++": "[green]Love it![/green]",
        "+": "[green]Good[/green]",
        "-": "[yellow]Not great[/yellow]",
        "--": "[red]Hate it[/red]",
    }

    for image_num in image_nums:
        # Try global image number lookup first (new flat structure)
        gen = manager.get_generation_by_number(image_num)
        gen_id = None
        resolved_number = image_num

        if gen:
            gen_id = gen.id
        else:
            # Fall back to batch-relative lookup for backward compatibility
            if manager.current_session_path.exists():
                gen_id = manager.resolve_image_number(image_num)

        if not gen_id:
            console.print(f"[red]Could not resolve image {image_num}.[/red]")
            console.print(f"[dim]Check that the image exists in generated_images/[/dim]")
            continue

        # Create and save feedback (replaces existing if re-rating)
        fb = Feedback(
            generation_id=gen_id,
            timestamp=datetime.now(),
            rating=rating,
        )
        was_replaced, old_rating = manager.add_or_replace_feedback(fb)

        if was_replaced:
            console.print(f"Rated Image {resolved_number}: {rating_display[old_rating]} -> {rating_display[rating]} [dim](updated)[/dim]")
        else:
            console.print(f"Rated Image {resolved_number}: {rating_display[rating]}")

    # Show quick stats
    if len(image_nums) > 1:
        console.print(f"\n[dim]Rated {len(image_nums)} images as {rating}[/dim]")

    # Suggest next steps
    profile = manager.load_profile()
    unprocessed = len(manager.get_unprocessed_feedback())
    config = Config.load()
    if unprocessed >= config.defaults.auto_learn_threshold:
        console.print(f"\n[cyan]You have {unprocessed} feedback entries. Consider running 'splatworld learn'.[/cyan]")


@main.command("brate")
@click.argument("ratings_input", nargs=-1, required=True)
def batch_rate(ratings_input: tuple):
    """Rate multiple images with different ratings in one command.

    Image numbers are the global sequential numbers assigned during generation
    (e.g., 1.png, 2.png in generated_images/).

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

    # Parse the input
    input_str = " ".join(ratings_input)
    pairs = parse_batch_ratings(input_str)

    if not pairs:
        console.print("[red]Could not parse ratings.[/red]")
        console.print("[yellow]Format: 1 ++ 2 - 3 -- 4 +[/yellow]")
        sys.exit(1)

    # Process each rating
    rating_display = {
        "++": "[green]Love it![/green]",
        "+": "[green]Good[/green]",
        "-": "[yellow]Not great[/yellow]",
        "--": "[red]Hate it[/red]",
    }

    for image_num, rating in pairs:
        # Try global image number lookup first (new flat structure)
        gen = manager.get_generation_by_number(image_num)
        gen_id = None

        if gen:
            gen_id = gen.id
        else:
            # Fall back to batch-relative lookup for backward compatibility
            if manager.current_session_path.exists():
                gen_id = manager.resolve_image_number(image_num)

        if not gen_id:
            console.print(f"[red]Could not resolve image {image_num}.[/red]")
            console.print(f"[dim]Check that the image exists in generated_images/[/dim]")
            continue

        fb = Feedback(
            generation_id=gen_id,
            timestamp=datetime.now(),
            rating=rating,
        )

        was_replaced, old_rating = manager.add_or_replace_feedback(fb)

        if was_replaced:
            console.print(f"Rated Image {image_num}: {rating_display[old_rating]} -> {rating_display[rating]} [dim](updated)[/dim]")
        else:
            console.print(f"Rated Image {image_num}: {rating_display[rating]}")

    # Check for missing ratings in current batch (RATE-02)
    # Only applies if there's an active batch context
    if manager.current_session_path.exists():
        try:
            with open(manager.current_session_path) as f:
                session = json.load(f)
            batch_gen_ids = session.get("batch_generation_ids", [])
            if batch_gen_ids:
                unrated = manager.get_unrated_in_batch(batch_gen_ids)
                if unrated:
                    console.print(f"\n[yellow]Unrated images in current batch: {', '.join(map(str, unrated))}[/yellow]")
                    if click.confirm("Rate remaining batch images?", default=True):
                        for img_num in unrated:
                            rating_input = click.prompt(
                                f"Rating for batch image {img_num}",
                                type=click.Choice(["++", "+", "-", "--", "s"]),
                                default="s"
                            )
                            if rating_input != "s":
                                gen_id = manager.resolve_image_number(img_num)
                                fb = Feedback(
                                    generation_id=gen_id,
                                    timestamp=datetime.now(),
                                    rating=rating_input,
                                )
                                manager.add_or_replace_feedback(fb)
                                console.print(f"Rated Image {img_num}: {rating_display[rating_input]}")
        except (json.JSONDecodeError, IOError):
            pass  # No batch context, skip unrated check

    # Summary
    console.print(f"\n[green]Rated {len(pairs)} images.[/green]")

    # Suggest next steps
    profile = manager.load_profile()
    unprocessed = len(manager.get_unprocessed_feedback())
    config = Config.load()
    if unprocessed >= config.defaults.auto_learn_threshold:
        console.print(f"\n[cyan]You have {unprocessed} feedback entries. Consider running 'splatworld learn'.[/cyan]")


@main.command()
@click.argument("prompt", nargs=-1, required=True)
@click.option("--count", "-n", default=5, help="Number of images to generate per cycle")
@click.option("--cycles", "-c", default=1, help="Number of cycles (generate, review, learn)")
@click.option("--generator", type=click.Choice(["nano", "gemini"]), default=None, help="Image generator to use")
@click.option("--mode", "-m", type=click.Choice(["explore", "refine"]), default=None, help="Exploration mode (explore=diverse, refine=targeted)")
@click.option("--inline", is_flag=True, default=False, help="Show inline image previews (iTerm2/Kitty/WezTerm)")
@click.option("--single-cycle", is_flag=True, help="Run one cycle without interactive pause (non-interactive mode)")
def batch(prompt: tuple, count: int, cycles: int, generator: str, mode: str, inline: bool, single_cycle: bool):
    """Generate a batch of images for review.

    Generates N images, then lets you review and rate them before
    optionally running more cycles with learned preferences.

    Exploration modes (MODE-01/MODE-02):
        --mode explore  Diverse variants across dimensions (default)
        --mode refine   Small targeted tweaks to what works

    Example workflow:
        splatworld batch "cozy cabin interior" -n 5 -c 2
        # Generates 5 images, you review them, learns preferences,
        # then generates 5 more with updated taste profile

        splatworld batch "cozy cabin" --mode refine
        # Fine-tune variants based on what's already working
    """
    prompt_text = " ".join(prompt)

    project_dir = get_project_dir()
    if not project_dir:
        console.print("[red]Error: Not in a SplatWorld project. Run 'splatworld init' first.[/red]")
        sys.exit(1)

    manager = ProfileManager(project_dir.parent)
    config = Config.load()

    # Determine exploration mode (command line overrides config)
    exploration_mode = ExplorationMode.from_string(mode or config.defaults.exploration_mode)

    # Validate config
    issues = config.validate()
    if issues:
        console.print("[red]Configuration issues:[/red]")
        for issue in issues:
            console.print(f"  - {issue}")
        sys.exit(1)

    gen_name = generator or config.defaults.image_generator

    # Single-cycle mode: force exactly one cycle, no interactive pause
    if single_cycle:
        cycles = 1

    # Check calibration status
    profile = manager.load_profile()
    if not profile.is_calibrated:
        console.print(Panel.fit(
            f"[yellow]Profile not calibrated[/yellow]\n\n"
            f"{profile.training_progress}\n\n"
            f"Consider running [cyan]splatworld train \"{prompt_text}\"[/cyan]\n"
            f"for guided training (generates, reviews, learns until calibrated).\n\n"
            f"[dim]Continuing with batch generation...[/dim]",
            title="Training Recommended",
        ))

    # Show mode info
    mode_display = "explore (diverse)" if exploration_mode == ExplorationMode.EXPLORE_WIDE else "refine (targeted)"
    console.print(f"[dim]Mode: {mode_display}[/dim]")

    for cycle_num in range(1, cycles + 1):
        console.print(Panel.fit(
            f"[bold]Cycle {cycle_num}/{cycles}[/bold]\n"
            f"Generating {count} images for: {prompt_text}\n"
            f"[dim]Mode: {mode_display}[/dim]",
            title="Batch Generation",
        ))

        profile = manager.load_profile()
        enhanced_prompt = enhance_prompt(prompt_text, profile)

        if enhanced_prompt != prompt_text:
            console.print(f"[dim]Enhanced with taste:[/dim] {enhanced_prompt}")

        # Track this batch
        batch_id = f"batch-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        batch_gen_ids = []
        batch_image_numbers = []  # Track image numbers for flat structure

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
                    image_number = manager.get_next_image_number()
                    batch_gen_ids.append(gen_id)
                    batch_image_numbers.append(image_number)

                    image_bytes = img_gen.generate(enhanced_prompt, seed=None)

                    # Save generation (dual-write: nested for backward compat, flat for new structure)
                    gen_timestamp = datetime.now()
                    image_dir, metadata_dir = manager.save_generation(Generation(
                        id=gen_id,
                        prompt=prompt_text,
                        enhanced_prompt=enhanced_prompt,
                        timestamp=gen_timestamp,
                        metadata={"generator": gen_name, "batch_id": batch_id, "batch_index": i, "image_number": image_number},
                    ))

                    # Save image to flat structure (N.png)
                    flat_image_path = manager.get_flat_image_path(image_number)
                    manager.images_dir.mkdir(exist_ok=True)
                    with open(flat_image_path, "wb") as f:
                        f.write(image_bytes)

                    # Also save to nested structure for backward compat
                    image_path = image_dir / "source.png"
                    with open(image_path, "wb") as f:
                        f.write(image_bytes)

                    # Save flat metadata
                    manager.save_image_metadata(image_number, {
                        "id": gen_id,
                        "image_number": image_number,
                        "prompt": prompt_text,
                        "enhanced_prompt": enhanced_prompt,
                        "timestamp": gen_timestamp.isoformat(),
                        "generator": gen_name,
                        "batch_id": batch_id,
                        "batch_index": i,
                    })

                    # Register mapping from gen_id to image_number
                    manager.register_image(gen_id, image_number)

                    # Update nested metadata with paths
                    metadata_path = metadata_dir / "metadata.json"
                    with open(metadata_path) as f:
                        gen_data = json.load(f)
                    gen_data["source_image_path"] = str(flat_image_path)
                    gen_data["image_number"] = image_number
                    with open(metadata_path, "w") as f:
                        json.dump(gen_data, f, indent=2)

                    progress.update(task, description=f"[green]Image {image_number} saved ({i+1}/{count})")

            img_gen.close()

            # Store batch context for numbered references
            manager.set_current_batch(batch_id, batch_gen_ids)

            console.print(f"\n[bold green]Batch complete![/bold green] Generated {count} images.")
            console.print(f"[dim]Batch ID: {batch_id}[/dim]")

            # Show image list with image numbers
            first_num = batch_image_numbers[0] if batch_image_numbers else 1
            last_num = batch_image_numbers[-1] if batch_image_numbers else count

            if inline:
                # Inline preview mode
                console.print("\n[bold]Your images:[/bold]")
                for img_num in batch_image_numbers:
                    flat_path = manager.get_flat_image_path(img_num)
                    console.print(f"\n[cyan]Image {img_num}[/cyan]")
                    displayed = display.display_image(flat_path, max_width=60)
                    if displayed:
                        console.print(f"[dim]{flat_path}[/dim]")
                    else:
                        console.print(f"  {flat_path}")
            else:
                # Non-intrusive mode (default) - DISP-03
                console.print(f"\n[bold]Images {first_num}-{last_num} generated.[/bold] View externally, then rate here.")
                console.print(f"[dim]Files saved to: {manager.images_dir}[/dim]")

            # Show rating instructions with actual image numbers
            console.print("\n[bold]Rate your images:[/bold]")
            console.print(f"  [cyan]splatworld rate {first_num} ++[/cyan]  - Love image {first_num}")
            console.print(f"  [cyan]splatworld rate {first_num + 2 if len(batch_image_numbers) > 2 else first_num} -[/cyan]   - Dislike image")
            console.print(f"  [cyan]splatworld rate {first_num} {last_num} +[/cyan] - Like multiple images")

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
    console.print("  1. [cyan]splatworld rate 1 ++[/cyan] - Rate images by number")
    console.print("  2. [cyan]splatworld learn[/cyan] - Learn from your feedback")
    console.print("  3. [cyan]splatworld convert[/cyan] - Convert favorites to 3D splats")


@main.command()
@click.option("--batch", "-b", help="Review specific batch ID")
@click.option("--all", "all_unrated", is_flag=True, help="Review ALL unrated images across all batches")
@click.option("--current", "-c", is_flag=True, help="Review current batch (default if no batch specified)")
@click.option("--limit", "-n", default=10, help="Number of images to review")
@click.option("--unrated", is_flag=True, help="Only show unrated images")
@click.option("--inline", is_flag=True, default=False, help="Show inline image previews (iTerm2/Kitty/WezTerm)")
@click.option("--list", "list_only", is_flag=True, help="List unrated images without prompting for ratings")
@click.option("--rate", "rate_value", help="Rate a generation (++/+/-/--)")
@click.option("--generation", "-g", "rate_gen_id", help="Generation ID to rate (use with --rate)")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON (for skill file parsing)")
def review(batch: str, current: bool, limit: int, unrated: bool, all_unrated: bool, inline: bool, list_only: bool, rate_value: str, rate_gen_id: str, json_output: bool):
    """Review and rate generated images.

    Usage:
      review --list                 List unrated images
      review --list --json          List unrated images as JSON (for skill files)
      review --rate ++ -g <id>      Rate a specific generation
      review --all                  Interactive review (terminal only)
    """
    project_dir = get_project_dir()
    if not project_dir:
        console.print("[red]Error: Not in a SplatWorld project.[/red]")
        sys.exit(1)

    manager = ProfileManager(project_dir.parent)

    # Non-interactive mode: Rate a specific generation
    if rate_value and rate_gen_id:
        if rate_value not in ("++", "+", "-", "--"):
            console.print(f"[red]Invalid rating: {rate_value}. Use ++, +, -, or --[/red]")
            sys.exit(1)

        gen = manager.get_generation(rate_gen_id)
        if not gen:
            console.print(f"[red]Generation not found: {rate_gen_id}[/red]")
            sys.exit(1)

        fb = Feedback(
            generation_id=gen.id,
            timestamp=datetime.now(),
            rating=rate_value,
        )
        manager.add_or_replace_feedback(fb)

        rating_display = {"++": "[green]Love it![/green]", "+": "[green]Good[/green]",
                          "-": "[yellow]Not great[/yellow]", "--": "[red]Hate it[/red]"}
        # Show image number if available, otherwise show legacy ID
        image_number = manager.get_image_number_for_generation(gen.id)
        if image_number:
            console.print(f"Rated Image {image_number}: {rating_display[rate_value]}")
        else:
            console.print(f"Rated Legacy: {gen.id[:8]}...: {rating_display[rate_value]}")
        console.print(f"[dim]Prompt: {gen.prompt[:50]}...[/dim]")
        return

    # Non-interactive mode: List unrated images
    if list_only:
        unrated_gens = manager.get_all_unrated_generations()

        # JSON output mode: output structured data (for skill file parsing)
        if json_output:
            if not unrated_gens:
                print(json.dumps([]))
                return
            items = []
            for gen, batch_ctx in unrated_gens[:limit]:
                image_number = manager.get_image_number_for_generation(gen.id)
                items.append({
                    "generation_id": gen.id,
                    "image_number": image_number,
                    "file_path": str(manager.get_flat_image_path(image_number)) if image_number else gen.source_image_path,
                    "prompt": gen.prompt,
                    "created_at": gen.created_at.isoformat() if gen.created_at else None,
                })
            print(json.dumps(items))
            return

        if not unrated_gens:
            console.print("[green]All images have been rated![/green]")
            return

        # Build list with image numbers
        lines = []
        for g, _ in unrated_gens[:limit]:
            image_num = manager.get_image_number_for_generation(g.id)
            if image_num:
                lines.append(f"  [cyan]Image {image_num}[/cyan]: {g.prompt[:45]}...")
            else:
                lines.append(f"  [dim]Legacy: {g.id[:8]}...[/dim]: {g.prompt[:45]}...")

        console.print(Panel.fit(
            f"[bold]Unrated Images ({len(unrated_gens)})[/bold]\n\n" +
            "\n".join(lines),
            title="Review",
        ))
        console.print(f"\n[dim]Use 'splatworld rate N RATING' to rate an image[/dim]")
        return

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
            # Show image number if available, otherwise show legacy ID
            image_number = manager.get_image_number_for_generation(gen.id)
            if image_number:
                console.print(f"\n[bold cyan]Image {image_number}[/bold cyan] ({i} of {total})")
                # Show flat file path
                flat_path = f"generated_images/{image_number}.png"
            else:
                console.print(f"\n[bold cyan]Legacy: {gen.id[:8]}...[/bold cyan] ({i} of {total})")
                flat_path = None

            console.print(format_batch_context(batch_ctx))
            console.print(f"Prompt: {gen.prompt}")

            # Show file path
            if flat_path:
                if inline:
                    # Inline preview mode - try flat path first
                    flat_full_path = manager.images_dir / f"{image_number}.png"
                    if flat_full_path.exists():
                        displayed = display.display_image(flat_full_path, max_width=80)
                        if displayed:
                            console.print("[dim]Preview shown above[/dim]")
                        else:
                            console.print(f"[cyan]File: {flat_path}[/cyan]")
                    elif gen.source_image_path:
                        displayed = display.display_image(Path(gen.source_image_path), max_width=80)
                        if displayed:
                            console.print("[dim]Preview shown above[/dim]")
                        else:
                            console.print(f"[cyan]File: {flat_path}[/cyan]")
                else:
                    console.print(f"[cyan]File: {flat_path}[/cyan]")
            elif gen.source_image_path:
                if inline:
                    # Inline preview mode for legacy images
                    displayed = display.display_image(Path(gen.source_image_path), max_width=80)
                    if displayed:
                        console.print("[dim]Preview shown above[/dim]")
                    else:
                        console.print(f"[cyan]File: {gen.source_image_path}[/cyan]")
                else:
                    # Non-intrusive mode (default) - DISP-05
                    console.print(f"[cyan]File: {gen.source_image_path}[/cyan]")

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
            console.print(f"\n[cyan]Run 'splatworld convert' to turn your {loved} loved images into 3D splats.[/cyan]")

        if reviewed >= 3:
            console.print(f"[cyan]Run 'splatworld learn' to update your taste profile.[/cyan]")

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
            console.print("[yellow]No current batch. Generate one with: splatworld batch \"prompt\"[/yellow]")
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
        # Show image number if available, otherwise show legacy ID
        image_number = manager.get_image_number_for_generation(gen.id)
        if image_number:
            console.print(f"\n[bold cyan]Image {image_number}[/bold cyan] ({i} of {len(generations)})")
            flat_path = f"generated_images/{image_number}.png"
        else:
            console.print(f"\n[bold cyan]Legacy: {gen.id[:8]}...[/bold cyan] ({i} of {len(generations)})")
            flat_path = None

        # Show batch index if available
        batch_index = gen.metadata.get("batch_index")
        if batch_index is not None:
            console.print(f"[dim]Batch position: {batch_index + 1}[/dim]")

        console.print(f"Prompt: {gen.prompt}")

        # Show image path
        if flat_path:
            if inline:
                # Inline preview mode - try flat path first
                flat_full_path = manager.images_dir / f"{image_number}.png"
                if flat_full_path.exists():
                    displayed = display.display_image(flat_full_path, max_width=80)
                    if displayed:
                        console.print("[dim]Preview shown above[/dim]")
                    else:
                        console.print(f"[cyan]File: {flat_path}[/cyan]")
                elif gen.source_image_path:
                    displayed = display.display_image(Path(gen.source_image_path), max_width=80)
                    if displayed:
                        console.print("[dim]Preview shown above[/dim]")
                    else:
                        console.print(f"[cyan]File: {flat_path}[/cyan]")
            else:
                console.print(f"[cyan]File: {flat_path}[/cyan]")
        elif gen.source_image_path:
            if inline:
                # Inline preview mode for legacy images
                displayed = display.display_image(Path(gen.source_image_path), max_width=80)
                if displayed:
                    console.print("[dim]Preview shown above[/dim]")
                else:
                    console.print(f"[cyan]File: {gen.source_image_path}[/cyan]")
            else:
                # Non-intrusive mode (default) - DISP-05
                console.print(f"[cyan]File: {gen.source_image_path}[/cyan]")

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
                was_replaced, old_rating = manager.add_or_replace_feedback(fb)

                rating_display = {"++": "[green]Love it![/green]", "+": "[green]Good[/green]",
                                  "-": "[yellow]Not great[/yellow]", "--": "[red]Hate it[/red]"}
                if was_replaced:
                    console.print(f"Updated: {rating_display[rating_input]} (was {old_rating})")
                else:
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
        console.print(f"\n[cyan]Run 'splatworld convert' to turn your {loved} loved images into 3D splats.[/cyan]")

    if reviewed >= 3:
        console.print(f"[cyan]Run 'splatworld learn' to update your taste profile.[/cyan]")


@main.command()
@click.argument("image_nums", nargs=-1, type=int, required=False)
@click.option("--all-loved", is_flag=True, help="Convert all '++' rated images")
@click.option("--all-positive", is_flag=True, help="Convert all positively rated (+ and ++)")
@click.option("--list", "list_only", is_flag=True, help="Just list available images, don't convert")
def convert(image_nums: tuple, all_loved: bool, all_positive: bool, list_only: bool):
    """Convert loved images to 3D splats.

    Usage:
      convert --list              Show available images
      convert 1 3                 Convert images 1 and 3
      convert --all-loved         Convert all '++' rated images

    Accepts image numbers (from batch or review commands).
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

    # Get image registry for number lookups
    registry = manager.get_image_registry()
    # Build reverse mapping: gen_id -> image_number
    id_to_number = {gen_id: num for gen_id, num in registry.items()}

    # Find generations that could be converted
    generations = manager.get_recent_generations(limit=50)
    feedbacks = {f.generation_id: f for f in manager.get_feedback_history()}

    available = []
    for gen in generations:
        fb = feedbacks.get(gen.id)
        if not fb:
            continue
        if gen.splat_path:
            continue
        if fb.rating in ("++", "+"):
            # Get image number (from registry or metadata)
            img_num = id_to_number.get(gen.id) or gen.metadata.get("image_number")
            available.append((gen, fb.rating, img_num))

    if not available:
        console.print("[yellow]No images available for conversion.[/yellow]")
        console.print("[dim]Rate images with '++' to mark them for conversion.[/dim]")
        return

    # Determine what to convert
    to_convert = []  # List of (gen, image_number) tuples

    if image_nums:
        # Specific image numbers requested
        for img_num in image_nums:
            gen = manager.get_generation_by_number(img_num)
            if gen:
                to_convert.append((gen, img_num))
            else:
                console.print(f"[yellow]Image not found: {img_num}[/yellow]")
    elif all_loved:
        # All '++' rated
        to_convert = [(g, n) for g, r, n in available if r == "++"]
    elif all_positive:
        # All '+' and '++' rated
        to_convert = [(g, n) for g, r, n in available]

    # If just listing or no action specified, show available and exit
    if list_only or (not image_nums and not all_loved and not all_positive):
        console.print(Panel.fit(
            f"[bold]Found {len(available)} images ready for 3D conversion[/bold]\n\n"
            "Available images:\n" +
            "\n".join(f"  [cyan]Image {n or '?'}[/cyan] [{r}]: {g.prompt[:45]}..." for g, r, n in available),
            title="3D Conversion",
        ))
        console.print("\n[dim]Use image numbers to convert specific images, or --all-loved to convert all.[/dim]")
        console.print("[dim]Example: splatworld convert 1 3 5[/dim]")
        return

    if not to_convert:
        console.print("[yellow]No matching images to convert.[/yellow]")
        return

    # Show what will be converted
    cost = len(to_convert) * 1.50
    console.print(f"\n[bold]Converting {len(to_convert)} image(s)[/bold] -- Estimated cost: ${cost:.2f}")
    for g, img_num in to_convert:
        console.print(f"  [cyan]Image {img_num or '?'}[/cyan]: {g.prompt[:50]}...")

    # Convert each image
    from splatworld_agent.core.marble import MarbleClient

    marble = MarbleClient(api_key=config.api_keys.marble)
    total_cost = 0.0
    converted = 0
    converted_results = []  # Track results to show links at end

    console.print("\n[bold cyan]Starting 3D conversion...[/bold cyan]")
    console.print("[dim]This may take a few minutes per image.[/dim]\n")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console,
        ) as progress:
            for i, (gen, img_num) in enumerate(to_convert):
                task = progress.add_task(f"Converting {i+1}/{len(to_convert)}: Image {img_num or '?'}...", total=None)

                # Load image from flat path if available, otherwise nested
                flat_path = manager.get_flat_image_path(img_num) if img_num else None
                if flat_path and flat_path.exists():
                    image_source_path = flat_path
                elif gen.source_image_path and Path(gen.source_image_path).exists():
                    image_source_path = Path(gen.source_image_path)
                else:
                    progress.update(task, description=f"[red]Missing image: Image {img_num or '?'}[/red]")
                    continue

                with open(image_source_path, "rb") as f:
                    image_bytes = f.read()

                image_b64 = base64.b64encode(image_bytes).decode()

                # Show generating status
                progress.update(task, description=f"[cyan]Generating 3D splat for Image {img_num or '?'}...[/cyan]")

                def on_progress(status: str, description: str):
                    status_text = description or status
                    progress.update(task, description=f"[cyan]Image {img_num or '?'}:[/cyan] {status_text}")

                try:
                    result = marble.generate_and_wait(
                        image_base64=image_b64,
                        mime_type="image/png",
                        display_name=f"Image {img_num}" if img_num else gen.id,
                        is_panorama=True,
                        on_progress=on_progress,
                    )
                except Exception as api_error:
                    progress.update(task, description=f"[red]API error for Image {img_num or '?'}: {api_error}[/red]")
                    continue

                progress.update(task, description=f"[green]Generated splat for Image {img_num or '?'}[/green]")

                # Download splat file to visible splats directory (flat: N.spz)
                splat_path = None
                mesh_path = None

                if result.splat_url and config.defaults.download_splats:
                    manager.splats_dir.mkdir(exist_ok=True)
                    # Use image number for flat structure
                    splat_path = manager.get_flat_splat_path(img_num) if img_num else manager.splats_dir / f"{gen.id}.spz"
                    try:
                        marble.download_file(result.splat_url, splat_path)
                    except Exception as e:
                        splat_path = None
                        error_msg = str(e)
                        if "403" in error_msg or "Forbidden" in error_msg:
                            console.print(f"[yellow]Splat download requires premium account. Viewer URL saved.[/yellow]")
                        else:
                            console.print(f"[yellow]Splat download failed: {error_msg}[/yellow]")
                        console.print(f"[dim]Run 'splatworld download-splats' later to retry.[/dim]")
                elif result.splat_url and not config.defaults.download_splats:
                    console.print(f"[dim]Splat download skipped (download_splats=false). Viewer URL saved.[/dim]")

                # Download mesh file to splats directory (flat structure)
                if result.mesh_url and config.defaults.download_meshes:
                    mesh_path = manager.splats_dir / f"{img_num}-collision.glb" if img_num else manager.splats_dir / f"{gen.id}-collision.glb"
                    try:
                        marble.download_file(result.mesh_url, mesh_path)
                    except Exception as e:
                        mesh_path = None
                        console.print(f"[yellow]Mesh download failed: {e}[/yellow]")

                # Update metadata in hidden directory
                metadata_dir = manager.get_metadata_dir(gen.id)
                if not metadata_dir:
                    console.print(f"[yellow]Could not find metadata for Image {img_num or '?'}[/yellow]")
                    continue
                metadata_path = metadata_dir / "metadata.json"
                with open(metadata_path) as f:
                    gen_data = json.load(f)

                if splat_path:
                    gen_data["splat_path"] = str(splat_path)
                if mesh_path:
                    gen_data["mesh_path"] = str(mesh_path)
                gen_data["viewer_url"] = result.viewer_url
                # Store splat_url for later download via download-splats command
                if result.splat_url:
                    gen_data["splat_url"] = result.splat_url

                with open(metadata_path, "w") as f:
                    json.dump(gen_data, f, indent=2)

                # Also update flat metadata if available
                if img_num:
                    flat_metadata = manager.load_image_metadata(img_num)
                    if flat_metadata:
                        if splat_path:
                            flat_metadata["splat_path"] = str(splat_path)
                        if mesh_path:
                            flat_metadata["mesh_path"] = str(mesh_path)
                        flat_metadata["viewer_url"] = result.viewer_url
                        if result.splat_url:
                            flat_metadata["splat_url"] = result.splat_url
                        manager.save_image_metadata(img_num, flat_metadata)

                total_cost += result.cost_usd
                converted += 1
                converted_results.append((img_num, result.viewer_url, splat_path))
                progress.update(task, description=f"[green]Converted: Image {img_num or '?'}[/green]")

    finally:
        marble.close()

    console.print(f"\n[bold green]Conversion complete![/bold green]")
    console.print(f"Converted: {converted}/{len(to_convert)} images")
    console.print(f"Total cost: [yellow]${total_cost:.2f}[/yellow]")

    # Show viewer links for all converted images
    if converted_results:
        console.print("\n[bold]View your 3D splats:[/bold]")
        for img_num, viewer_url, splat_path in converted_results:
            console.print(f"  [cyan]Image {img_num or '?'}[/cyan]")
            if viewer_url:
                console.print(f"    Marble Labs: [link={viewer_url}]{viewer_url}[/link]")
            if splat_path:
                console.print(f"    Local file: {splat_path}")


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
            console.print("[dim]Use 'splatworld profile' to view your updated taste profile.[/dim]")

    except Exception as e:
        console.print(f"\n[red]Learning failed:[/red] {e}")
        sys.exit(1)


@main.command()
@click.argument("args", nargs=-1, required=False)
@click.option("--count", "-n", default=None, type=int, help="Number of images to generate (default: infinite until stopped)")
@click.option("--generator", type=click.Choice(["nano", "gemini"]), default=None, help="Image generator")
@click.option("--no-rate", "no_rate", is_flag=True, help="Generate without prompting for ratings (rate later with review)")
@click.option("--single", is_flag=True, help="Generate exactly one image and exit (for scripted workflows)")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON (for skill file parsing)")
def train(args: tuple, count: int, generator: str, no_rate: bool, single: bool, json_output: bool):
    """Adaptive training mode - generates images with prompt adaptation.

    Usage:
      train "prompt"                  Interactive training (terminal only)
      train "prompt" --single         Generate ONE image and exit
      train "prompt" -n 5 --no-rate   Generate 5 images, rate later with review
      train "prompt" --single --json  Generate ONE image, output JSON (for skill files)

    For Claude Code / scripted use, always use --single or --no-rate.
    """
    project_dir = get_project_dir()
    if not project_dir:
        console.print("[red]Error: Not in a SplatWorld project. Run 'splatworld init' first.[/red]")
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
    gen_name = generator or config.defaults.image_generator

    # Check for existing training state to resume
    training_state = _load_training_state(manager)

    # Parse args: first arg could be count (if numeric) or start of prompt
    # /train 5 -> count=5, prompt from state
    # /train "my prompt" -> count=None, prompt="my prompt"
    # /train 5 "my prompt" -> count=5, prompt="my prompt"
    prompt_parts = []
    for i, arg in enumerate(args):
        if i == 0 and count is None and arg.isdigit():
            count = int(arg)
        else:
            prompt_parts.append(arg)

    # Handle --single flag
    if single:
        count = 1
        no_rate = True

    # Determine the prompt to use
    if prompt_parts:
        prompt_text = " ".join(prompt_parts)
    elif training_state and training_state.get("base_prompt"):
        prompt_text = training_state["base_prompt"]
        console.print(f"[cyan]Resuming training with:[/cyan] {prompt_text}")
    else:
        console.print("[red]Error: No prompt provided and no training state to resume.[/red]")
        console.print("[dim]Usage: splatworld train \"your prompt\"[/dim]")
        console.print("[dim]       splatworld train 5  (to train 5 images with saved prompt)[/dim]")
        sys.exit(1)

    # Initialize training session
    session_id = f"train-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"

    # Initialize or restore training state
    if training_state and training_state.get("base_prompt") == prompt_text:
        images_generated = training_state.get("images_generated", 0)
        last_variant_id = training_state.get("last_variant_id")
        session_id = training_state.get("session_id", session_id)
        console.print(f"[dim]Resuming from image {images_generated + 1}[/dim]")
    else:
        images_generated = 0
        last_variant_id = None
        # Start fresh training state
        _save_training_state(manager, {
            "session_id": session_id,
            "base_prompt": prompt_text,
            "images_generated": 0,
            "last_variant_id": None,
            "started_at": datetime.now().isoformat(),
            "status": "active",
        })

    # Default to gemini if not specified (no interactive prompt needed)
    if not generator:
        gen_name = "gemini"
        console.print("[dim]Using Gemini for image generation (use --generator nano for Nano Banana Pro)[/dim]")

    # Display training panel
    target_str = f"{count} images" if count else "until stopped"
    console.print(Panel.fit(
        f"[bold]Adaptive Training Mode[/bold]\n\n"
        f"Base prompt: [cyan]{prompt_text}[/cyan]\n"
        f"Target: {target_str}\n"
        f"Generator: {gen_name}\n"
        f"Progress: {images_generated} images generated\n\n"
        f"[bold]During training:[/bold]\n"
        f"  Rate each image: [green]++[/green]=love [green]+[/green]=like [yellow]-[/yellow]=meh [red]--[/red]=hate\n"
        f"  Skip: [dim]s[/dim] | Cancel: [dim]cancel[/dim] or Ctrl+C\n"
        f"  Change prompt: Type [cyan]new: your new prompt[/cyan]\n\n"
        f"[dim]Claude will adapt variants based on your ratings.[/dim]",
        title="SplatWorld Training",
    ))

    # Initialize generators (both for fallback support)
    from splatworld_agent.generators.nano import NanoGenerator
    from splatworld_agent.generators.gemini import GeminiGenerator

    nano_gen = NanoGenerator(api_key=config.api_keys.nano or config.api_keys.google)
    gemini_gen = GeminiGenerator(api_key=config.api_keys.google)

    # Primary generator based on choice (bidirectional fallback)
    if gen_name == "nano":
        img_gen = nano_gen
        fallback_gen = gemini_gen
        fallback_name = "gemini"
    else:
        img_gen = gemini_gen
        fallback_gen = nano_gen
        fallback_name = "nano"

    # Initialize prompt adapter for variant generation
    adapter = PromptAdapter(api_key=config.api_keys.anthropic)

    # Get recent feedback for context
    recent_feedback = _get_recent_feedback_for_adapter(manager, limit=10)

    cancelled = False
    images_since_prompt_check = 0
    PROMPT_CHECK_INTERVAL = 5  # ADAPT-05: Suggest prompt change every 5 images

    try:
        while True:
            # Check count limit
            if count and images_generated >= count:
                console.print(f"\n[green]Reached target of {count} images![/green]")
                break

            images_generated += 1
            images_since_prompt_check += 1

            console.print(f"\n[bold cyan]--- Image {images_generated} ---[/bold cyan]")

            # Generate variant using PromptAdapter (ADAPT-02, ADAPT-03)
            profile = manager.load_profile()

            try:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    task = progress.add_task("Generating prompt variant...", total=None)

                    variant = adapter.generate_variant(
                        base_prompt=prompt_text,
                        profile=profile,
                        recent_feedback=recent_feedback,
                    )

                    progress.update(task, description="Variant generated!")

                # ADAPT-04: Show reasoning
                console.print(f"\n[bold]Variant:[/bold] {variant.variant_prompt}")
                if variant.reasoning:
                    console.print(f"[dim]Reasoning: {variant.reasoning}[/dim]")
                if variant.modifications:
                    console.print(f"[dim]Modifications: {', '.join(variant.modifications[:3])}[/dim]")

            except Exception as e:
                # Fall back to enhanced prompt if adapter fails
                console.print(f"[yellow]Adapter failed, using enhanced prompt: {e}[/yellow]")
                variant = None
                enhanced_prompt = enhance_prompt(prompt_text, profile)

            # Generate image
            gen_id = f"gen-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
            variant_id = f"var-{uuid.uuid4().hex[:12]}"

            final_prompt = variant.variant_prompt if variant else enhanced_prompt
            actual_generator = gen_name  # Track which generator was actually used

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task(f"Generating image with {gen_name}...", total=None)

                try:
                    image_bytes = img_gen.generate(final_prompt, seed=None)
                except Exception as gen_error:
                    # Check if this is a recoverable error and we have a fallback
                    error_str = str(gen_error).lower()
                    is_recoverable_error = any(term in error_str for term in [
                        "quota", "rate limit", "exhausted", "limit exceeded",
                        "too many requests", "429", "resource_exhausted",
                        "404", "not found", "not available", "does not exist",
                        "model", "invalid", "503", "500", "unavailable"
                    ])

                    if is_recoverable_error and fallback_gen is not None:
                        progress.update(task, description=f"[yellow]{gen_name} failed, trying {fallback_name}...")
                        console.print(f"\n[yellow]⚠ {gen_name} error: {gen_error}[/yellow]")
                        console.print(f"[yellow]  Switching to {fallback_name}...[/yellow]")

                        try:
                            image_bytes = fallback_gen.generate(final_prompt, seed=None)
                            actual_generator = fallback_name
                            # Switch to fallback for remaining images
                            img_gen = fallback_gen
                            gen_name = fallback_name
                            fallback_gen = None
                            console.print(f"[green]✓ Switched to {gen_name} for remaining images[/green]")
                        except Exception as fallback_error:
                            console.print(f"\n[red]Both generators failed:[/red]")
                            console.print(f"  {gen_name}: {gen_error}")
                            console.print(f"  {fallback_name}: {fallback_error}")
                            console.print("[dim]Try again later when quota resets.[/dim]")
                            cancelled = True
                            break
                    else:
                        console.print(f"\n[red]Image generation failed: {gen_error}[/red]")
                        if fallback_gen is None:
                            console.print("[dim]No fallback generator available.[/dim]")
                        cancelled = True
                        break

                # Get global image number for flat structure
                image_number = manager.get_next_image_number()

                # Save generation (dual-write: nested for backward compat, flat for new structure)
                gen_timestamp = datetime.now()
                image_dir, metadata_dir = manager.save_generation(Generation(
                    id=gen_id,
                    prompt=prompt_text,
                    enhanced_prompt=final_prompt,
                    timestamp=gen_timestamp,
                    metadata={
                        "generator": actual_generator,
                        "training_session": session_id,
                        "variant_id": variant_id,
                        "image_number": image_number,
                    },
                ))

                # Save image to flat structure (N.png)
                flat_image_path = manager.get_flat_image_path(image_number)
                manager.images_dir.mkdir(exist_ok=True)
                with open(flat_image_path, "wb") as f:
                    f.write(image_bytes)

                # Also save to nested structure for backward compat
                image_path = image_dir / "source.png"
                with open(image_path, "wb") as f:
                    f.write(image_bytes)

                # Save flat metadata
                manager.save_image_metadata(image_number, {
                    "id": gen_id,
                    "image_number": image_number,
                    "prompt": prompt_text,
                    "enhanced_prompt": final_prompt,
                    "timestamp": gen_timestamp.isoformat(),
                    "generator": actual_generator,
                    "training_session": session_id,
                    "variant_id": variant_id,
                })

                # Register mapping from gen_id to image_number
                manager.register_image(gen_id, image_number)

                # Update nested metadata with paths
                metadata_path = metadata_dir / "metadata.json"
                with open(metadata_path) as f:
                    gen_data = json.load(f)
                gen_data["source_image_path"] = str(flat_image_path)
                gen_data["image_number"] = image_number
                with open(metadata_path, "w") as f:
                    json.dump(gen_data, f, indent=2)

                progress.update(task, description=f"[green]Image {image_number} generated with {actual_generator}!")

            console.print(f"[green]Image {image_number} saved:[/green] {flat_image_path}")

            # Don't auto-open image - let user view on their own
            # The file path is shown above so they can open it manually

            # Save prompt history entry (unrated initially)
            history_entry = PromptHistoryEntry(
                variant_id=variant_id,
                base_prompt=prompt_text,
                variant_prompt=final_prompt,
                rating=None,
                parent_variant_id=last_variant_id,
                reasoning=variant.reasoning if variant else "",
                generation_id=gen_id,
                session_id=session_id,
            )
            manager.save_prompt_variant(history_entry)

            # JSON output mode: output structured data and exit (for skill file parsing)
            if json_output and single:
                result = {
                    "image_number": image_number,
                    "generation_id": gen_id,
                    "file_path": str(flat_image_path),
                    "prompt": final_prompt,
                    "variant_id": variant_id,
                }
                print(json.dumps(result))
                return  # Exit without interactive loop

            # Get rating (skip if --no-rate flag is set)
            rating = None
            if no_rate:
                # Non-interactive mode: skip rating, user can rate later with review
                console.print(f"[dim]Generated: Image {image_number} ({gen_id})[/dim]")
                console.print(f"[dim]Rate later with: splatworld review --rate RATING {image_number}[/dim]")
            else:
                # Interactive mode: prompt for rating
                while True:
                    try:
                        rating_input = input("\nRating (++/+/-/--/s/cancel) or 'new: prompt': ").strip()
                    except (KeyboardInterrupt, EOFError):
                        cancelled = True
                        break

                    # Check for cancel
                    if rating_input.lower() == "cancel":
                        cancelled = True
                        break

                    # Check for prompt change (ADAPT-06)
                    if rating_input.lower().startswith("new:"):
                        new_prompt = rating_input[4:].strip()
                        if new_prompt:
                            prompt_text = new_prompt
                            console.print(f"[cyan]Base prompt changed to:[/cyan] {prompt_text}")
                            images_since_prompt_check = 0  # Reset counter
                            # Update training state with new prompt
                            _save_training_state(manager, {
                                "session_id": session_id,
                                "base_prompt": prompt_text,
                                "images_generated": images_generated,
                                "last_variant_id": variant_id,
                                "started_at": training_state.get("started_at") if training_state else datetime.now().isoformat(),
                                "status": "active",
                            })
                            # Clear recent feedback context since we're exploring new territory
                            recent_feedback = []
                        else:
                            console.print("[yellow]Please provide a prompt after 'new:'[/yellow]")
                        continue

                    # Check for skip - exits training without forcing more ratings
                    if rating_input.lower() == "s":
                        console.print("[dim]Skipping ratings - you can review unrated images later with /splatworld:review[/dim]")
                        cancelled = True
                        break

                    # Parse rating
                    if rating_input in ("++", "+", "-", "--"):
                        rating = rating_input

                        # Save feedback
                        fb = Feedback(
                            generation_id=gen_id,
                            timestamp=datetime.now(),
                            rating=rating,
                        )
                        manager.add_or_replace_feedback(fb)

                    # Update prompt history with rating
                    manager.update_prompt_variant_rating(variant_id, rating)

                    rating_display = {
                        "++": "[green]Love![/green]",
                        "+": "[green]Good[/green]",
                        "-": "[yellow]Meh[/yellow]",
                        "--": "[red]Hate[/red]"
                    }
                    console.print(rating_display[rating])

                    # Update recent feedback for next adaptation (ADAPT-03)
                    gen = manager.get_generation(gen_id)
                    if gen:
                        recent_feedback.append((gen, fb))
                        # Keep only last 10
                        recent_feedback = recent_feedback[-10:]

                    break
                else:
                    console.print("[red]Invalid. Use ++, +, -, --, s, cancel, or 'new: prompt'[/red]")

            if cancelled:
                break

            # Update training state
            last_variant_id = variant_id
            _save_training_state(manager, {
                "session_id": session_id,
                "base_prompt": prompt_text,
                "images_generated": images_generated,
                "last_variant_id": variant_id,
                "started_at": training_state.get("started_at") if training_state else datetime.now().isoformat(),
                "status": "active",
            })

            # ADAPT-05: Suggest prompt change every 5 images (interactive mode only)
            if not no_rate and images_since_prompt_check >= PROMPT_CHECK_INTERVAL:
                console.print(f"\n[cyan]You've generated {images_since_prompt_check} images with this prompt.[/cyan]")
                console.print("Continue with same prompt, or type 'new: your new prompt' to change.")

                try:
                    change_input = input("Press Enter to continue, or enter new prompt: ").strip()
                except (KeyboardInterrupt, EOFError):
                    cancelled = True
                    break

                if change_input.lower().startswith("new:"):
                    new_prompt = change_input[4:].strip()
                    if new_prompt:
                        prompt_text = new_prompt
                        console.print(f"[cyan]Base prompt changed to:[/cyan] {prompt_text}")
                        recent_feedback = []  # Clear context for new prompt
                elif change_input and not change_input.lower().startswith("new:"):
                    # Treat bare input as new prompt
                    prompt_text = change_input
                    console.print(f"[cyan]Base prompt changed to:[/cyan] {prompt_text}")
                    recent_feedback = []

                images_since_prompt_check = 0

            # Learn periodically (every 5 ratings)
            profile = manager.load_profile()
            if profile.stats.feedback_count > 0 and profile.stats.feedback_count % 5 == 0:
                try:
                    console.print("[dim]Learning from recent feedback...[/dim]")
                    engine = LearningEngine(api_key=config.api_keys.anthropic)
                    generations = manager.get_recent_generations(limit=20)
                    feedbacks = manager.get_feedback_history()
                    result = engine.synthesize_from_history(generations, feedbacks, profile)

                    if result.get("updates"):
                        profile = engine.apply_updates(profile, result["updates"])
                        profile.calibration.last_learn_at = datetime.now()
                        profile.calibration.learn_count += 1

                        # Check calibration
                        can_calibrate, _ = profile.stats.can_calibrate()
                        if can_calibrate and not profile.calibration.is_calibrated:
                            profile.calibration.is_calibrated = True
                            profile.calibration.calibrated_at = datetime.now()
                            console.print("[bold green]Profile calibrated![/bold green]")

                        manager.save_profile(profile)
                except Exception as e:
                    console.print(f"[dim]Learning skipped: {e}[/dim]")

    except KeyboardInterrupt:
        cancelled = True

    finally:
        img_gen.close()

        # Update training state
        status = "cancelled" if cancelled else "completed"
        _save_training_state(manager, {
            "session_id": session_id,
            "base_prompt": prompt_text,
            "images_generated": images_generated,
            "last_variant_id": last_variant_id,
            "started_at": training_state.get("started_at") if training_state else datetime.now().isoformat(),
            "ended_at": datetime.now().isoformat(),
            "status": status,
        })

    # Show summary
    profile = manager.load_profile()
    console.print(Panel.fit(
        f"[bold]Training {'Cancelled' if cancelled else 'Complete'}[/bold]\n\n"
        f"Images generated: {images_generated}\n"
        f"Total ratings: {profile.stats.feedback_count}\n"
        f"Profile: {'[green]CALIBRATED[/green]' if profile.is_calibrated else profile.training_progress}\n\n"
        f"[bold]Resume:[/bold]\n"
        f"  [cyan]splatworld resume[/cyan] - Continue training\n"
        f"  [cyan]splatworld train \"{prompt_text}\"[/cyan] - Continue with same prompt\n\n"
        f"[bold]Next steps:[/bold]\n"
        f"  [cyan]splatworld convert[/cyan] - Convert loved images to 3D splats",
        title="Training Summary",
    ))


def _load_training_state(manager: ProfileManager) -> dict:
    """Load current training state from session file."""
    if not manager.current_session_path.exists():
        return {}

    try:
        with open(manager.current_session_path) as f:
            data = json.load(f)
        # Only return if it's a training session
        if data.get("session_id", "").startswith("train-"):
            return data
        return {}
    except (json.JSONDecodeError, IOError):
        return {}


def _save_training_state(manager: ProfileManager, state: dict) -> None:
    """Save training state to session file."""
    temp_path = manager.current_session_path.with_suffix(".tmp")
    with open(temp_path, "w") as f:
        json.dump(state, f, indent=2)
    temp_path.replace(manager.current_session_path)


def _get_recent_feedback_for_adapter(manager: ProfileManager, limit: int = 10) -> list:
    """Get recent (generation, feedback) pairs for adapter context."""
    feedbacks = manager.get_feedback_history(limit=limit * 2)
    pairs = []

    for fb in feedbacks[-limit:]:
        gen = manager.get_generation(fb.generation_id)
        if gen:
            pairs.append((gen, fb))

    return pairs


@main.command("cancel")
def cancel_training():
    """Stop the current training session gracefully (ADAPT-08).

    Saves the current training state so you can resume later with 'resume'.
    This is equivalent to pressing Ctrl+C during training or typing 'cancel'.

    Example:
        splatworld cancel
    """
    project_dir = get_project_dir()
    if not project_dir:
        console.print("[red]Error: Not in a SplatWorld project.[/red]")
        sys.exit(1)

    manager = ProfileManager(project_dir.parent)

    # Check for active training session
    training_state = _load_training_state(manager)

    if not training_state:
        console.print("[yellow]No active training session to cancel.[/yellow]")
        return

    if training_state.get("status") != "active":
        console.print(f"[yellow]Training session already {training_state.get('status', 'ended')}.[/yellow]")
        return

    # Mark as cancelled
    training_state["status"] = "cancelled"
    training_state["ended_at"] = datetime.now().isoformat()
    _save_training_state(manager, training_state)

    console.print(Panel.fit(
        f"[bold]Training Cancelled[/bold]\n\n"
        f"Session: {training_state.get('session_id', 'unknown')}\n"
        f"Images generated: {training_state.get('images_generated', 0)}\n"
        f"Base prompt: {training_state.get('base_prompt', 'unknown')}\n\n"
        f"[bold]Resume with:[/bold]\n"
        f"  [cyan]splatworld resume[/cyan]\n"
        f"  [cyan]splatworld train[/cyan] (auto-resumes)",
        title="Training Stopped",
    ))


@main.command("resume")
def resume_training():
    """Continue an interrupted training session (SESS-01, SESS-02).

    Prompts: "Continue training on unrated images, or start new generations?"

    - Continue unrated: Shows unrated images from previous session for rating
    - Start new: Resumes generating new images with same base prompt

    Example:
        splatworld resume
    """
    project_dir = get_project_dir()
    if not project_dir:
        console.print("[red]Error: Not in a SplatWorld project.[/red]")
        sys.exit(1)

    manager = ProfileManager(project_dir.parent)

    # Check for training state
    training_state = _load_training_state(manager)

    if not training_state:
        console.print("[yellow]No training session to resume.[/yellow]")
        console.print("[dim]Start a new session with: splatworld train \"your prompt\"[/dim]")
        return

    session_id = training_state.get("session_id", "unknown")
    base_prompt = training_state.get("base_prompt", "")
    images_generated = training_state.get("images_generated", 0)
    status = training_state.get("status", "unknown")

    console.print(Panel.fit(
        f"[bold]Training Session Found[/bold]\n\n"
        f"Session: {session_id}\n"
        f"Status: {status}\n"
        f"Base prompt: [cyan]{base_prompt}[/cyan]\n"
        f"Images generated: {images_generated}",
        title="Resume Training",
    ))

    # SESS-02: Check for unrated images from this session
    unrated_in_session = []
    all_unrated = manager.get_all_unrated_generations()

    for gen, batch_ctx in all_unrated:
        if gen.metadata.get("training_session") == session_id:
            unrated_in_session.append(gen)

    if unrated_in_session:
        console.print(f"\n[yellow]Found {len(unrated_in_session)} unrated images from this session.[/yellow]")
        console.print("\nOptions:")
        console.print("  [cyan]1[/cyan] - Rate unrated images first, then continue")
        console.print("  [cyan]2[/cyan] - Skip unrated and start new generations")
        console.print("  [cyan]q[/cyan] - Quit without resuming")

        try:
            choice = input("\nChoice (1/2/q): ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            choice = "q"

        if choice == "q":
            console.print("[dim]Resume cancelled.[/dim]")
            return

        if choice == "1":
            # Rate unrated images
            console.print(f"\n[bold]Rating {len(unrated_in_session)} unrated images[/bold]")
            console.print("Rate each: [green]++[/green]=love [green]+[/green]=like [yellow]-[/yellow]=meh [red]--[/red]=hate [dim]s[/dim]=skip")

            rated = 0
            for gen in unrated_in_session:
                console.print(f"\n[dim]Image {rated + 1}/{len(unrated_in_session)}[/dim]")
                console.print(f"Variant: {gen.enhanced_prompt[:80]}...")

                if gen.source_image_path:
                    console.print(f"[dim]File: {gen.source_image_path}[/dim]")
                    # Don't auto-open - user can view in IDE/file explorer

                while True:
                    try:
                        rating_input = input("Rating (++/+/-/--/s/q): ").strip()
                    except (KeyboardInterrupt, EOFError):
                        rating_input = "q"

                    if rating_input.lower() == "q":
                        console.print("[dim]Stopped rating.[/dim]")
                        break
                    elif rating_input.lower() == "s":
                        console.print("[dim]Skipped[/dim]")
                        break
                    elif rating_input in ("++", "+", "-", "--"):
                        fb = Feedback(
                            generation_id=gen.id,
                            timestamp=datetime.now(),
                            rating=rating_input,
                        )
                        manager.add_or_replace_feedback(fb)

                        # Update prompt history if variant_id exists
                        variant_id = gen.metadata.get("variant_id")
                        if variant_id:
                            manager.update_prompt_variant_rating(variant_id, rating_input)

                        rating_display = {"++": "[green]Love![/green]", "+": "[green]Good[/green]",
                                          "-": "[yellow]Meh[/yellow]", "--": "[red]Hate[/red]"}
                        console.print(rating_display[rating_input])
                        rated += 1
                        break
                    else:
                        console.print("[red]Invalid. Use ++, +, -, --, s, or q[/red]")

                if rating_input.lower() == "q":
                    break

            console.print(f"\n[green]Rated {rated} images.[/green]")

    # Now continue with new generations
    console.print(f"\n[cyan]Continuing training with: {base_prompt}[/cyan]")
    console.print("[dim]Run 'splatworld train' to generate new images.[/dim]")

    # Reactivate session for train command to pick up
    training_state["status"] = "active"
    _save_training_state(manager, training_state)


@main.command("install-prompts")
def install_prompts():
    """Install Claude Code slash command prompts."""
    prompts_dir = Path(__file__).parent.parent / "prompts"
    target_dir = Path.home() / ".claude" / "splatworld"

    if not prompts_dir.exists():
        console.print("[red]Prompts directory not found in package.[/red]")
        sys.exit(1)

    target_dir.mkdir(parents=True, exist_ok=True)

    import shutil
    for prompt_file in prompts_dir.glob("*.md"):
        shutil.copy2(prompt_file, target_dir / prompt_file.name)
        console.print(f"Installed: {prompt_file.name}")

    console.print(f"\n[green]Prompts installed to {target_dir}[/green]")
    console.print("You can now use /splatworld:* commands in Claude Code.")


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
        f"  Exploration mode: {cfg.defaults.exploration_mode}\n"
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
@click.argument("new_mode", required=False, type=click.Choice(["explore", "refine"]))
def mode(new_mode: str):
    """View or set the exploration mode (MODE-03).

    Exploration modes control how prompt variants are generated:

    \b
    explore - Generate DIVERSE variants across multiple dimensions
              (lighting, mood, composition, atmosphere, colors)
              Best for: discovering what you like, exploring possibilities

    \b
    refine  - Make SMALL targeted tweaks to successful elements
              (subtle variations on what's working)
              Best for: fine-tuning after finding promising directions

    Examples:
        splatworld mode           # Show current mode
        splatworld mode explore   # Switch to explore mode
        splatworld mode refine    # Switch to refine mode

    You can switch modes at any time during a training session.
    """
    cfg = Config.load()

    if new_mode is None:
        # Show current mode
        current_mode = cfg.defaults.exploration_mode
        mode_desc = {
            "explore": "EXPLORE WIDELY - Diverse variants across dimensions",
            "refine": "REFINE NARROWLY - Small targeted tweaks",
        }.get(current_mode, current_mode)

        console.print(Panel.fit(
            f"[bold]Current Mode:[/bold] {current_mode}\n\n"
            f"{mode_desc}\n\n"
            f"[dim]Switch modes:[/dim]\n"
            f"  [cyan]splatworld mode explore[/cyan] - Try diverse approaches\n"
            f"  [cyan]splatworld mode refine[/cyan]  - Fine-tune what works",
            title="Exploration Mode",
        ))
    else:
        # Set new mode
        old_mode = cfg.defaults.exploration_mode
        cfg.defaults.exploration_mode = new_mode
        cfg.save()

        mode_desc = {
            "explore": "Generating DIVERSE variants across dimensions",
            "refine": "Making SMALL targeted tweaks to successful elements",
        }.get(new_mode, new_mode)

        console.print(f"[green]Mode changed:[/green] {old_mode} -> [bold]{new_mode}[/bold]")
        console.print(f"[dim]{mode_desc}[/dim]")

        if new_mode == "refine":
            console.print("\n[dim]Tip: Refine mode works best after you've found styles you like.[/dim]")
        else:
            console.print("\n[dim]Tip: Explore mode helps discover new directions.[/dim]")


@main.command()
def cancel():
    """Cancel any ongoing SplatWorld action.

    Stops training sessions, conversions, or other operations.
    State is preserved so you can resume later.
    """
    project_dir = get_project_dir()
    if not project_dir:
        console.print("[yellow]No active SplatWorld project found.[/yellow]")
        return

    manager = ProfileManager(project_dir.parent)

    # Check for active training session
    if manager.current_session_path.exists():
        try:
            with open(manager.current_session_path) as f:
                session_data = json.load(f)

            if session_data.get("type") == "training" and session_data.get("status") == "active":
                # Mark as cancelled
                session_data["status"] = "cancelled"
                session_data["ended_at"] = datetime.now().isoformat()

                with open(manager.current_session_path, "w") as f:
                    json.dump(session_data, f, indent=2)

                console.print("[green]Training session cancelled.[/green]")
                console.print(f"[dim]Images generated: {session_data.get('images_generated', 0)}[/dim]")
                console.print(f"\n[cyan]Resume with:[/cyan] splatworld resume")
                return
        except Exception:
            pass

    console.print("[green]Cancelled.[/green]")
    console.print("[dim]Any pending operations have been stopped.[/dim]")
    console.print("[dim]Use /splatworld:resume-work to continue later.[/dim]")


@main.command()
def update():
    """Update SplatWorld Agent to the latest version.

    Pulls the latest changes from the git repository.
    Shows current and available versions during update.
    """
    import subprocess

    # Find the package directory
    package_dir = Path(__file__).parent.parent
    version_file = package_dir / "VERSION"

    # Read current version
    current_version = __version__

    # Check if it's a git repo
    git_dir = package_dir / ".git"
    if not git_dir.exists():
        console.print("[red]Error: SplatWorld Agent is not installed from git.[/red]")
        console.print(f"[dim]Package location: {package_dir}[/dim]")
        sys.exit(1)

    console.print(f"[bold]SplatWorld Agent[/bold] v{current_version}")
    console.print(f"[dim]Location: {package_dir}[/dim]")

    try:
        # Fetch first to see what's available (with 30-second timeout)
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Fetching updates...", total=None)

            # Fetch with timeout
            try:
                result = subprocess.run(
                    ["git", "fetch"],
                    cwd=package_dir,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
            except subprocess.TimeoutExpired:
                console.print("\n[red]Error: Network timeout while fetching updates.[/red]")
                console.print("[yellow]Check your internet connection and try again.[/yellow]")
                sys.exit(1)

            if result.returncode != 0:
                console.print(f"\n[red]Fetch failed:[/red] {result.stderr}")
                sys.exit(1)

            # Get remote version if available
            remote_version = None
            try:
                result = subprocess.run(
                    ["git", "show", "origin/main:VERSION"],
                    cwd=package_dir,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    remote_version = result.stdout.strip()
            except (subprocess.TimeoutExpired, Exception):
                pass  # Remote version display is optional

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
                console.print(f"\n[green]SplatWorld Agent v{current_version} is already up to date.[/green]")
                return

            progress.update(task, description=f"Found {len(new_commits)} new commits...")

            # Show version transition if available
            if remote_version and remote_version != current_version:
                console.print(f"\n[bold cyan]Updating: v{current_version} -> v{remote_version}[/bold cyan]")
            else:
                console.print(f"\n[bold]New updates available:[/bold]")

            # Show what's coming
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
                timeout=60,
            )

            if result.returncode != 0:
                error_msg = result.stderr.strip()
                console.print(f"\n[red]Pull failed:[/red] {error_msg}")
                if "local changes" in error_msg.lower() or "would be overwritten" in error_msg.lower():
                    console.print("\n[yellow]You have local changes that would be overwritten.[/yellow]")
                    console.print("Options:")
                    console.print(f"  1. Stash changes:  cd {package_dir} && git stash && git pull && git stash pop")
                    console.print(f"  2. Discard changes: cd {package_dir} && git checkout . && git pull")
                else:
                    console.print("[yellow]Try pulling manually:[/yellow]")
                    console.print(f"  cd {package_dir} && git pull --ff-only")
                sys.exit(1)

            progress.update(task, description="[green]Update complete!")

        # Read the new version after update
        new_version = current_version
        if version_file.exists():
            new_version = version_file.read_text().strip()

        # Show success panel with version info
        version_info = f"v{new_version}" if new_version == current_version else f"v{current_version} -> v{new_version}"
        console.print(Panel.fit(
            f"[bold green]Updated Successfully![/bold green]\n\n"
            f"Version: {version_info}\n"
            f"Commits: {len(new_commits)} new commit(s)\n\n"
            f"[dim]Run '/splatworld:help' to see new commands.[/dim]",
            title="SplatWorld Agent",
        ))

    except FileNotFoundError:
        console.print("[red]Error: git not found. Please install git.[/red]")
        sys.exit(1)
    except subprocess.TimeoutExpired:
        console.print("\n[red]Error: Operation timed out.[/red]")
        console.print("[yellow]Check your internet connection and try again.[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Update failed:[/red] {e}")
        sys.exit(1)


@main.command()
def version():
    """Show SplatWorld Agent version information.

    Displays version, install location, and git commit info if available.
    """
    import subprocess

    # Find the package directory
    package_dir = Path(__file__).parent.parent
    git_dir = package_dir / ".git"

    # Build version info
    lines = [
        f"[bold]SplatWorld Agent[/bold] v{__version__}",
        f"[dim]Location: {package_dir}[/dim]",
    ]

    # Add git info if available
    if git_dir.exists():
        try:
            # Get current commit
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=package_dir,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                commit_hash = result.stdout.strip()

                # Get commit date
                result = subprocess.run(
                    ["git", "log", "-1", "--format=%ci", "HEAD"],
                    cwd=package_dir,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                commit_date = result.stdout.strip()[:10] if result.returncode == 0 else "unknown"

                lines.append(f"[dim]Commit: {commit_hash} ({commit_date})[/dim]")

            # Check if there are uncommitted changes
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=package_dir,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                lines.append("[yellow]Local modifications present[/yellow]")

        except (subprocess.TimeoutExpired, Exception):
            pass  # Git info is optional

    console.print("\n".join(lines))


@main.command()
def help():
    """Show help and available commands."""
    console.print(Panel.fit(
        "[bold]SplatWorld Agent[/bold]\n"
        "Iterative 3D splat generation with taste learning.\n\n"
        "[bold]Adaptive Training (Recommended):[/bold]\n"
        "  train          Adaptive training - one image at a time\n"
        "  resume         Continue interrupted training session\n"
        "  cancel         Stop training gracefully\n"
        "  learn          Manually run learning on feedback\n\n"
        "[bold]Session Management:[/bold]\n"
        "  resume-work    Resume from previous work session\n"
        "  exit           Save session and exit\n\n"
        "[bold]Batch Workflow:[/bold]\n"
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
        "[dim]Use 'splatworld COMMAND --help' for command details.[/dim]",
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
        console.print("[dim]Start a session with 'splatworld resume-work' first.[/dim]")
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
        + f"\n\n[dim]Session saved. Use 'splatworld resume-work' to continue.[/dim]",
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
            f"[dim]Use 'splatworld exit' to end this session first,[/dim]\n"
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
        console.print(f"  [dim]Run 'splatworld review --unrated' to rate them[/dim]")

    # Check for loved images without splats
    loved_without_splats = []
    for gen in recent_gens:
        fb = feedbacks.get(gen.id)
        if fb and fb.rating == "++" and not gen.splat_path:
            loved_without_splats.append(gen)

    if loved_without_splats:
        console.print(f"\n  [cyan]Loved images ready for conversion: {len(loved_without_splats)}[/cyan]")
        console.print(f"  [dim]Run 'splatworld convert' to create 3D splats[/dim]")

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


@main.command()
@click.argument("image_nums", nargs=-1, type=int, required=False)
@click.option("--open", "open_num", type=int, help="Open viewer URL for a specific image number")
@click.pass_context
def splats(ctx: click.Context, image_nums: tuple, open_num: int = None) -> None:
    """List all converted splats with their World Labs viewer URLs.

    Usage:
      splats                      List all splats
      splats 1 3                  Show splats for images 1 and 3
      splats --open 1             Open viewer for image 1
    """
    project_dir = get_project_dir()
    if not project_dir:
        console.print("[red]Error: Not in a SplatWorld project.[/red]")
        sys.exit(1)

    manager = ProfileManager(project_dir.parent)
    generations = manager.get_all_generations()

    # Get image registry for number lookups
    registry = manager.get_image_registry()
    # Build reverse mapping: gen_id -> image_number
    id_to_number = {gen_id: num for gen_id, num in registry.items()}

    # Filter to only generations with viewer_url and enrich with image numbers
    splat_gens = []
    for g in generations:
        if g.viewer_url:
            img_num = id_to_number.get(g.id) or g.metadata.get("image_number")
            splat_gens.append((g, img_num))

    if not splat_gens:
        console.print("[yellow]No converted splats found.[/yellow]")
        console.print("[dim]Run 'splatworld convert' to create 3D splats from loved images.[/dim]")
        return

    # Handle --open option
    if open_num:
        # Find the generation by image number
        gen_tuple = next((t for t in splat_gens if t[1] == open_num), None)
        if not gen_tuple:
            console.print(f"[red]No splat found for Image {open_num}[/red]")
            return
        gen, _ = gen_tuple
        import webbrowser
        console.print(f"[blue]Opening viewer for Image {open_num}...[/blue]")
        webbrowser.open(gen.viewer_url)
        return

    # Filter to specific image numbers if provided
    if image_nums:
        filtered = [(g, n) for g, n in splat_gens if n in image_nums]
        if not filtered:
            console.print(f"[yellow]No splats found for images: {image_nums}[/yellow]")
            return
        splat_gens = filtered

    # List all splats
    console.print(f"\n[bold]Converted Splats ({len(splat_gens)})[/bold]\n")

    for gen, img_num in sorted(splat_gens, key=lambda t: t[0].timestamp, reverse=True):
        console.print(f"[cyan]Image {img_num or '?'}[/cyan]")
        console.print(f"  Prompt: {gen.prompt[:60]}{'...' if len(gen.prompt) > 60 else ''}")
        console.print(f"  [blue]Viewer: {gen.viewer_url}[/blue]")
        console.print()

    console.print("[dim]Tip: Use --open N to open a viewer directly, e.g.: splatworld splats --open 1[/dim]")


@main.command("download-splats")
@click.argument("image_nums", nargs=-1, type=int, required=False)
@click.option("--all", "download_all", is_flag=True, help="Download all missing splats without confirmation")
@click.option("--list", "list_only", is_flag=True, help="List missing splats without downloading")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON (use with --list)")
def download_splats(image_nums: tuple, download_all: bool, list_only: bool, json_output: bool):
    """Download splat files that haven't been downloaded yet.

    Usage:
      download-splats                  List missing splats, prompt to download
      download-splats 1 3              Download splats for images 1 and 3
      download-splats --all            Download all missing splats

    Note: Downloading splats may require a premium WorldLabs account.
    """
    project_dir = get_project_dir()
    if not project_dir:
        console.print("[red]Error: Not in a SplatWorld project.[/red]")
        sys.exit(1)

    config = Config.load()
    if not config.api_keys.marble:
        console.print("[red]Error: Marble API key required for splat downloads.[/red]")
        console.print("Set WORLDLABS_API_KEY environment variable.")
        sys.exit(1)

    manager = ProfileManager(project_dir.parent)
    generations = manager.get_all_generations()

    # Get image registry for number lookups
    registry = manager.get_image_registry()
    # Build reverse mapping: gen_id -> image_number
    id_to_number = {gen_id: num for gen_id, num in registry.items()}

    # Find generations with splat_url but missing local splat file
    missing_splats = []
    for gen in generations:
        if gen.splat_url:
            # Check if local splat file exists
            if gen.splat_path and Path(gen.splat_path).exists():
                continue  # Already have local file
            # Get image number
            img_num = id_to_number.get(gen.id) or gen.metadata.get("image_number")
            missing_splats.append((gen, img_num))

    # Filter to specific image numbers if provided
    if image_nums:
        filtered = [(g, n) for g, n in missing_splats if n in image_nums]
        # Also check if requested numbers exist but have splat_url
        for img_num in image_nums:
            gen = manager.get_generation_by_number(img_num)
            if gen and gen.splat_url and not any(t[1] == img_num for t in filtered):
                # Check if it's actually downloaded
                splat_path = manager.get_flat_splat_path(img_num)
                if not splat_path.exists():
                    filtered.append((gen, img_num))
        missing_splats = filtered

    # Handle --list --json mode: output JSON array and exit
    if list_only and json_output:
        output = []
        for gen, img_num in missing_splats:
            output.append({
                "generation_id": gen.id,
                "image_number": img_num,
                "prompt": gen.prompt,
                "viewer_url": gen.viewer_url,
            })
        print(json.dumps(output, indent=2))
        return

    if not missing_splats:
        console.print("[green]All splats are downloaded![/green]")
        console.print("[dim]No missing splat files found.[/dim]")
        return

    # Handle --list mode without JSON: show list and exit
    if list_only:
        console.print(f"\n[bold]Missing Splats ({len(missing_splats)})[/bold]\n")
        for gen, img_num in missing_splats:
            console.print(f"[cyan]Image {img_num or '?'}[/cyan]")
            console.print(f"  Prompt: {gen.prompt[:50]}{'...' if len(gen.prompt) > 50 else ''}")
            console.print(f"  [blue]Viewer: {gen.viewer_url}[/blue]")
            console.print()
        return

    console.print(f"\n[bold]Missing Splats ({len(missing_splats)})[/bold]\n")

    for gen, img_num in missing_splats:
        console.print(f"[cyan]Image {img_num or '?'}[/cyan]")
        console.print(f"  Prompt: {gen.prompt[:50]}{'...' if len(gen.prompt) > 50 else ''}")
        console.print(f"  [blue]Viewer: {gen.viewer_url}[/blue]")
        console.print()

    # If specific images were requested, proceed without confirmation
    if not download_all and not image_nums:
        try:
            confirm = input(f"\nDownload {len(missing_splats)} splat file(s)? (y/N): ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            confirm = "n"

        if confirm != "y":
            console.print("[yellow]Download cancelled.[/yellow]")
            return

    # Download missing splats to visible splats directory
    from splatworld_agent.core.marble import MarbleClient

    marble = MarbleClient(api_key=config.api_keys.marble)
    downloaded = 0
    failed = 0

    # Ensure splats directory exists
    manager.splats_dir.mkdir(exist_ok=True)

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            for i, (gen, img_num) in enumerate(missing_splats):
                task = progress.add_task(f"Downloading {i+1}/{len(missing_splats)}: Image {img_num or '?'}...", total=None)

                # Save splat to visible splats directory using image number (flat: N.spz)
                splat_path = manager.get_flat_splat_path(img_num) if img_num else manager.splats_dir / f"{gen.id}.spz"

                try:
                    marble.download_file(gen.splat_url, splat_path)

                    # Update metadata in hidden directory
                    metadata_dir = manager.get_metadata_dir(gen.id)
                    if metadata_dir:
                        metadata_path = metadata_dir / "metadata.json"
                        if metadata_path.exists():
                            with open(metadata_path) as f:
                                gen_data = json.load(f)
                            gen_data["splat_path"] = str(splat_path)
                            with open(metadata_path, "w") as f:
                                json.dump(gen_data, f, indent=2)

                    # Also update flat metadata
                    if img_num:
                        flat_metadata = manager.load_image_metadata(img_num)
                        if flat_metadata:
                            flat_metadata["splat_path"] = str(splat_path)
                            manager.save_image_metadata(img_num, flat_metadata)

                    downloaded += 1
                    progress.update(task, description=f"[green]Downloaded: Image {img_num or '?'}[/green]")

                except Exception as e:
                    failed += 1
                    error_msg = str(e)
                    if "403" in error_msg or "Forbidden" in error_msg:
                        progress.update(task, description=f"[red]Image {img_num or '?'}: Premium account required[/red]")
                    else:
                        progress.update(task, description=f"[red]Image {img_num or '?'}: {error_msg}[/red]")

    finally:
        marble.close()

    console.print(f"\n[bold]Download complete![/bold]")
    console.print(f"  Downloaded: [green]{downloaded}[/green]")
    if failed > 0:
        console.print(f"  Failed: [red]{failed}[/red]")
        console.print(f"\n[yellow]Some downloads failed. This may be due to:[/yellow]")
        console.print(f"  - Premium WorldLabs account required")
        console.print(f"  - Network issues")
        console.print(f"  - Splat no longer available")


@main.command("worlds")
@click.option("--open", "open_id", help="Open viewer URL for a specific world ID")
def list_worlds(open_id: str = None) -> None:
    """List all worlds from your Marble/WorldLabs account.

    Fetches directly from the Marble API to show all worlds
    you've created, not just those tracked locally.

    Examples:
        splatworld worlds                    # List all worlds
        splatworld worlds --open <world_id>  # Open viewer for a world
    """
    config = Config.load()
    if not config.api_keys.marble:
        console.print("[red]Error: Marble API key required.[/red]")
        console.print("Set WORLDLABS_API_KEY environment variable.")
        sys.exit(1)

    from splatworld_agent.core.marble import MarbleClient, MarbleError

    try:
        marble = MarbleClient(api_key=config.api_keys.marble)
        worlds = marble.list_worlds()
        marble.close()
    except MarbleError as e:
        console.print(f"[red]Error fetching worlds: {e}[/red]")
        sys.exit(1)

    if not worlds:
        console.print("[yellow]No worlds found in your account.[/yellow]")
        return

    if open_id:
        # Find and open a specific world
        world = next((w for w in worlds if w.get("world_id") == open_id or w.get("world_id", "").startswith(open_id)), None)
        if not world:
            console.print(f"[red]No world found with ID: {open_id}[/red]")
            return
        import webbrowser
        viewer_url = f"https://marble.worldlabs.ai/world/{world.get('world_id')}"
        console.print(f"[blue]Opening viewer for {world.get('world_id')}...[/blue]")
        webbrowser.open(viewer_url)
        return

    # List all worlds
    console.print(f"\n[bold]Your Marble Worlds ({len(worlds)})[/bold]\n")

    for world in worlds:
        world_id = world.get("world_id", world.get("name", "unknown"))
        display_name = world.get("display_name", "Untitled")
        created = world.get("create_time", "")
        visibility = world.get("visibility", "unknown")

        console.print(f"[cyan]{world_id}[/cyan]")
        console.print(f"  Name: {display_name}")
        if created:
            console.print(f"  Created: {created}")
        console.print(f"  Visibility: {visibility}")
        console.print(f"  [blue]Viewer: https://marble.worldlabs.ai/world/{world_id}[/blue]")
        console.print()

    console.print("[dim]Tip: Use --open <id> to open a viewer directly[/dim]")


@main.command("prompt-history")
@click.option("--limit", "-n", default=20, help="Number of entries to show")
@click.option("--session", "-s", help="Filter to a specific training session ID")
@click.option("--lineage", "-l", help="Show lineage for a specific variant ID")
@click.option("--stats", is_flag=True, help="Show statistics only")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def prompt_history(limit: int, session: str, lineage: str, stats: bool, as_json: bool):
    """View prompt variant history from training sessions (HIST-03).

    Shows all prompt variants tried during training, their ratings,
    and lineage (which variants led to which).

    Examples:
        splatworld prompt-history              # Recent variants
        splatworld prompt-history --stats      # Show statistics
        splatworld prompt-history -l var-xxx   # Show variant lineage
    """
    project_dir = get_project_dir()
    if not project_dir:
        console.print("[red]Error: Not in a SplatWorld project.[/red]")
        sys.exit(1)

    manager = ProfileManager(project_dir.parent)

    # Check if prompt history exists
    if not manager.prompt_history_path.exists():
        console.print("[yellow]No prompt history found.[/yellow]")
        console.print("[dim]Prompt history is recorded during training sessions.[/dim]")
        console.print("[dim]Run 'splatworld train \"prompt\"' to start recording.[/dim]")
        return

    # Stats mode
    if stats:
        history_stats = manager.get_prompt_history_stats()

        if as_json:
            console.print(json.dumps(history_stats, indent=2))
            return

        console.print(Panel.fit(
            f"[bold]Prompt History Statistics[/bold]\n\n"
            f"Total variants: {history_stats['total_variants']}\n"
            f"Rated: {history_stats['rated']} "
            f"([green]{history_stats['positive']}+[/green] / "
            f"[red]{history_stats['negative']}-[/red])\n"
            f"Unrated: {history_stats['unrated']}\n\n"
            f"Unique base prompts: {history_stats['unique_base_prompts']}\n"
            f"Training sessions: {history_stats['training_sessions']}",
            title="Prompt History Stats",
        ))
        return

    # Lineage mode
    if lineage:
        chain = manager.get_variant_lineage(lineage)

        if not chain:
            console.print(f"[red]Variant not found: {lineage}[/red]")
            return

        if as_json:
            console.print(json.dumps([e.to_dict() for e in chain], indent=2))
            return

        console.print(f"\n[bold]Lineage for {lineage}[/bold]\n")

        for i, entry in enumerate(chain):
            prefix = "  " * i
            rating_str = ""
            if entry.rating:
                rating_colors = {"++": "green", "+": "green", "-": "yellow", "--": "red"}
                color = rating_colors.get(entry.rating, "white")
                rating_str = f" [{color}]{entry.rating}[/{color}]"

            connector = "|-> " if i > 0 else ""
            console.print(f"{prefix}{connector}[cyan]{entry.variant_id[:12]}...[/cyan]{rating_str}")
            console.print(f"{prefix}    {entry.variant_prompt[:60]}{'...' if len(entry.variant_prompt) > 60 else ''}")

            if entry.reasoning:
                console.print(f"{prefix}    [dim]Reason: {entry.reasoning[:50]}...[/dim]")

        return

    # Default: list history
    entries = manager.get_prompt_history(limit=limit, session_id=session)

    if not entries:
        if session:
            console.print(f"[yellow]No entries found for session: {session}[/yellow]")
        else:
            console.print("[yellow]No prompt history entries.[/yellow]")
        return

    if as_json:
        console.print(json.dumps([e.to_dict() for e in entries], indent=2))
        return

    # Build table
    table = Table(title=f"Prompt History (last {limit})")
    table.add_column("Variant ID", style="cyan", no_wrap=True)
    table.add_column("Base Prompt")
    table.add_column("Variant")
    table.add_column("Rating", justify="center")
    table.add_column("Parent", style="dim")
    table.add_column("Time")

    for entry in entries:
        rating_str = ""
        if entry.rating:
            rating_colors = {"++": "green", "+": "green", "-": "yellow", "--": "red"}
            color = rating_colors.get(entry.rating, "white")
            rating_str = f"[{color}]{entry.rating}[/{color}]"
        else:
            rating_str = "[dim]-[/dim]"

        parent_str = entry.parent_variant_id[:10] + "..." if entry.parent_variant_id else "[dim]root[/dim]"

        table.add_row(
            entry.variant_id[:13] + "...",
            entry.base_prompt[:18] + ("..." if len(entry.base_prompt) > 18 else ""),
            entry.variant_prompt[:33] + ("..." if len(entry.variant_prompt) > 33 else ""),
            rating_str,
            parent_str,
            entry.timestamp.strftime("%m-%d %H:%M"),
        )

    console.print(table)

    # Show tips
    console.print("\n[dim]Tips:[/dim]")
    console.print("[dim]  --lineage <id>  Show variant evolution chain[/dim]")
    console.print("[dim]  --stats         Show summary statistics[/dim]")
    console.print("[dim]  --session <id>  Filter by training session[/dim]")


@main.command()
@click.option("--dry-run", is_flag=True, help="Show what would be migrated without making changes")
@click.option("--verify", "verify_only", is_flag=True, help="Only show migration status, don't migrate")
def migrate(dry_run: bool, verify_only: bool):
    """Migrate existing images to flat file structure.

    Converts nested structure (generated_images/DATE/UUID/source.png)
    to flat structure (generated_images/N.png) with sequential numbering.

    Migration is idempotent - already-migrated images are skipped.
    Images are numbered chronologically (oldest gets lowest numbers).
    """
    project_dir = get_project_dir()
    if not project_dir:
        console.print("[red]Error: Not in a SplatWorld project. Run 'splatworld init' first.[/red]")
        sys.exit(1)

    manager = ProfileManager(project_dir.parent)

    if verify_only:
        # Just show status
        stats = manager.verify_migration()

        console.print(Panel.fit(
            f"[bold]Migration Status[/bold]\n\n"
            f"[cyan]Nested Structure:[/cyan]\n"
            f"  Images: {stats['nested_images']}\n"
            f"  Splats: {stats['nested_splats']}\n\n"
            f"[cyan]Flat Structure:[/cyan]\n"
            f"  Images: {stats['flat_images']}\n"
            f"  Splats: {stats['flat_splats']}\n\n"
            f"[cyan]Registry:[/cyan]\n"
            f"  Registered IDs: {stats['registered']}\n"
            f"  Next number: {stats['next_number']}\n"
            f"  Unregistered: {len(stats['unregistered'])}",
            title="File Structure Status",
        ))

        if stats['unregistered']:
            console.print(f"\n[yellow]Unregistered images found ({len(stats['unregistered'])}):[/yellow]")
            for uid in stats['unregistered'][:5]:
                console.print(f"  - {uid}")
            if len(stats['unregistered']) > 5:
                console.print(f"  ... and {len(stats['unregistered']) - 5} more")
            console.print("\n[dim]Run 'splatworld migrate' to migrate these images.[/dim]")
        else:
            console.print("\n[green]All images are migrated.[/green]")

        return

    # Run migration
    if dry_run:
        console.print("[yellow]DRY RUN - no changes will be made[/yellow]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Migrating images...", total=None)
        stats = manager.migrate_existing_generations(dry_run=dry_run)
        progress.update(task, completed=True)

    # Display results
    action = "Would migrate" if dry_run else "Migrated"
    console.print(Panel.fit(
        f"[bold]Migration {'Preview' if dry_run else 'Complete'}[/bold]\n\n"
        f"{action}: [green]{stats['migrated']}[/green] images\n"
        f"Skipped (already migrated): [dim]{stats['skipped']}[/dim]\n"
        f"Splats {action.lower()}: [cyan]{stats['splats_migrated']}[/cyan]",
        title="Migration Results",
    ))

    if stats['errors']:
        console.print(f"\n[red]Errors ({len(stats['errors'])}):[/red]")
        for error in stats['errors'][:10]:
            console.print(f"  - {error}")
        if len(stats['errors']) > 10:
            console.print(f"  ... and {len(stats['errors']) - 10} more")

    if dry_run and stats['migrated'] > 0:
        console.print("\n[dim]Run without --dry-run to perform migration.[/dim]")


if __name__ == "__main__":
    main()
