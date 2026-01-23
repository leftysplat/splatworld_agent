<purpose>
Show available SplatWorld Agent commands and how to use them.
</purpose>

<execution>
```bash
splatworld help
```

Then provide additional context:

## Quick Start

1. **Initialize**: `/splatworld:init` — Set up .splatworld/ in your project
2. **Generate**: `/splatworld:generate modern kitchen` — Create a splat
3. **Feedback**: `/splatworld:feedback love it` — Teach your preferences
4. **Learn**: `/splatworld:learn` — Update profile from feedback

## All Commands

| Command | Description |
|---------|-------------|
| `/splatworld:init` | Initialize project |
| `/splatworld:generate <prompt>` | Generate with taste enhancement |
| `/splatworld:train <prompt>` | Guided training to calibrate your taste |
| `/splatworld:review` | Review and rate unrated images |
| `/splatworld:resume` | Continue an interrupted training session |
| `/splatworld:feedback <text>` | Rate/critique generation |
| `/splatworld:exemplar <path>` | Add reference image you love |
| `/splatworld:profile` | View taste profile |
| `/splatworld:history` | Browse past generations |
| `/splatworld:learn` | Synthesize feedback into preferences |
| `/splatworld:download-splats` | Download 3D splat files |
| `/splatworld:config` | View configuration |
| `/splatworld:help` | Show this help |

## Interaction Pattern

All user interaction flows through Claude, not the CLI directly.
When you use commands like /train or /review, Claude will:
1. Call the Python CLI with appropriate flags
2. Parse the output
3. Ask you questions via structured prompts
4. Record your responses by calling the CLI again

This provides a consistent, guided experience without raw terminal interaction.

## How Learning Works

1. You generate content and provide feedback
2. Feedback accumulates in .splatworld/feedback.jsonl
3. When you run `/splatworld:learn`, patterns are extracted
4. Your taste profile is updated with learned preferences
5. Future generations are automatically enhanced with your taste
</execution>
