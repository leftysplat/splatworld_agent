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
| `/splatworld:feedback <text>` | Rate/critique generation |
| `/splatworld:exemplar <path>` | Add reference image you love |
| `/splatworld:profile` | View taste profile |
| `/splatworld:history` | Browse past generations |
| `/splatworld:learn` | Synthesize feedback into preferences |
| `/splatworld:config` | View configuration |
| `/splatworld:help` | Show this help |

## How Learning Works

1. You generate content and provide feedback
2. Feedback accumulates in .splatworld/feedback.jsonl
3. When you run `/splatworld:learn`, patterns are extracted
4. Your taste profile is updated with learned preferences
5. Future generations are automatically enhanced with your taste
</execution>
