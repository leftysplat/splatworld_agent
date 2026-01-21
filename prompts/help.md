<purpose>
Show available SplatWorld Agent commands and how to use them.
</purpose>

<execution>
```bash
splatworld-agent help
```

Then provide additional context:

## Quick Start

1. **Initialize**: `/splatworld-agent:init` — Set up .splatworld/ in your project
2. **Generate**: `/splatworld-agent:generate modern kitchen` — Create a splat
3. **Feedback**: `/splatworld-agent:feedback love it` — Teach your preferences
4. **Learn**: `/splatworld-agent:learn` — Update profile from feedback

## All Commands

| Command | Description |
|---------|-------------|
| `/splatworld-agent:init` | Initialize project |
| `/splatworld-agent:generate <prompt>` | Generate with taste enhancement |
| `/splatworld-agent:feedback <text>` | Rate/critique generation |
| `/splatworld-agent:exemplar <path>` | Add reference image you love |
| `/splatworld-agent:profile` | View taste profile |
| `/splatworld-agent:history` | Browse past generations |
| `/splatworld-agent:learn` | Synthesize feedback into preferences |
| `/splatworld-agent:config` | View configuration |
| `/splatworld-agent:help` | Show this help |

## How Learning Works

1. You generate content and provide feedback
2. Feedback accumulates in .splatworld/feedback.jsonl
3. When you run `/splatworld-agent:learn`, patterns are extracted
4. Your taste profile is updated with learned preferences
5. Future generations are automatically enhanced with your taste
</execution>
