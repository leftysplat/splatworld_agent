<purpose>
Adaptive training mode - generates one image at a time with immediate adaptation.
Each rating is used immediately to influence the next variant.
</purpose>

<process>
1. Generate prompt variant using PromptAdapter (based on feedback context)
2. Show variant reasoning to user
3. Generate image from variant
4. User rates immediately (++/+/-/--/s)
5. Adapt next variant based on that rating
6. Repeat until count reached or cancelled
7. Every 5 images, offer to change base prompt
</process>

<execution>
```bash
# Train until stopped
splatworld-agent train "cozy cabin interior"

# Train for specific count
splatworld-agent train "cozy cabin interior" -n 10

# Resume previous session (auto-detects state)
splatworld-agent train
```

During training:
- Rate: ++/+/-/-- (love/like/meh/hate)
- Skip: s
- Change prompt: type "new: your new prompt"
- Cancel: type "cancel" or Ctrl+C
</execution>

<features>
ADAPT-01: One image at a time (not batches)
ADAPT-03: Immediate adaptation after each rating
ADAPT-05: Prompt change suggestion every 5 images
ADAPT-06: Manual prompt change with "new:" syntax
ADAPT-07: Optional count parameter (-n)
ADAPT-08: Graceful cancel
SESS-01: Resume with /resume command
SESS-02: Option to rate unrated images on resume
SESS-03: State persists on exit
</features>

<notes>
- Training state saved in current_session.json
- Use 'cancel' command or Ctrl+C to stop gracefully
- Use 'resume' to continue interrupted session
- Variant reasoning helps understand Claude's choices
</notes>
