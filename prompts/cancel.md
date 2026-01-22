<purpose>
Stop the current training session gracefully (ADAPT-08).
Saves training state so you can resume later.
</purpose>

<process>
1. Check for active training session
2. Mark session as cancelled
3. Save training state with ended_at timestamp
4. Show session summary
5. Suggest resume command
</process>

<execution>
```bash
splatworld-agent cancel
```

This is equivalent to:
- Pressing Ctrl+C during training
- Typing "cancel" during rating prompt
</execution>

<notes>
- Session state is preserved in current_session.json
- Use 'resume' or 'train' to continue later
- Training state includes: prompt, images generated, last variant
- Can also cancel by pressing Ctrl+C during training
</notes>
