<purpose>
Continue an interrupted training session (SESS-01, SESS-02).
Offers choice: rate unrated images first, or start new generations.
</purpose>

<process>
1. Check for saved training state
2. Show session summary (prompt, images generated, status)
3. Check for unrated images from session
4. If unrated exist: offer choice
   - Option 1: Rate unrated images first
   - Option 2: Skip unrated, generate new
5. Reactivate session for train command
6. Suggest running train to continue
</process>

<execution>
```bash
splatworld-agent resume
```

If unrated images exist:
```
Options:
  1 - Rate unrated images first, then continue
  2 - Skip unrated and start new generations
  q - Quit without resuming
```
</execution>

<features>
SESS-01: Resume continues interrupted training
SESS-02: Choice to rate unrated vs start new
SESS-03: State persistence (handled by train command)
</features>

<notes>
- Training state saved in current_session.json
- After resume, run 'train' to continue generating
- Unrated images are detected by checking training_session metadata
- Can also resume by running 'train' directly (auto-detects state)
</notes>
