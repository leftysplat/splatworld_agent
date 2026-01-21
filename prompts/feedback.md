<purpose>
Provide feedback on a generation to teach the agent your preferences.
Feedback is used to learn patterns and improve future generations.
</purpose>

<arguments>
- feedback: The feedback text or quick rating
- --generation/-g: Specific generation ID (defaults to most recent)
</arguments>

<quick_ratings>
- ++ or "love it" → Strong positive, agent should do more like this
- + or "good" → Positive
- - or "meh" → Negative
- -- or "hate it" → Strong negative, agent should avoid this
- Any other text → Specific critique that will be analyzed for patterns
</quick_ratings>

<process>
1. Determine if this is a quick rating or detailed feedback
2. Apply to the specified generation (or most recent)
3. Save to feedback log
4. Update profile stats
5. If enough feedback accumulated, suggest running learn
</process>

<execution>
Parse the user's feedback:

```bash
splatworld-agent feedback "<feedback_text>" [--generation <id>]
```

After recording feedback:
1. Confirm what was recorded
2. If 10+ unprocessed feedback entries, suggest: "Consider running `/splatworld-agent:learn` to update your taste profile."

<examples>
User: "love it"
→ splatworld-agent feedback "++"

User: "too dark and cluttered"
→ splatworld-agent feedback "too dark and cluttered"

User: "the lighting is perfect but composition feels off"
→ splatworld-agent feedback "the lighting is perfect but composition feels off"

User: "--" (or "hate it")
→ splatworld-agent feedback "--"
</examples>
</execution>
