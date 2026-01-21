<purpose>
Synthesize accumulated feedback into updated taste preferences.
Analyzes patterns in your feedback to extract what you consistently like and dislike.
</purpose>

<process>
1. Load feedback history
2. Analyze patterns using Claude
3. Extract preference updates
4. Update taste profile with new learnings
5. Show what was learned
</process>

<execution>
```bash
splatworld-agent learn
```

After learning completes:
1. Show what patterns were identified
2. Show how the profile was updated
3. Show the new prompt enhancement preview
4. Explain that future generations will use these learned preferences
</execution>

<notes>
- Requires at least 3 feedback entries to find patterns
- More feedback = better pattern detection
- Manual profile edits are preserved (learning adds, doesn't replace)
- Run periodically as you accumulate more feedback
</notes>
