<purpose>
Generate a 3D Gaussian splat from a text prompt, enhanced by the user's learned taste profile.
</purpose>

<arguments>
- prompt: The generation prompt (required, passed as remaining arguments)
- --seed: Optional random seed for reproducibility
- --no-enhance: Skip taste profile enhancement
</arguments>

<process>
1. Verify project is initialized
2. Load taste profile
3. Enhance prompt with learned preferences (unless --no-enhance)
4. Generate image via configured backend (Nano/Gemini)
5. Convert to 3D splat via Marble API
6. Save to .splatworld/generations/
7. Display result and prompt for feedback
</process>

<execution>
Parse the user's request to extract the prompt and any options.

Run the generate command:

```bash
splatworld generate "<prompt>" [--seed <seed>] [--no-enhance]
```

After generation completes:
1. Show the user where files were saved
2. If the taste profile enhanced the prompt, show both original and enhanced
3. Ask for feedback: "What do you think? Use `/splatworld:feedback` to rate this generation."

If generation fails, explain the error and suggest fixes (API keys, network, etc.).
</execution>

<examples>
User: "modern kitchen with marble counters"
→ splatworld generate "modern kitchen with marble counters"

User: "warehouse scene, use seed 42"
→ splatworld generate "warehouse scene" --seed 42

User: "just a simple forest, don't enhance"
→ splatworld generate "simple forest" --no-enhance
</examples>
