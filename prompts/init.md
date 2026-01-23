<purpose>
Initialize SplatWorld Agent in the current project directory.
Creates .splatworld/ with an empty taste profile ready for learning.
</purpose>

<process>
1. Check if already initialized
2. Run the CLI init command
3. Confirm success and explain next steps
</process>

<execution>
Run the following command:

```bash
splatworld init
```

After initialization, explain to the user:
- Their .splatworld/ directory has been created
- Their taste profile starts empty
- They should use `/splatworld:generate` to create content
- They should use `/splatworld:feedback` to teach the agent their preferences
- Over time, the agent will learn and enhance their prompts automatically
</execution>
