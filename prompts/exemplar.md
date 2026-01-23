<purpose>
Add an exemplar image to your taste profile.
Exemplars are reference images that capture exactly what you want.
The agent will reference these when generating.
</purpose>

<arguments>
- image_path: Path to the image file (required)
- --notes/-n: Notes explaining why you like this image
</arguments>

<process>
1. Verify the image exists
2. Copy to .splatworld/exemplars/
3. Add to profile with timestamp and notes
4. Confirm addition
</process>

<execution>
```bash
splatworld exemplar "<path>" [--notes "<notes>"]
```

After adding:
1. Confirm the exemplar was added
2. Explain that the agent will reference this when generating
3. Suggest adding notes if not provided ("Notes help the agent understand WHY you like this")
</execution>

<examples>
User: "add this reference image ./reference/perfect-kitchen.jpg"
→ splatworld exemplar "./reference/perfect-kitchen.jpg"

User: "add ./moody-warehouse.png with notes about the lighting"
→ splatworld exemplar "./moody-warehouse.png" --notes "Love the dramatic shadows and warm highlights"
</examples>
