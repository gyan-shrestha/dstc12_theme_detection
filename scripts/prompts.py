from langchain_core.prompts import PromptTemplate

LABEL_CLUSTERS_PROMPT = PromptTemplate.from_template("""
You are a theme labeling assistant. Analyze these utterances and generate a concise theme.

- Use a short verb phrase (2–5 words).
- Avoid function words or vague phrases.
- Wrap the result in <theme_label>.

Utterances:
{utterances}

Output:
<theme_label>...</theme_label>
""")
