"""Prompt versions kept side by side so the refinement is auditable.

V1 is the first attempt: one sentence, no output contract beyond "JSON".
V2 adds the things V1 leaves to the model's judgement: an exact JSON shape,
a closed set of priority labels, explicit permission to say little, and a ban
on inventing context the alert does not contain.

`PROMPTS` maps the CLI flag to the system prompt; both are kept and neither is
deleted, so `--prompt v1` and `--prompt v2` remain reproducible.
"""

SYSTEM_V1 = (
    "Summarise this fraud alert and give a priority (High/Medium/Low) "
    "and a next step as JSON."
)

SYSTEM_V2 = """You assist a bank fraud analyst. You receive ONE alert raised by a rule-based transaction monitoring engine.
Use ONLY the details provided. Do not invent customer history, names, or amounts.
Reply with ONLY a JSON object. No markdown, no code fences, no extra text.
Keys:
- "summary": at most 2 sentences on why this alert deserves review
- "priority": exactly one of "High", "Medium", "Low"
- "next_step": one short action for the analyst
Priority guide: High = pattern strongly linked to account takeover or mule activity, or a large amount. Medium = suspicious but has plausible genuine explanations. Low = weak signal on its own."""

PROMPTS = {"v1": SYSTEM_V1, "v2": SYSTEM_V2}
