from jinja2 import Template

JUDGEMENT_PROMPT_TEMPLATE = Template("""\
Semantically compare the expected and generated reasons for a planning application decision.

For each expected reason, find the closest semantic match in the generated list.
Items do not need to match word-for-word but must cover the same
substantive requirement or reason.

Real reasons (ground truth):
{% for item in real %}
{{ loop.index }}. {{ item }}
{% endfor %}

Generated reasons (to evaluate):
{% for item in generated %}
{{ loop.index }}. {{ item }}
{% endfor %}

Return:
- matched: pairs of (real, generated) items with a similarity label:
    "perfect" = essentially identical wording and meaning
    "excellent" = same requirement with minor phrasing differences
    "very_good" = same core requirement, some detail variation
    "good" = captures main intent, moderate specificity differences
    "okay" = partial overlap, addresses same topic but misses aspects
    "bad" = weak semantic overlap, substantially different requirements
  Do not include unrelated items as matches; place them in missing/extra instead.
- missing_in_generation: real items with no adequate match in generated
- extra_in_generation: generated items with no adequate match in real
""")
