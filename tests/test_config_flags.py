import pytest

from config import load_settings


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, True),
        ("", True),
        ("1", True),
        ("0", False),
        ("false", False),
        ("no", False),
    ],
)
def test_refinement_flags_default_on(monkeypatch, value, expected):
    for name in ("REFINE_TABLES", "CAPTION_FIGURES"):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    s = load_settings(require_key=False)
    assert s.refine_tables is expected
    assert s.caption_figures is expected


def test_second_verifier_model_is_configurable(monkeypatch) -> None:
    monkeypatch.delenv("MODEL_VERIFIER2", raising=False)
    default = load_settings(require_key=False).verifier2_model
    monkeypatch.setenv("MODEL_VERIFIER2", "anthropic/claude-sonnet-5.5")
    s = load_settings(require_key=False)
    assert s.verifier2_model == "anthropic/claude-sonnet-5.5" != default
    assert s.all_cheap().verifier2_model == s.cheap_model
