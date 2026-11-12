"""docs/EDIT_DSL.md is in sync with the schema and config, and its examples are valid on AlexNet."""
import json

from src.utils.config import REPO_ROOT, load_config
from src.wsp.dsl_doc import EXAMPLES, render_dsl_doc
from src.wsp.edit_schema import Proposal
from src.wsp.edit_validator import ValidationContext, validate_proposal

CFG = load_config()


def test_doc_file_matches_generator() -> None:
    assert (REPO_ROOT / "docs" / "EDIT_DSL.md").read_text(encoding="utf-8") == render_dsl_doc(CFG), \
        "run: python scripts/export_edit_dsl.py"


def test_embedded_schema_is_current() -> None:
    text = (REPO_ROOT / "docs" / "EDIT_DSL.md").read_text(encoding="utf-8")
    assert json.dumps(Proposal.model_json_schema(), indent=1) in text


def test_examples_are_valid_on_alexnet(model) -> None:
    ctx = ValidationContext(CFG.dsl, tuple(CFG.dsl.allowed_layers), tuple(CFG.dsl.allowed_ops), 57044)
    assert len(EXAMPLES) >= 3
    for title, example in EXAMPLES:
        result = validate_proposal(example, model, ctx)
        assert result.ok, (title, result.errors)
