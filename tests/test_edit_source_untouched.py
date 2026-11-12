"""G4: running real edits through the shared pipeline never changes the real common checkpoint."""
import pytest
import torch

from src.models.checkpoint import read_registry, require_common_checkpoint
from src.training.evaluate import metrics_from_logits
from src.utils.config import load_config
from src.utils.hashing import sha256_file
from src.utils.config import resolve_path
from src.utils.reproducibility import sha256_tensor_dict
from src.wsp.candidate import CandidateBudget, evaluate_candidate
from src.wsp.edit_validator import ValidationContext
from src.wsp.weight_editor import ProtectedModelError, apply_edit, protect_model
from src.wsp.edit_schema import parse_proposal
from tests.test_edit_validator import bias, delta, prop, scale

CFG = load_config()
DSL = CFG.dsl


@pytest.mark.skipif("0" not in read_registry(CFG), reason="seed-0 common checkpoint is not registered")
def test_real_checkpoint_never_modified() -> None:
    entry = read_registry(CFG)["0"]
    file_before = sha256_file(resolve_path(entry["path"]))
    model, state_hash = require_common_checkpoint(CFG, 0, "cpu")
    protect_model(model)
    assert sha256_tensor_dict(model.state_dict()) == state_hash
    ctx = ValidationContext(DSL, tuple(DSL.allowed_layers), tuple(DSL.allowed_ops), 57044)
    x, y = torch.randn(2, 3, 224, 224), torch.tensor([0, 1])
    evaluate = lambda m: metrics_from_logits(m.eval()(x).detach(), y)  # noqa: E731
    budget = CandidateBudget(max_evaluations=3)
    for p in (prop(scale()), prop(bias(), delta()), prop(scale(f=(7.0,)))):
        evaluate_candidate(model, p, ctx, evaluate, budget)
    with pytest.raises(ProtectedModelError):
        apply_edit(model, parse_proposal(prop(bias()))[0], DSL, tuple(DSL.allowed_layers), tuple(DSL.allowed_ops))
    assert budget.evaluations_used == 2 and budget.validator_rejections == 1
    assert sha256_tensor_dict(model.state_dict()) == state_hash == entry["state_dict_sha256"]
    assert sha256_file(resolve_path(entry["path"])) == file_before == entry["file_sha256"]
