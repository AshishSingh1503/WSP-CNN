"""AlexNet construction: shapes, head, parameter count, deterministic head init."""
import torch

from src.models.alexnet import build_model, describe_model, feature_extractor

OLD_HEAD = 4096 * 1000 + 1000
NEW_HEAD = 4096 * 10 + 10
ALEXNET_1000 = 61_100_840


def test_output_shape_and_head() -> None:
    model = build_model(pretrained=False).eval()
    assert model(torch.zeros(2, 3, 224, 224)).shape == (2, 10)
    head = model.classifier[6]
    assert isinstance(head, torch.nn.Linear) and (head.in_features, head.out_features) == (4096, 10)


def test_param_count() -> None:
    total = sum(p.numel() for p in build_model(pretrained=False).parameters())
    assert total == ALEXNET_1000 - OLD_HEAD + NEW_HEAD == 57_044_810


def test_head_init_deterministic() -> None:
    a, b, c = (build_model(pretrained=False, seed=s).classifier[6] for s in (3, 3, 4))
    assert torch.equal(a.weight, b.weight) and torch.equal(a.bias, b.bias)
    assert not torch.equal(a.weight, c.weight)
    bound = 1 / 64
    assert a.weight.abs().max() <= bound and a.bias.abs().max() <= bound


def test_describe_model() -> None:
    table = describe_model(build_model(pretrained=False))
    assert len(table) == 16 and sum(r["numel"] for r in table) == 57_044_810
    assert {"name", "type", "shape", "numel", "trainable"} <= table[0].keys()
    assert table[-2]["name"] == "classifier.6.weight" and table[-2]["shape"] == [10, 4096]


def test_feature_extractor_matches_forward() -> None:
    model = build_model(pretrained=False).eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        feats = feature_extractor(model)(x)
        assert feats.shape == (2, 4096) and (feats >= 0).all()
        assert torch.allclose(model.classifier[6](feats), model(x), atol=1e-5)
