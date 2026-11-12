"""Diagnostics are deterministic across repeated runs (CPU, synthetic AlexNet)."""
from src.diagnostics.packet import build_hybrid_packet, build_pure_packet, to_prompt_json
from tests.test_packet_isolation import _synthetic


def test_pure_and_hybrid_repeatable() -> None:
    model, kwargs = _synthetic()
    pure = [build_pure_packet(model, **kwargs).model_dump_json() for _ in range(2)]
    hybrid = [build_hybrid_packet(model, **kwargs).model_dump_json() for _ in range(2)]
    assert pure[0] == pure[1] and hybrid[0] == hybrid[1]
    assert to_prompt_json(build_pure_packet(model, **kwargs)) == to_prompt_json(build_pure_packet(model, **kwargs))


def test_pure_packet_is_prefix_free_of_hybrid_gradient_text() -> None:
    model, kwargs = _synthetic()
    assert "classifier6" not in build_pure_packet(model, **kwargs).model_dump_json()
    assert "classifier6" in build_hybrid_packet(model, **kwargs).model_dump_json()
