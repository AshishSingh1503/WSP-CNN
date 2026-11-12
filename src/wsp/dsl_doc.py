"""Render docs/EDIT_DSL.md (inserted into the LLM prompt later) from the schema and config."""
from __future__ import annotations

import json
from typing import Any

from src.utils.config import Config
from src.wsp.edit_schema import Proposal

BASE = {"reason": "Short justification.", "evidence": ["metrics.per_class_accuracy"],
        "expected_effect": "What should improve.", "confidence": 0.5, "risk": "low"}

EXAMPLES: list[tuple[str, dict[str, Any]]] = [
    ("Scale two channels of features.10", {**BASE, "proposal_id": "ex1", "operations": [
        {"operation": "scale_output_channel", "target_layer": "features.10",
         "channel_indices": [12, 77], "scale_factors": [1.2, 0.8]}]}),
    ("Bundle a bias shift and a sparse weight delta in the head", {**BASE, "proposal_id": "ex2", "operations": [
        {"operation": "bias_shift", "target_layer": "classifier.6", "indices": [3], "shifts": [0.5]},
        {"operation": "add_delta", "target_param": "classifier.6.weight",
         "indices": [[3, 100], [3, 101]], "deltas": [0.01, -0.01]}]}),
    ("Rank-1 update on a block of the head weight", {**BASE, "proposal_id": "ex3", "operations": [
        {"operation": "low_rank_update", "target_param": "classifier.6.weight", "rows": [3],
         "cols": [0, 1, 2, 3], "u": [[0.1]], "v": [[0.05, 0.05, 0.05, 0.05]]}]}),
    ("Propose nothing", {**BASE, "proposal_id": "ex4", "operations": [
        {"operation": "no_edit", "reason": "No change is clearly beneficial."}]}),
]


def render_dsl_doc(cfg: Config) -> str:
    """Markdown describing the edit DSL: rules (from config), JSON schema and valid examples."""
    d = cfg.dsl
    rules = f"""# Weight Edit DSL

Reply with ONE JSON object (a *proposal*) and nothing else. Extra fields, NaN/inf and numeric strings are rejected.

## Rules
- A proposal has 1..{d.max_ops_per_proposal} operations; `no_edit` must be the only operation if used.
- Allowed layers: {", ".join(d.allowed_layers)}. Allowed operations: {", ".join(d.allowed_ops)}.
- `reason` <= {d.limits.max_reason_chars} chars; `expected_effect` and `risk` <= {d.limits.max_text_chars} chars;
  `evidence` <= {d.limits.max_evidence_items} packet field paths; `confidence` in [0, 1];
  `proposal_id` matches [A-Za-z0-9_.-], <= {d.limits.max_id_chars} chars.
- The edit budget counts scalar parameters whose value changes (a channel scale changes fan_in*kh*kw weights + 1 bias).
- Index lists must not contain duplicates; all indices must be in range.

| operation | fields | bounds |
|---|---|---|
| add_delta | target_param, indices (full index tuples), deltas | <= {d.add_delta.max_indices} indices; \\|delta\\| <= {d.add_delta.max_abs_delta} and <= {d.add_delta.rel_value_factor}*\\|current value\\| + {d.add_delta.rel_value_offset} |
| scale_output_channel | target_layer, channel_indices, scale_factors | <= {d.scale_output_channel.max_channels} channels; factors in [{d.scale_output_channel.scale_min}, {d.scale_output_channel.scale_max}]{" or exactly 0.0" if d.scale_output_channel.allow_zero_channel else ""}; scales weights and bias of each channel |
| bias_shift | target_layer, indices, shifts | <= {d.bias_shift.max_indices} indices; \\|shift\\| <= {d.bias_shift.max_bias_shift_classifier} for {d.bias_shift.classifier_layer}, else <= {d.bias_shift.max_bias_shift_other} |
| low_rank_update | target_param (Linear weight), rows, cols, u (rank x len(rows)), v (rank x len(cols)) | rank <= {d.low_rank_update.max_rank}; <= {d.low_rank_update.max_rows} rows, <= {d.low_rank_update.max_cols} cols; \\|\\|update\\|\\|_F <= {d.low_rank_update.max_rel_norm}*\\|\\|W\\|\\|_F; update = sum_i u_i (outer) v_i on the rows x cols block |
| no_edit | reason | none |

## JSON schema
```json
{json.dumps(Proposal.model_json_schema(), indent=1)}
```

## Valid examples
"""
    for title, example in EXAMPLES:
        rules += f"\n**{title}**\n```json\n{json.dumps(example, indent=1)}\n```\n"
    return rules
