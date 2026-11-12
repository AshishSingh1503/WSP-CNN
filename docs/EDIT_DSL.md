# Weight Edit DSL

Reply with ONE JSON object (a *proposal*) and nothing else. Extra fields, NaN/inf and numeric strings are rejected.

## Rules
- A proposal has 1..3 operations; `no_edit` must be the only operation if used.
- Allowed layers: features.0, features.3, features.6, features.8, features.10, classifier.1, classifier.4, classifier.6. Allowed operations: add_delta, scale_output_channel, bias_shift, low_rank_update, no_edit.
- `reason` <= 800 chars; `expected_effect` and `risk` <= 400 chars;
  `evidence` <= 20 packet field paths; `confidence` in [0, 1];
  `proposal_id` matches [A-Za-z0-9_.-], <= 64 chars.
- The edit budget counts scalar parameters whose value changes (a channel scale changes fan_in*kh*kw weights + 1 bias).
- Index lists must not contain duplicates; all indices must be in range.

| operation | fields | bounds |
|---|---|---|
| add_delta | target_param, indices (full index tuples), deltas | <= 512 indices; \|delta\| <= 0.05 and <= 0.5*\|current value\| + 0.01 |
| scale_output_channel | target_layer, channel_indices, scale_factors | <= 8 channels; factors in [0.5, 1.5]; scales weights and bias of each channel |
| bias_shift | target_layer, indices, shifts | <= 16 indices; \|shift\| <= 1.0 for classifier.6, else <= 0.2 |
| low_rank_update | target_param (Linear weight), rows, cols, u (rank x len(rows)), v (rank x len(cols)) | rank <= 4; <= 32 rows, <= 64 cols; \|\|update\|\|_F <= 0.02*\|\|W\|\|_F; update = sum_i u_i (outer) v_i on the rows x cols block |
| no_edit | reason | none |

## JSON schema
```json
{
 "$defs": {
  "AddDelta": {
   "additionalProperties": false,
   "description": "Add ``deltas[i]`` to element ``indices[i]`` (a full index tuple) of ``target_param``.",
   "properties": {
    "operation": {
     "const": "add_delta",
     "title": "Operation",
     "type": "string"
    },
    "target_param": {
     "title": "Target Param",
     "type": "string"
    },
    "indices": {
     "items": {
      "items": {
       "type": "integer"
      },
      "type": "array"
     },
     "title": "Indices",
     "type": "array"
    },
    "deltas": {
     "items": {
      "type": "number"
     },
     "title": "Deltas",
     "type": "array"
    }
   },
   "required": [
    "operation",
    "target_param",
    "indices",
    "deltas"
   ],
   "title": "AddDelta",
   "type": "object"
  },
  "BiasShift": {
   "additionalProperties": false,
   "description": "Add ``shifts[i]`` to bias element ``indices[i]`` of ``target_layer``.",
   "properties": {
    "operation": {
     "const": "bias_shift",
     "title": "Operation",
     "type": "string"
    },
    "target_layer": {
     "title": "Target Layer",
     "type": "string"
    },
    "indices": {
     "items": {
      "type": "integer"
     },
     "title": "Indices",
     "type": "array"
    },
    "shifts": {
     "items": {
      "type": "number"
     },
     "title": "Shifts",
     "type": "array"
    }
   },
   "required": [
    "operation",
    "target_layer",
    "indices",
    "shifts"
   ],
   "title": "BiasShift",
   "type": "object"
  },
  "LowRankUpdate": {
   "additionalProperties": false,
   "description": "Add sum_i u_i (outer) v_i on the ``rows`` x ``cols`` block of a Linear weight (A, B zero-padded).",
   "properties": {
    "operation": {
     "const": "low_rank_update",
     "title": "Operation",
     "type": "string"
    },
    "target_param": {
     "title": "Target Param",
     "type": "string"
    },
    "rows": {
     "items": {
      "type": "integer"
     },
     "title": "Rows",
     "type": "array"
    },
    "cols": {
     "items": {
      "type": "integer"
     },
     "title": "Cols",
     "type": "array"
    },
    "u": {
     "items": {
      "items": {
       "type": "number"
      },
      "type": "array"
     },
     "title": "U",
     "type": "array"
    },
    "v": {
     "items": {
      "items": {
       "type": "number"
      },
      "type": "array"
     },
     "title": "V",
     "type": "array"
    }
   },
   "required": [
    "operation",
    "target_param",
    "rows",
    "cols",
    "u",
    "v"
   ],
   "title": "LowRankUpdate",
   "type": "object"
  },
  "NoEdit": {
   "additionalProperties": false,
   "description": "Explicitly propose no change.",
   "properties": {
    "operation": {
     "const": "no_edit",
     "title": "Operation",
     "type": "string"
    },
    "reason": {
     "default": "",
     "title": "Reason",
     "type": "string"
    }
   },
   "required": [
    "operation"
   ],
   "title": "NoEdit",
   "type": "object"
  },
  "ScaleOutputChannel": {
   "additionalProperties": false,
   "description": "Multiply each listed output channel's weights AND bias by its scale factor.",
   "properties": {
    "operation": {
     "const": "scale_output_channel",
     "title": "Operation",
     "type": "string"
    },
    "target_layer": {
     "title": "Target Layer",
     "type": "string"
    },
    "channel_indices": {
     "items": {
      "type": "integer"
     },
     "title": "Channel Indices",
     "type": "array"
    },
    "scale_factors": {
     "items": {
      "type": "number"
     },
     "title": "Scale Factors",
     "type": "array"
    }
   },
   "required": [
    "operation",
    "target_layer",
    "channel_indices",
    "scale_factors"
   ],
   "title": "ScaleOutputChannel",
   "type": "object"
  }
 },
 "additionalProperties": false,
 "description": "One LLM proposal: up to a few operations plus justification metadata.",
 "properties": {
  "proposal_id": {
   "title": "Proposal Id",
   "type": "string"
  },
  "reason": {
   "title": "Reason",
   "type": "string"
  },
  "evidence": {
   "items": {
    "type": "string"
   },
   "title": "Evidence",
   "type": "array"
  },
  "expected_effect": {
   "title": "Expected Effect",
   "type": "string"
  },
  "confidence": {
   "maximum": 1.0,
   "minimum": 0.0,
   "title": "Confidence",
   "type": "number"
  },
  "risk": {
   "title": "Risk",
   "type": "string"
  },
  "operations": {
   "items": {
    "discriminator": {
     "mapping": {
      "add_delta": "#/$defs/AddDelta",
      "bias_shift": "#/$defs/BiasShift",
      "low_rank_update": "#/$defs/LowRankUpdate",
      "no_edit": "#/$defs/NoEdit",
      "scale_output_channel": "#/$defs/ScaleOutputChannel"
     },
     "propertyName": "operation"
    },
    "oneOf": [
     {
      "$ref": "#/$defs/AddDelta"
     },
     {
      "$ref": "#/$defs/ScaleOutputChannel"
     },
     {
      "$ref": "#/$defs/BiasShift"
     },
     {
      "$ref": "#/$defs/LowRankUpdate"
     },
     {
      "$ref": "#/$defs/NoEdit"
     }
    ]
   },
   "title": "Operations",
   "type": "array"
  }
 },
 "required": [
  "proposal_id",
  "reason",
  "evidence",
  "expected_effect",
  "confidence",
  "risk",
  "operations"
 ],
 "title": "Proposal",
 "type": "object"
}
```

## Valid examples

**Scale two channels of features.10**
```json
{
 "reason": "Short justification.",
 "evidence": [
  "metrics.per_class_accuracy"
 ],
 "expected_effect": "What should improve.",
 "confidence": 0.5,
 "risk": "low",
 "proposal_id": "ex1",
 "operations": [
  {
   "operation": "scale_output_channel",
   "target_layer": "features.10",
   "channel_indices": [
    12,
    77
   ],
   "scale_factors": [
    1.2,
    0.8
   ]
  }
 ]
}
```

**Bundle a bias shift and a sparse weight delta in the head**
```json
{
 "reason": "Short justification.",
 "evidence": [
  "metrics.per_class_accuracy"
 ],
 "expected_effect": "What should improve.",
 "confidence": 0.5,
 "risk": "low",
 "proposal_id": "ex2",
 "operations": [
  {
   "operation": "bias_shift",
   "target_layer": "classifier.6",
   "indices": [
    3
   ],
   "shifts": [
    0.5
   ]
  },
  {
   "operation": "add_delta",
   "target_param": "classifier.6.weight",
   "indices": [
    [
     3,
     100
    ],
    [
     3,
     101
    ]
   ],
   "deltas": [
    0.01,
    -0.01
   ]
  }
 ]
}
```

**Rank-1 update on a block of the head weight**
```json
{
 "reason": "Short justification.",
 "evidence": [
  "metrics.per_class_accuracy"
 ],
 "expected_effect": "What should improve.",
 "confidence": 0.5,
 "risk": "low",
 "proposal_id": "ex3",
 "operations": [
  {
   "operation": "low_rank_update",
   "target_param": "classifier.6.weight",
   "rows": [
    3
   ],
   "cols": [
    0,
    1,
    2,
    3
   ],
   "u": [
    [
     0.1
    ]
   ],
   "v": [
    [
     0.05,
     0.05,
     0.05,
     0.05
    ]
   ]
  }
 ]
}
```

**Propose nothing**
```json
{
 "reason": "Short justification.",
 "evidence": [
  "metrics.per_class_accuracy"
 ],
 "expected_effect": "What should improve.",
 "confidence": 0.5,
 "risk": "low",
 "proposal_id": "ex4",
 "operations": [
  {
   "operation": "no_edit",
   "reason": "No change is clearly beneficial."
  }
 ]
}
```
