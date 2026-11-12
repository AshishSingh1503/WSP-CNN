# WSP-CNN

Pre-registered study of whether an LLM can propose direct, structured weight edits to an
ImageNet-pretrained AlexNet adapted to CIFAR-10. Read `PROTOCOL.md` first; negative results are valid.

## Setup (Windows PowerShell)

```powershell
cd C:\WSP-CNN
py -3.12 -m venv venv                  # Python 3.12 (3.11 also works)
.\venv\Scripts\Activate.ps1            # if blocked: Set-ExecutionPolicy -Scope Process Bypass
python -m pip install --upgrade pip
pip install -r requirements.txt        # PyTorch CUDA 12.4 wheels, ~2.5 GB download, ~6 GB on disk
python scripts\record_env.py           # writes results\env.json
python -m pytest -q                    # run all tests
python main.py --experiment make_splits
python main.py --experiment check_data
```

Notes: `data/`, `checkpoints/`, raw `results/` and `.env` are git-ignored. DataLoader
`num_workers` defaults to 2 in `configs/config.yaml`; set it to 0 if workers hang on Windows.
Put API keys in `.env` (never commit it).

## Layout

`configs/` YAML config, `data/` CIFAR-10 and split indices, `checkpoints/`, `src/{data,models,diagnostics,wsp,training,experiments,utils}/`,
`prompts/`, `results/`, `tests/`, `scripts/`, `main.py`.

## Test-set isolation

The CIFAR-10 test set is only reachable through `get_test_loader(token, cfg)`, which raises
`PermissionError` unless `results/FROZEN_PROTOCOL.lock` exists and `token` is its SHA-256.
`tests/test_test_isolation.py` statically scans `src/` to ensure no other module references it.
