# Contributing

Thank you for improving this research repository. Contributions should preserve the distinction between paper reproduction and later extensions.

## Good contribution areas

- Reproducibility and environment improvements.
- Tests and bug fixes.
- Documentation and teaching examples.
- Dataset validation and adapters.
- Evaluation, calibration, and error analysis.
- Efficiency benchmarks.
- Additional models or datasets clearly labelled as extensions.

## Development setup

```bash
git clone https://github.com/sauravsingla/Transformer-Based-Pre-Trained-NLP-Models.git
cd Transformer-Based-Pre-Trained-NLP-Models
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
pip install -r requirements-dev.txt
```

Windows PowerShell activation:

```powershell
.venv\Scripts\Activate.ps1
```

## Before submitting a change

Run:

```bash
ruff check src tests
pytest -q
python -m pip check
```

A pull request should:

- Explain the problem and proposed solution.
- Keep unrelated changes separate.
- Add or update tests for behavioural changes.
- Update documentation when interfaces or outputs change.
- Avoid committing datasets, model checkpoints, secrets, or large generated artifacts.
- State whether the change is paper-aligned or an extension.

## Reproduction claims

Do not add a score to the reproduced-results table without retaining the required artifacts described in `docs/REPRODUCIBILITY.md`.

Published values must be labelled **published**. New values generated with unchanged paper-aligned settings may be labelled **reproduced**. Values generated after changing the model, data, seed protocol, preprocessing, or training procedure must be labelled **extended**.

## Code style

- Use type hints for public functions where practical.
- Prefer small, testable functions.
- Raise clear errors for invalid data and configuration.
- Keep file and network side effects explicit.
- Do not silence warnings globally without a documented reason.
- Keep user-facing outputs stable or document changes.

## Commit style

Use focused, human-readable commits such as:

```text
fix: preserve custom head settings when reloading checkpoints
test: cover invalid sentiment labels
docs: clarify published and reproduced metrics
```

## Pull-request review checklist

- [ ] Tests pass locally.
- [ ] Lint passes locally.
- [ ] New behaviour is tested.
- [ ] Documentation is current.
- [ ] No private or licensed dataset content is included.
- [ ] No result is overstated.
- [ ] Backward compatibility is considered.

## Reporting issues

A useful bug report contains the command, configuration, stack trace, Python and package versions, operating system, hardware, and a minimal reproducible example. Do not post private data or credentials.
