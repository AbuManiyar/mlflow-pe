# Signature verification experiments

This mini-project makes Siamese signature-verification into
repeatable, comparable MLflow runs. Each run uses a person-disjoint train,
validation, and test split, chooses a classification threshold using validation
F1, evaluates the holdout test split, and records its parameters, metrics,
checkpoint, pair CSVs, person split CSVs, and plots.

## 1. Install

Use Python 3.10 or newer in a virtual environment:

```powershell
py -m venv venv
venv\Scripts\Activate
python -m pip install -e ".[test]"
```

## 2. Prepare the images

The image root must contain one directory per signer and a corresponding
`<signer>_forg` directory, for example:

```text
data/signatures/extract/
  001/
  001_forg/
  002/
  002_forg/
```

To download the dataset used in the notebook, configure Kaggle access for
KaggleHub, then run with `--download`:

```powershell
signet-experiment --download
```

Or point at an already-downloaded image root:

```powershell
signet-experiment --data-dir "E:\datasets\signatures\extract"
```

## 3. Run experiments

Every invocation creates a separate run under the local
`Signature Verification Experiments` MLflow experiment. Change parameters
between invocations to compare configurations:

```powershell
signet-experiment --run-name adam-lr-3e-4 --learning-rate 0.0003 --seed 42
signet-experiment --run-name adam-lr-1e-4 --learning-rate 0.0001 --seed 42
signet-experiment --run-name larger-batch --batch-size 32 --seed 42
```

The script defaults to local tracking at `./mlruns`, so no DagsHub account or
remote credentials are required. Training uses the available CUDA device when
present; `--num-workers 0` is the safe default across operating systems.

## 4. Compare and select a run

After at least one run has completed, open the MLflow UI in another terminal:

```powershell
mlflow ui --backend-store-uri ./mlruns
```

Open the URL printed by MLflow (normally `http://127.0.0.1:5000`) and compare
runs in the experiment table. Use validation metrics such as
`validation_f1_forged`, `validation_far`, and `validation_frr` to select a model.
The threshold is selected by maximum validation F1. Holdout `test_*` metrics
are logged for final evaluation, not for tuning.

Each run includes `best_model.pth`, which contains the model state, architecture
configuration, and selected threshold. The same run logs training curves,
validation FAR/FRR, a test confusion matrix, and the exact person/pair CSVs.

## Useful options

```text
--data-dir, --download, --tracking-uri, --experiment-name, --run-name
--seed, --validation-fraction, --test-fraction
--image-height, --image-width, --embedding-dim, --dropout
--batch-size, --learning-rate, --weight-decay, --max-epochs, --patience
--num-workers
```

The notebook's label convention is retained: label `0` is a genuine-genuine
pair and label `1` is a genuine-forged pair. Accordingly, FAR is forged pairs
accepted as genuine (`FN / forged`), and FRR is genuine pairs rejected as forged
(`FP / genuine`).

## Tests

```powershell
python -m pytest
```
