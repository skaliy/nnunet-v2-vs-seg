# DICOM inference

Runs a trained nnU-Net v2 vestibular-schwannoma model on one DICOM T1 series
and writes two derived DICOM series: a binary mask and a uint16 foreground
probability map.

## Run

From the repository root:

```bash
python -m nnunet_inference /path/to/dicom [output_dir] \
    --model-dir /path/to/trained-model \
    --device auto --checkpoint final --folds auto
```

`output_dir` defaults to the input series parent. Outputs are written to
`mask/` and `vote_map/`. `--folds auto` accepts either `fold_all` or a complete
`fold_0`–`fold_4` ensemble; select folds explicitly when both layouts exist.
Mirror TTA is enabled by default and can be disabled with `--no-tta`.

The pipeline reorients inputs to the LPS storage orientation used during
training, predicts synchronously with nnU-Net, maps outputs back to the native
DICOM grid, and assigns deterministic derived UIDs tied to the model and run
settings.

## Layout

- `pipeline.py`: command-line interface and DICOM-to-model orchestration
- `predictor.py`: device, fold, model loading, and one-case nnU-Net prediction
- `dicom_io.py`: orientation, resampling, DICOM writing, and UID provenance
- `tests/`: unit tests and optional real-DICOM validation
- `integration/`: fixed Research PACS container build and qualification

## Verify

```bash
python -m unittest discover -s nnunet_inference/tests -p 'test_*.py' -v

NNUNET_DICOM_FIXTURE=/path/to/dicom \
NNUNET_MODEL_DIR=/path/to/trained-model \
    python -m nnunet_inference.tests.validate_dicom
```

Medical images and models are intentionally excluded from Git. See
[`integration/README.md`](integration/README.md) for the PACS container.
