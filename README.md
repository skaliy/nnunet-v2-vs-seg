# nnU-Net v2 vestibular-schwannoma workflow

Utilities for preparing a five-fold nnU-Net v2 dataset, training and evaluating
the folds, and running NIfTI or DICOM inference.

Intended for research use only. This software is not a medical device and has not been
validated for clinical diagnosis or treatment.

## Setup

Create an environment with Python 3.11 and install a PyTorch build appropriate
for the available CPU/CUDA hardware. Then install the project dependencies:

```bash
python -m pip install \
    nnunetv2==2.6.2 imagedata==3.8.14 SimpleITK==2.5.2 \
    pydicom==2.4.4 pandas numpy scipy nibabel
```

Cross-validation evaluation uses fastMONAI's VS metrics. Install the tested
revision next to this repository:

```bash
git clone https://github.com/MMIV-ML/fastMONAI.git ../fastMONAI
git -C ../fastMONAI checkout 77588648b3d30ef7a0b09458dd6e9cdc19843723
python -m pip install -e ../fastMONAI
```

```bash
export NNUNET_WORKSPACE="$PWD/nnUNet_data"
export nnUNet_raw="$NNUNET_WORKSPACE/nnUNet_raw"
export nnUNet_preprocessed="$NNUNET_WORKSPACE/nnUNet_preprocessed"
export nnUNet_results="$NNUNET_WORKSPACE/nnUNet_results"
```

## Prepare a dataset

Provide a CSV with one row per case:

```csv
case_id,t1_img_path,t1_seg_path,fold
case001,images/case001.nii.gz,masks/case001.nii.gz,1
case002,images/case002.nii.gz,masks/case002.nii.gz,2
```

Case IDs must be unique, every image/mask pair must have matching geometry, and
`fold` must assign each case to one of five validation folds. Paths may be
absolute or relative to `--source-root`.

```bash
DATASET_ID=1
DATASET=Dataset001_VSSegmentation

python training/prepare_dataset.py \
    --data-csv /path/to/ml_dataset.csv \
    --source-root /path/to/dataset \
    --dataset-id "$DATASET_ID" \
    --dataset-name VSSegmentation \
    --label-mode largest-component

nnUNetv2_plan_and_preprocess \
    -d "$DATASET_ID" -pl nnUNetPlannerResEncL -c 3d_fullres \
    --verify_dataset_integrity -np 8 -npfp 8

cp "$nnUNet_raw/$DATASET/splits_final.json" \
   "$nnUNet_preprocessed/$DATASET/splits_final.json"
```

`largest-component` removes disconnected label blobs; use `binary` to retain
all foreground components. The preparation script refuses to overwrite an
existing dataset.

## Train and evaluate five folds

```bash
for fold in 0 1 2 3 4; do
    nnUNetv2_train "$DATASET_ID" 3d_fullres "$fold" \
        -p nnUNetResEncUNetLPlans --npz
done

python training/evaluate_cv.py --dataset "$DATASET"
```

Evaluation writes per-case and summary CSV files to `cv_metrics/` in the
trained model directory.

`fold=all` trains one model on all cases; it is not cross-validation. Use
`--c` to continue an interrupted fold.

## Inference

```bash
nnUNetv2_predict -i /path/to/images -o /path/to/predictions \
    -d "$DATASET_ID" -c 3d_fullres -p nnUNetResEncUNetLPlans \
    -f 0 1 2 3 4
```

NIfTI inputs require the `_0000.nii.gz` channel suffix. For DICOM inference
and Research PACS deployment, see [`nnunet_inference/README.md`](nnunet_inference/README.md).

Datasets, trained weights, and qualified container archives are not included.
The source code is available under the [MIT License](LICENSE).
