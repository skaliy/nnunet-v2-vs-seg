"""Model loading and one-case nnU-Net prediction for VS segmentation."""
from pathlib import Path

import torch


FIVE_FOLDS = (0, 1, 2, 3, 4)


def _load_predictor_class():
    """Import nnU-Net only when a prediction model is built."""
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    return nnUNetPredictor


def resolve_device(device):
    """Map 'auto'/'cpu'/'cuda' to a torch.device.

    'auto' -> cuda if available else cpu. 'cuda' without CUDA -> cpu (notice).
    """
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device not in ("cpu", "cuda"):
        raise ValueError(f"device must be auto|cpu|cuda, got {device!r}")
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device)


def normalize_folds(folds):
    """Normalize a fold selection to ``"auto"``, ``("all",)`` or integers.

    Accepted user-facing values are ``auto``, ``all`` and comma-separated fold
    numbers such as ``0,1,2,3,4``. Iterables of integers are accepted for Python
    callers.
    """
    if folds is None:
        return "auto"
    if isinstance(folds, str):
        value = folds.strip().lower()
        if value == "auto":
            return "auto"
        if value == "all":
            return ("all",)
        try:
            normalized = tuple(int(part.strip()) for part in value.split(","))
        except ValueError as exc:
            raise ValueError(
                "folds must be auto, all, or comma-separated integers; "
                f"got {folds!r}"
            ) from exc
    elif isinstance(folds, int):
        normalized = (folds,)
    else:
        normalized = tuple(folds)

    if normalized == ("all",):
        return normalized
    if not normalized or any(not isinstance(fold, int) or fold < 0
                             for fold in normalized):
        raise ValueError(
            "folds must contain one or more non-negative integers"
        )
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"folds contains duplicates: {normalized}")
    return normalized


def resolve_folds(model_dir, checkpoint_name, folds="auto"):
    """Resolve a requested fold selection against the model directory.

    Auto-detection supports the two deployment layouts used by this project:
    one ``fold_all`` checkpoint or a complete five-fold ``fold_0`` ...
    ``fold_4`` ensemble. Partial ensembles remain available when explicitly
    selected with ``folds``.
    """
    model_dir = Path(model_dir)
    requested = normalize_folds(folds)
    if requested != "auto":
        return requested

    has_all = (model_dir / "fold_all" / checkpoint_name).is_file()
    available = tuple(
        fold for fold in FIVE_FOLDS
        if (model_dir / f"fold_{fold}" / checkpoint_name).is_file()
    )
    has_five_folds = available == FIVE_FOLDS

    if has_all and has_five_folds:
        raise ValueError(
            "Model folder contains both fold_all and folds 0-4; select one "
            "explicitly with --folds"
        )
    if has_five_folds:
        return FIVE_FOLDS
    if has_all:
        return ("all",)
    if available:
        missing = sorted(set(FIVE_FOLDS) - set(available))
        raise FileNotFoundError(
            f"Auto-detected an incomplete five-fold model in {model_dir}: "
            f"available={list(available)}, missing={missing}. Select an "
            "intentional subset explicitly with --folds."
        )
    raise FileNotFoundError(
        f"No {checkpoint_name} found in fold_all or folds 0-4 under {model_dir}"
    )


def _validate_model_dir(model_dir, checkpoint_name, folds="auto"):
    model_dir = Path(model_dir)
    required = [model_dir / "plans.json", model_dir / "dataset.json"]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Model folder is missing required files:\n  " + "\n  ".join(missing)
        )
    resolved_folds = resolve_folds(model_dir, checkpoint_name, folds)
    fold_dirs = [
        "fold_all" if fold == "all" else f"fold_{fold}"
        for fold in resolved_folds
    ]
    missing = [
        str(model_dir / fold_dir / checkpoint_name)
        for fold_dir in fold_dirs
        if not (model_dir / fold_dir / checkpoint_name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Model folder is missing required checkpoint files:\n  "
            + "\n  ".join(missing)
        )
    return resolved_folds


def build_predictor(model_dir, device="auto",
                    checkpoint_name="checkpoint_final.pth", disable_tta=False,
                    folds="auto"):
    """Construct an nnUNetPredictor initialised from a trained model folder
    using either ``fold_all`` or an ensemble of folds. ``folds="auto"`` detects
    a complete five-fold model before falling back to ``fold_all``. Mirrors
    nnUNetv2_predict defaults (mirroring TTA on).
    """
    resolved_folds = _validate_model_dir(model_dir, checkpoint_name, folds)
    predictor_class = _load_predictor_class()
    predictor = predictor_class(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=not disable_tta,
        device=resolve_device(device),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    predictor.initialize_from_trained_model_folder(
        str(model_dir), use_folds=resolved_folds, checkpoint_name=checkpoint_name
    )
    return predictor


def predict(predictor, in_nifti_dir, out_dir):
    """Run one-case prediction, saving segmentation and probabilities.

    The PACS service always submits one case. Use nnU-Net's synchronous
    single-array API so the full preprocessed volume is not transferred
    through a multiprocessing queue. Docker's default 64 MB shared-memory
    mount is too small for that transfer and kills predict_from_files workers.

    Returns (seg_nifti_path, prob_npz_path).
    """
    in_nifti_dir = Path(in_nifti_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs = sorted(in_nifti_dir.glob("*_0000.nii.gz"))
    if len(inputs) != 1:
        raise RuntimeError(
            f"Expected exactly one *_0000.nii.gz in {in_nifti_dir}; "
            f"got {[p.name for p in inputs]}"
        )
    input_file = inputs[0]
    case_id = input_file.name[:-len("_0000.nii.gz")]
    output_truncated = out_dir / case_id

    reader_writer = predictor.plans_manager.image_reader_writer_class()
    image, properties = reader_writer.read_images([str(input_file)])
    predictor.predict_single_npy_array(
        input_image=image,
        image_properties=properties,
        segmentation_previous_stage=None,
        output_file_truncated=str(output_truncated),
        save_or_return_probabilities=True,
    )

    seg = sorted(p for p in out_dir.glob("*.nii.gz"))
    npz = sorted(out_dir.glob("*.npz"))
    if len(seg) != 1 or len(npz) != 1:
        raise RuntimeError(
            f"Expected exactly one seg .nii.gz and one .npz in {out_dir}; "
            f"got seg={[p.name for p in seg]} npz={[p.name for p in npz]}"
        )
    return seg[0], npz[0]
