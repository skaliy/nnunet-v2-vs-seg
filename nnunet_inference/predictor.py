"""Model loading and one-case nnU-Net prediction for VS segmentation."""
import copy
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt


FIVE_FOLDS = (0, 1, 2, 3, 4)

# The mask is restored to the input grid by interpolating each label's signed
# distance to its edge with this spline order, then taking the label a voxel
# lies deepest inside. nnU-Net's own export copies the nearest 1.5 mm model
# slice instead (order_z=0), which repeats every third slice of 1.0 mm input.
# On the 18 held-out 1.0 mm cases cubic beat linear (Dice 0.886 vs 0.883) and
# shrank volumes less (-5.0% vs -7.3% against manual).
MASK_INTERPOLATION_ORDER = 3

# The probabilities behind the vote map are interpolated linearly along the
# slice axis as well, instead of copying the nearest model slice (order_z=0).
PROBABILITY_ORDER_Z = 1

# TTA averages the original and one copy flipped along both of these axes at
# once: nnU-Net axis 0 (inferior-superior, the slice axis of the LPS inputs)
# and axis 1 (anterior-posterior). On the 344 held-out cases this had as few or
# fewer false positives than the other 2-flip sets and nnU-Net's eight-flip
# mirroring, at a quarter of the eight-flip cost.
TTA_FLIP_AXES = (0, 1)


class FlipPairMixin:
    """Replace nnU-Net's mirroring, which averages every combination of the
    allowed axes, with the original plus one flip over all of them together."""
    def _internal_maybe_mirror_and_predict(self, x):
        prediction = self.network(x)
        if self.use_mirroring and self.allowed_mirroring_axes:
            axes = [axis + 2 for axis in self.allowed_mirroring_axes]
            prediction += torch.flip(self.network(torch.flip(x, axes)), axes)
            prediction /= 2
        return prediction


def _load_predictor_class():
    """Import nnU-Net only when a prediction model is built."""
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    return type("FlipPairNNUNetPredictor", (FlipPairMixin, nnUNetPredictor), {})


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
                    folds="auto", parallel=False):
    """Construct an nnUNetPredictor initialised from a trained model folder
    using either ``fold_all`` or an ensemble of folds. ``folds="auto"`` detects
    a complete five-fold model before falling back to ``fold_all``. TTA is on
    by default and adds one copy flipped along ``TTA_FLIP_AXES`` (two forward
    passes per fold) instead of every combination of the trained mirror axes.
    """
    resolved_folds = _validate_model_dir(model_dir, checkpoint_name, folds)
    predictor_class = _load_predictor_class()
    if parallel:
        from .parallel import ParallelFoldMixin, assign_worker_cpus
        if resolve_device(device).type != "cpu":
            raise ValueError("Parallel fold inference requires CPU")
        assign_worker_cpus(len(resolved_folds))
        predictor_class = type("ParallelNNUNetPredictor",
                               (ParallelFoldMixin, predictor_class), {})
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
        str(model_dir), use_folds=resolved_folds[:1] if parallel else resolved_folds,
        checkpoint_name=checkpoint_name
    )
    # FlipPairMixin flips all allowed axes together in one copy. Without the
    # trained mirroring for every TTA axis, TTA is off.
    trained_axes = predictor.allowed_mirroring_axes or ()
    predictor.allowed_mirroring_axes = (
        TTA_FLIP_AXES if set(TTA_FLIP_AXES) <= set(trained_axes) else None)
    if parallel:
        predictor.parallel_folds = resolved_folds
        predictor.parallel_model_dir = str(model_dir)
        predictor.parallel_checkpoint_name = checkpoint_name
        # The coordinator needs the plans and label manager; only workers use
        # networks. Release its initial fold before launching five workers.
        predictor.network = None
        predictor.list_of_parameters = []
    return predictor


def _signed_depth(mask, spacing, margin=8):
    """Signed distance to the mask edge in mm: positive inside, negative outside.

    Computed on the mask's bounding box plus margin voxels, which is wide
    enough for cubic interpolation to match a full-volume distance map. Voxels
    outside the box get a value below every value inside it, so they stay
    outside.
    """
    outside = -4.0 * float(max(spacing))
    if not mask.any():
        return np.full(mask.shape, outside, dtype=np.float32)
    if mask.all():
        return np.full(mask.shape, -outside, dtype=np.float32)
    coords = np.argwhere(mask)
    lo = np.maximum(coords.min(0) - margin, 0)
    hi = np.minimum(coords.max(0) + margin + 1, mask.shape)
    box = tuple(slice(a, b) for a, b in zip(lo, hi))
    inner = mask[box]
    inner_depth = (distance_transform_edt(inner, sampling=spacing)
                   - distance_transform_edt(~inner, sampling=spacing))
    depth = np.full(mask.shape, min(outside, float(inner_depth.min())), dtype=np.float32)
    depth[box] = inner_depth
    return depth


def _export_configuration(predictor, **resampling):
    """The model configuration with other orders for nnU-Net's export resampling."""
    from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager

    configuration = copy.deepcopy(predictor.configuration_manager.configuration)
    configuration["resampling_fn_probabilities_kwargs"].update(resampling)
    return ConfigurationManager(configuration)


def shape_based_segmentation(predictor, logits, properties):
    """Restore the segmentation to the input grid with shape-based interpolation.

    The argmax on the model grid is turned into one signed distance map per
    label and resampled with nnU-Net's export resampling at
    MASK_INTERPOLATION_ORDER. Each voxel takes the label it lies deepest
    inside, so a slice between two model slices gets an outline between
    theirs instead of a copy of the nearer one.
    """
    from nnunetv2.inference.export_prediction import (
        convert_predicted_logits_to_segmentation_with_correct_shape,
    )

    label_manager = predictor.label_manager
    if label_manager.has_regions:
        raise ValueError("Shape-based mask restoration needs a label-based model")
    labels = np.asarray(label_manager.convert_logits_to_segmentation(logits))
    spacing = predictor.configuration_manager.spacing
    if len(spacing) != labels.ndim:
        raise ValueError(f"Model spacing {spacing} does not match a {labels.ndim}D prediction")
    # Background scores 0, so a voxel outside every label stays background.
    scores = np.zeros((len(label_manager.all_labels), *labels.shape), dtype=np.float32)
    for channel, value in enumerate(label_manager.all_labels):
        if value != 0:
            scores[channel] = _signed_depth(labels == value, spacing)
    configuration = _export_configuration(
        predictor, order=MASK_INTERPOLATION_ORDER, order_z=MASK_INTERPOLATION_ORDER)
    return convert_predicted_logits_to_segmentation_with_correct_shape(
        torch.from_numpy(scores), predictor.plans_manager, configuration,
        label_manager, properties)


def predict(predictor, in_nifti_dir, out_dir):
    """Run one-case prediction, saving segmentation and probabilities.

    The PACS service always submits one case. Preprocess with nnU-Net's
    synchronous single-array adapter so the full preprocessed volume is not
    transferred through a multiprocessing queue. Docker's default 64 MB
    shared-memory mount is too small for that transfer and kills
    predict_from_files workers. The probabilities use nnU-Net's own export
    with PROBABILITY_ORDER_Z along the slice axis; the segmentation uses
    shape_based_segmentation.

    Returns (seg_nifti_path, prob_npz_path).
    """
    from nnunetv2.inference.data_iterators import PreprocessAdapterFromNpy
    from nnunetv2.inference.export_prediction import (
        convert_predicted_logits_to_segmentation_with_correct_shape,
    )

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
    preprocessed = next(PreprocessAdapterFromNpy(
        [image], [None], [properties], [None], predictor.plans_manager,
        predictor.dataset_json, predictor.configuration_manager,
        num_threads_in_multithreaded=1, verbose=predictor.verbose,
    ))
    data_properties = preprocessed["data_properties"]
    logits = predictor.predict_logits_from_preprocessed_data(preprocessed["data"]).cpu()

    _, probabilities = convert_predicted_logits_to_segmentation_with_correct_shape(
        logits, predictor.plans_manager,
        _export_configuration(predictor, order_z=PROBABILITY_ORDER_Z),
        predictor.label_manager, data_properties, return_probabilities=True,
    )
    np.savez_compressed(f"{output_truncated}.npz", probabilities=probabilities)
    del probabilities
    segmentation = shape_based_segmentation(predictor, logits, data_properties)
    reader_writer.write_seg(
        segmentation, f"{output_truncated}{predictor.dataset_json['file_ending']}",
        data_properties,
    )

    seg = sorted(p for p in out_dir.glob("*.nii.gz"))
    npz = sorted(out_dir.glob("*.npz"))
    if len(seg) != 1 or len(npz) != 1:
        raise RuntimeError(
            f"Expected exactly one seg .nii.gz and one .npz in {out_dir}; "
            f"got seg={[p.name for p in seg]} npz={[p.name for p in npz]}"
        )
    return seg[0], npz[0]
