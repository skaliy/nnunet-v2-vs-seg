"""Unit tests for nnunet_inference.predictor."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from nnunet_inference import predictor as predictor_module


class TestResolveDevice(unittest.TestCase):
    def test_auto_matches_cuda_availability(self):
        dev = predictor_module.resolve_device("auto")
        expected = "cuda" if torch.cuda.is_available() else "cpu"
        self.assertEqual(dev.type, expected)

    def test_cpu_is_honoured(self):
        self.assertEqual(predictor_module.resolve_device("cpu").type, "cpu")

    def test_invalid_raises(self):
        with self.assertRaises(ValueError):
            predictor_module.resolve_device("tpu")


class TestValidateModelDir(unittest.TestCase):
    def test_missing_files_raise_with_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError) as ctx:
                predictor_module.build_predictor(tmp, device="cpu")
            self.assertIn("plans.json", str(ctx.exception))


class TestFoldSelection(unittest.TestCase):
    @staticmethod
    def _model_dir(tmp, fold_dirs):
        model_dir = Path(tmp)
        (model_dir / "plans.json").write_text("{}")
        (model_dir / "dataset.json").write_text("{}")
        for fold_dir in fold_dirs:
            path = model_dir / fold_dir
            path.mkdir()
            (path / "checkpoint_final.pth").touch()
        return model_dir

    def test_auto_detects_fold_all(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = self._model_dir(tmp, ["fold_all"])
            self.assertEqual(
                predictor_module.resolve_folds(
                    model_dir, "checkpoint_final.pth", "auto"
                ),
                ("all",),
            )

    def test_auto_detects_complete_five_fold_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = self._model_dir(
                tmp, [f"fold_{fold}" for fold in range(5)]
            )
            self.assertEqual(
                predictor_module.resolve_folds(
                    model_dir, "checkpoint_final.pth", "auto"
                ),
                (0, 1, 2, 3, 4),
            )

    def test_auto_rejects_incomplete_five_fold_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = self._model_dir(tmp, ["fold_0", "fold_1"])
            with self.assertRaises(FileNotFoundError) as ctx:
                predictor_module.resolve_folds(
                    model_dir, "checkpoint_final.pth", "auto"
                )
            self.assertIn("incomplete five-fold", str(ctx.exception))

    def test_explicit_fold_list_is_normalized(self):
        self.assertEqual(
            predictor_module.normalize_folds("0,1,2,3,4"),
            (0, 1, 2, 3, 4),
        )

    def test_build_passes_five_folds_and_tta_setting_to_predictor(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = self._model_dir(
                tmp, [f"fold_{fold}" for fold in range(5)]
            )
            with mock.patch.object(
                predictor_module, "_load_predictor_class"
            ) as load:
                cls = load.return_value
                predictor = predictor_module.build_predictor(
                    model_dir, device="cpu", disable_tta=True, folds="auto"
                )

            self.assertIs(predictor, cls.return_value)
            self.assertFalse(cls.call_args.kwargs["use_mirroring"])
            cls.return_value.initialize_from_trained_model_folder.assert_called_once_with(
                str(model_dir), use_folds=(0, 1, 2, 3, 4),
                checkpoint_name="checkpoint_final.pth",
            )

    def test_tta_flips_inferior_superior_and_anterior_posterior(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = self._model_dir(
                tmp, [f"fold_{fold}" for fold in range(5)]
            )
            for trained, expected in [((0, 1, 2), (0, 1)), (None, None), ((1, 2), None)]:
                with mock.patch.object(
                    predictor_module, "_load_predictor_class"
                ) as load:
                    load.return_value.return_value.allowed_mirroring_axes = trained
                    predictor = predictor_module.build_predictor(
                        model_dir, device="cpu", folds="auto"
                    )
                self.assertTrue(load.return_value.call_args.kwargs["use_mirroring"])
                self.assertEqual(predictor.allowed_mirroring_axes, expected)


class TestMirroringPasses(unittest.TestCase):
    def test_tta_is_original_plus_one_combined_flip(self):
        try:
            nnUNetPredictor = predictor_module._load_predictor_class()
        except ImportError:
            self.skipTest("nnunetv2 is not installed")
        inputs = []

        def network(x):
            inputs.append(x.clone())
            return x * torch.arange(x.shape[2], dtype=x.dtype).view(1, 1, -1, 1, 1)

        predictor = nnUNetPredictor(device=torch.device("cpu"), allow_tqdm=False)
        predictor.network = network
        predictor.use_mirroring = True
        predictor.allowed_mirroring_axes = predictor_module.TTA_FLIP_AXES
        x = torch.rand(1, 1, 4, 3, 2)
        result = predictor._internal_maybe_mirror_and_predict(x)

        self.assertEqual(len(inputs), 2)
        torch.testing.assert_close(inputs[1], torch.flip(x, (2, 3)))
        expected = (network(x) + torch.flip(network(torch.flip(x, (2, 3))), (2, 3))) / 2
        torch.testing.assert_close(result, expected)


def _ball(shape, spacing, centre_mm, radius_mm):
    """Ball sampled at voxel centres, measured from the volume edge."""
    grids = np.meshgrid(*[(np.arange(n) + 0.5) * s for n, s in zip(shape, spacing)],
                        indexing="ij")
    return sum((g - c) ** 2 for g, c in zip(grids, centre_mm)) <= radius_mm ** 2


def _dice(a, b):
    return 2 * (a & b).sum() / (a.sum() + b.sum())


def _identical_neighbours(mask):
    return sum(mask[k].any() and np.array_equal(mask[k], mask[k + 1])
               for k in range(len(mask) - 1))


class TestShapeBasedSegmentation(unittest.TestCase):
    """A ball predicted on 1.5 mm slices and restored to 1.0 mm slices."""

    def setUp(self):
        from nnunetv2.utilities.label_handling.label_handling import LabelManager
        from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager

        model_shape, self.target_shape = (16, 48, 48), (24, 48, 48)
        centre, radius = (12.0, 9.6, 9.6), 6.0
        # 1.5 / 0.4 is above nnU-Net's anisotropy threshold, so its export
        # resamples the slice axis separately with order_z.
        self.configuration = {
            "spacing": [1.5, 0.4, 0.4],
            "architecture": {},
            "resampling_fn_probabilities": "resample_data_or_seg_to_shape",
            "resampling_fn_probabilities_kwargs": {
                "is_seg": False, "order": 1, "order_z": 0, "force_separate_z": None},
        }
        self.predictor = SimpleNamespace(
            label_manager=LabelManager({"background": 0, "tumor": 1}, regions_class_order=None),
            configuration_manager=ConfigurationManager(self.configuration),
            plans_manager=SimpleNamespace(transpose_forward=[0, 1, 2],
                                          transpose_backward=[0, 1, 2]),
        )
        self.properties = {
            "spacing": [1.0, 0.4, 0.4],
            "shape_after_cropping_and_before_resampling": self.target_shape,
            "shape_before_cropping": self.target_shape,
            "bbox_used_for_cropping": [[0, n] for n in self.target_shape],
        }
        ball = _ball(model_shape, (1.5, 0.4, 0.4), centre, radius)
        self.logits = torch.from_numpy(np.stack([~ball, ball]).astype(np.float32) * 10 - 5)
        self.truth = _ball(self.target_shape, (1.0, 0.4, 0.4), centre, radius)

    def test_restores_outline_between_model_slices(self):
        from nnunetv2.inference.export_prediction import (
            convert_predicted_logits_to_segmentation_with_correct_shape,
        )

        shape_based = predictor_module.shape_based_segmentation(
            self.predictor, self.logits, self.properties) > 0
        nearest = convert_predicted_logits_to_segmentation_with_correct_shape(
            self.logits, self.predictor.plans_manager, self.predictor.configuration_manager,
            self.predictor.label_manager, self.properties) > 0

        self.assertEqual(shape_based.shape, self.target_shape)
        self.assertGreater(_dice(shape_based, self.truth), _dice(nearest, self.truth))
        self.assertLess(_identical_neighbours(shape_based), _identical_neighbours(nearest))
        self.assertLess(abs(shape_based.sum() / self.truth.sum() - 1), 0.1)
        # The configured export resampling itself is left untouched.
        self.assertEqual(
            self.configuration["resampling_fn_probabilities_kwargs"]["order_z"], 0)

    def test_probabilities_are_interpolated_between_model_slices(self):
        from nnunetv2.inference.export_prediction import (
            convert_predicted_logits_to_segmentation_with_correct_shape,
        )

        def foreground(configuration):
            _, probabilities = convert_predicted_logits_to_segmentation_with_correct_shape(
                self.logits, self.predictor.plans_manager, configuration,
                self.predictor.label_manager, self.properties, return_probabilities=True)
            return probabilities[1]

        def repeated_slices(probability):
            return sum(probability[k].max() > 0.5
                       and np.array_equal(probability[k], probability[k + 1])
                       for k in range(len(probability) - 1))

        def blended(probability):
            return int(((probability > 0.1) & (probability < 0.9)).sum())

        linear = foreground(predictor_module._export_configuration(
            self.predictor, order_z=predictor_module.PROBABILITY_ORDER_Z))
        nearest = foreground(self.predictor.configuration_manager)
        self.assertEqual(linear.shape, self.target_shape)
        # Only the slices on either side of the ball's centre stay identical.
        self.assertLess(repeated_slices(linear), repeated_slices(nearest))
        self.assertEqual(blended(nearest), 0)
        self.assertGreater(blended(linear), 0)

    def test_empty_prediction_stays_empty(self):
        empty = torch.stack([torch.ones(16, 48, 48), -torch.ones(16, 48, 48)])
        mask = predictor_module.shape_based_segmentation(
            self.predictor, empty, self.properties)
        self.assertEqual(mask.shape, self.target_shape)
        self.assertEqual(int(mask.sum()), 0)


class TestPredictSingleCase(unittest.TestCase):
    def test_writes_nnunet_probabilities_and_shape_based_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            input_file = source / "VS_0000.nii.gz"
            input_file.touch()

            reader_writer = mock.Mock()
            reader_writer.read_images.return_value = (
                mock.sentinel.image,
                {"spacing": (1, 1, 1)},
            )
            reader_writer.write_seg.side_effect = (
                lambda segmentation, path, properties: Path(path).touch())
            predictor = mock.Mock(verbose=False, dataset_json={"file_ending": ".nii.gz"})
            predictor.plans_manager.image_reader_writer_class.return_value = (
                reader_writer
            )
            configuration = {
                "architecture": {},
                "resampling_fn_probabilities_kwargs": {
                    "is_seg": False, "order": 1, "order_z": 0, "force_separate_z": None},
            }
            predictor.configuration_manager.configuration = configuration
            logits = torch.zeros(2, 2, 2, 2)
            predictor.predict_logits_from_preprocessed_data.return_value = logits
            preprocessed = {"data": mock.sentinel.data,
                            "data_properties": mock.sentinel.properties}
            probabilities = np.full((2, 2, 2, 2), 0.5, dtype=np.float32)

            with mock.patch(
                "nnunetv2.inference.data_iterators.PreprocessAdapterFromNpy",
                return_value=iter([preprocessed]),
            ) as adapter, mock.patch(
                "nnunetv2.inference.export_prediction."
                "convert_predicted_logits_to_segmentation_with_correct_shape",
                return_value=(mock.sentinel.nearest_mask, probabilities),
            ) as convert, mock.patch.object(
                predictor_module, "shape_based_segmentation",
                return_value=mock.sentinel.mask,
            ) as shape_based:
                seg, probability_file = predictor_module.predict(
                    predictor, source, output
                )

            self.assertEqual(seg, output / "VS.nii.gz")
            self.assertEqual(probability_file, output / "VS.npz")
            np.testing.assert_array_equal(
                np.load(probability_file)["probabilities"], probabilities)
            reader_writer.read_images.assert_called_once_with([str(input_file)])
            self.assertEqual(adapter.call_args.args[0], [mock.sentinel.image])
            predictor.predict_logits_from_preprocessed_data.assert_called_once_with(
                mock.sentinel.data)
            self.assertIs(convert.call_args.args[0], logits)
            self.assertTrue(convert.call_args.kwargs["return_probabilities"])
            exported = convert.call_args.args[2].configuration
            self.assertEqual(exported["resampling_fn_probabilities_kwargs"]["order_z"],
                             predictor_module.PROBABILITY_ORDER_Z)
            self.assertEqual(configuration["resampling_fn_probabilities_kwargs"]["order_z"], 0)
            self.assertIs(shape_based.call_args.args[1], logits)
            self.assertIs(shape_based.call_args.args[2], mock.sentinel.properties)
            reader_writer.write_seg.assert_called_once_with(
                mock.sentinel.mask, str(output / "VS.nii.gz"), mock.sentinel.properties)
            predictor.predict_single_npy_array.assert_not_called()
            predictor.predict_from_files.assert_not_called()

    def test_rejects_more_than_one_input_case(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            source.mkdir()
            (source / "A_0000.nii.gz").touch()
            (source / "B_0000.nii.gz").touch()
            with self.assertRaisesRegex(RuntimeError, "exactly one"):
                predictor_module.predict(mock.Mock(), source, Path(tmp) / "out")


if __name__ == "__main__":
    unittest.main()
