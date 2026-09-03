"""Unit tests for nnunet_inference.predictor."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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


class TestPredictSingleCase(unittest.TestCase):
    def test_uses_synchronous_single_array_api(self):
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
            predictor = mock.Mock()
            predictor.plans_manager.image_reader_writer_class.return_value = (
                reader_writer
            )

            def write_prediction(**kwargs):
                truncated = Path(kwargs["output_file_truncated"])
                Path(str(truncated) + ".nii.gz").touch()
                Path(str(truncated) + ".npz").touch()

            predictor.predict_single_npy_array.side_effect = write_prediction

            seg, probabilities = predictor_module.predict(
                predictor, source, output
            )

            self.assertEqual(seg, output / "VS.nii.gz")
            self.assertEqual(probabilities, output / "VS.npz")
            reader_writer.read_images.assert_called_once_with([str(input_file)])
            predictor.predict_single_npy_array.assert_called_once_with(
                input_image=mock.sentinel.image,
                image_properties={"spacing": (1, 1, 1)},
                segmentation_previous_stage=None,
                output_file_truncated=str(output / "VS"),
                save_or_return_probabilities=True,
            )
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
