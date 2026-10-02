"""The optional DICOM validator must accept valid empty predictions."""
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import SimpleITK as sitk

from nnunet_inference import dicom_io
from nnunet_inference.tests import validate_dicom
from nnunet_inference.tests.synthetic import write_image


class ValidationTests(unittest.TestCase):
    def test_empty_mask_passes_real_dicom_validation_and_writes_overlay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            for index in range(3):
                write_image(source / f"{index}.dcm", index,
                            pixels=np.arange(64).reshape(8, 8) + index)
            native = root / "VS_0000.nii.gz"
            dicom_io.series_to_nifti(dicom_io.read_series(source), native)
            image = sitk.ReadImage(str(native))
            mask_image = sitk.GetImageFromArray(np.zeros(
                sitk.GetArrayFromImage(image).shape, dtype=np.uint16))
            mask_image.CopyInformation(image)
            empty = root / "empty.nii.gz"
            sitk.WriteImage(mask_image, str(empty))
            identity = dicom_io.make_uid_context(
                "Synthetic validation", "a" * 64, "checkpoint_best.pth", (0,), False)
            versions = ["Synthetic validation", "nnUNetv2 test"]
            output = root / "output"
            mask = dicom_io.write_mask_dicom(empty, source, output / "mask", identity,
                                              software_versions=versions)
            probabilities = dicom_io.write_prob_dicom(empty, source, output / "vote_map", identity,
                                                       software_versions=versions)
            summary = dict(mask_dir=mask, vote_map_dir=probabilities, tumor_voxels=0,
                           tumor_pct=0.0, uid_context=identity)
            overlay = root / "overlay.png"
            with patch.multiple(validate_dicom, INPUT_DIR=source, OUTPUT_DIR=output,
                                OVERLAY_PNG=overlay), \
                    patch.dict(os.environ, NNUNET_MODEL_DIR=str(root / "model")), \
                    patch.object(validate_dicom.pipeline, "run_inference", return_value=summary), \
                    redirect_stdout(io.StringIO()):
                validate_dicom.main()
            self.assertTrue(overlay.is_file())
