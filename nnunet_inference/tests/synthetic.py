"""Small generated DICOM fixture; no medical images or identifiers."""
from pathlib import Path

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage


def write_image(path, index, *, study_uid="1.2.3", series_uid="1.2.3.4",
                sop_uid=None, orientation=(1, 0, 0, 0, 1, 0),
                pixel_spacing=(1, 1), position=None, pixels=None):
    pixels = np.zeros((2, 3), dtype=np.uint16) if pixels is None else np.asarray(pixels, dtype="<u2")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sop_uid = sop_uid or f"1.2.3.4.{index + 1}"
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = MRImageStorage
    meta.MediaStorageSOPInstanceUID = sop_uid
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    d = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    for key, value in dict(SOPClassUID=MRImageStorage, SOPInstanceUID=sop_uid,
                           StudyInstanceUID=study_uid, SeriesInstanceUID=series_uid,
                           FrameOfReferenceUID="1.2.3.5", Modality="MR",
                           PatientID="SYNTHETIC", PatientName="Synthetic^Test",
                           ReferringPhysicianName="EventName:test_arm_1",
                           Rows=pixels.shape[0], Columns=pixels.shape[1],
                           PixelSpacing=list(pixel_spacing),
                           ImageOrientationPatient=list(orientation),
                           ImagePositionPatient=list((0, 0, index) if position is None else position),
                           SamplesPerPixel=1, PhotometricInterpretation="MONOCHROME2",
                           BitsAllocated=16, BitsStored=16, HighBit=15,
                           PixelRepresentation=0, PixelData=pixels.tobytes()).items():
        setattr(d, key, value)
    d.save_as(path, enforce_file_format=True)
    return path
