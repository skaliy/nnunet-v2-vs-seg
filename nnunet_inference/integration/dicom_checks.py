"""Checks shared by validate_output.py and validate_dicom.py."""
from pydicom.uid import UID


def check_derived_identity(dataset, path):
    """A derived file needs valid Series and SOP Instance UIDs of at most 64
    characters, and file meta that names the same SOP instance."""
    series_uid = str(dataset.SeriesInstanceUID)
    sop_uid = str(dataset.SOPInstanceUID)
    if not UID(series_uid).is_valid or len(series_uid) > 64:
        raise RuntimeError(f"invalid SeriesInstanceUID in {path}: {series_uid}")
    if not UID(sop_uid).is_valid or len(sop_uid) > 64:
        raise RuntimeError(f"invalid SOPInstanceUID in {path}: {sop_uid}")
    if str(dataset.file_meta.MediaStorageSOPInstanceUID) != sop_uid:
        raise RuntimeError(f"dataset/file-meta SOP UID mismatch in {path}")
