from pathlib import Path
from dcm_anonymizers.ps_3_3 import DCMPS33Anonymizer, DcmPHIDetector


DEID_DATASET_ROOT = '/home/r079a/Desktop/de-identification/dataset'
SHIFT_DATE_OFFSET = 120

def ensure_dir(path: Path):
    return path.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    sample_img_path = Path(DEID_DATASET_ROOT, 'images/manifest-1617826555824/Pseudo-PHI-DICOM-Data/292821506/07-13-2013-NA-XR CHEST AP PORTABLE for Douglas Davidson-46198/1002.000000-NA-53238')
    sample_out_path = Path('/home/r079a/Desktop/de-identification/dicom-output', '292821506/07-13-2013-NA-XR CHEST AP PORTABLE for Douglas Davidson-46198/1002.000000-NA-53238')
    
    ensure_dir(sample_out_path)

    phi_detector = DcmPHIDetector()

    anonymizer = DCMPS33Anonymizer(phi_detector=phi_detector)

    uid_map, id_map, history = anonymizer.anonymize(
        input_path=str(sample_img_path),
        output_path=str(sample_out_path)
    )

    print(uid_map)

    print(id_map)

    print(history)