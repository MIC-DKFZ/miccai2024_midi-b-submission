from pathlib import Path

from dcm_anonymizers.anonymizer import Anonymizer

DEID_DATASET_ROOT = '/home/r079a/Desktop/de-identification/dataset'

if __name__ == "__main__":
    anonymizer = Anonymizer(
        input_path=Path(DEID_DATASET_ROOT, 'images/manifest-1617826555824/Pseudo-PHI-DICOM-Data/339833062/07-05-2001-NA-NA-19638/3001578.000000-NA-60758'),
        output_path=Path(DEID_DATASET_ROOT, 'anonymizer-output/Pseudo-PHI-DICOM-Data')
    )

    anonymizer.run()