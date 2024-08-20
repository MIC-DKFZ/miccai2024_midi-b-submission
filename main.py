from pathlib import Path

from dcm_anonymizers.anonymizer import Anonymizer

# DEID_DATASET_ROOT = '/home/r079a/Desktop/de-identification/dataset'

# if __name__ == "__main__":
#     anonymizer = Anonymizer(
#         input_path=Path(DEID_DATASET_ROOT, 'images/manifest-1617826555824/Pseudo-PHI-DICOM-Data'),
#         output_path=Path(DEID_DATASET_ROOT, 'anonymizer-output/Pseudo-PHI-DICOM-Data-3')
#     )

#     anonymizer.run()
    
DEID_DATASET_ROOT = '/home/r079a/Desktop/de-identification/dataset/midi-validation-data'

# 404058
if __name__ == "__main__":
    anonymizer = Anonymizer(
        input_path=Path(DEID_DATASET_ROOT, 'input_data'),
        output_path=Path(DEID_DATASET_ROOT, 'output_data')
    )

    anonymizer.run()