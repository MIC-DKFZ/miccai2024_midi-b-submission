from pathlib import Path

from dcm_anonymizers.anonymizer import Anonymizer

# DEID_DATASET_ROOT = '/home/r079a/Desktop/de-identification/dataset'

# if __name__ == "__main__":
#     anonymizer = Anonymizer(
#         input_path=Path(DEID_DATASET_ROOT, 'images/manifest-1617826555824/Pseudo-PHI-DICOM-Data'),
#         output_path=Path(DEID_DATASET_ROOT, 'anonymizer-output/Pseudo-PHI-DICOM-Data-10-removed-ctp-custom'),
#         detector_logging=True,
#     )

#     anonymizer.run(debug_item=((0x0040, 0x1400), "Requested Procedure Comments"))
    
DEID_DATASET_ROOT = '/home/r079a/Desktop/de-identification/dataset/midi-test-data'

# 1226392
if __name__ == "__main__":
    anonymizer = Anonymizer(
        input_path=Path(DEID_DATASET_ROOT, 'input_data_half'),
        output_path=Path(DEID_DATASET_ROOT, 'output_data_half'),
        detector_logging=True,
    )

    anonymizer.run()