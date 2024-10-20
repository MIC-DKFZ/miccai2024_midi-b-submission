from pathlib import Path

from dcm_deidentifers.deidentifer import Anonymizer

# DEID_DATASET_ROOT = '/home/r079a/Desktop/de-identification/dataset'

# if __name__ == "__main__":
#     anonymizer = Anonymizer(
#         input_path=Path(DEID_DATASET_ROOT, 'images/manifest-1617826555824/Pseudo-PHI-DICOM-Data'),
#         output_path=Path(DEID_DATASET_ROOT, 'anonymizer-output/Pseudo-PHI-DICOM-Data-10-removed-ctp-custom'),
#         detector_logging=True,
#     )

#     anonymizer.run()
    
DEID_DATASET_ROOT = '/home/r079a/Desktop/de-identification/dataset/midi-test-data'

# 1221642
if __name__ == "__main__":
    anonymizer = Anonymizer(
        input_path=Path(DEID_DATASET_ROOT, 'input_data'),
        output_path=Path(DEID_DATASET_ROOT, 'output_data_sample'),
    )

    # anonymizer.run()

    dcm_path = anonymizer.get_dcm_path_from_idx(9687)
    print(dcm_path)