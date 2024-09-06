from pprint import pprint
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
    
DEID_DATASET_ROOT = '/home/r079a/Desktop/de-identification/dataset/midi-validation-data'

# 1168116
if __name__ == "__main__":
    anonymizer = Anonymizer(
        input_path=Path(DEID_DATASET_ROOT, 'input_data'),
        output_path=Path(DEID_DATASET_ROOT, 'output_data_sample'),
        detector_logging=True,
    )

    anonymizer.run()

    pprint(anonymizer.validator.added_attr_log)

    # dcm_path = anonymizer.get_dcm_path_from_idx(2073)
    # print(dcm_path)