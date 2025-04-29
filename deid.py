import argparse
from pathlib import Path

from dcm_deidentifiers.deidentifier import Anonymizer

DEID_DATASET_ROOT = '/home/r079a/Desktop/de-identification/dataset'
# DEID_DATASET_ROOT = '/home/r079a/Desktop/de-identification/dataset/midi-validation-data'

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_path", 
        default=Path(DEID_DATASET_ROOT, 'images/manifest-1617826555824/Pseudo-PHI-DICOM-Data'),
        help="absolute path of the directory that contains all the dicoms"
    )

    parser.add_argument(
        "--output_path", 
        default=Path(DEID_DATASET_ROOT, 'anonymizer-output/Pseudo-PHI-DICOM-Data-retry'),
        help="absolute path of the directory for the outputs"
    )

    args = parser.parse_args()

    input_path = args.input_path
    if isinstance(input_path, str):
        input_path = Path(input_path)
    
    output_path = args.output_path
    if isinstance(output_path, str):
        output_path = Path(output_path)

    anonymizer = Anonymizer(
        input_path=input_path,
        output_path=output_path,
        detector_logging=True,
    )

    anonymizer.run()
    


# if __name__ == "__main__":
#     anonymizer = Anonymizer(
#         input_path=Path(DEID_DATASET_ROOT, 'input_data'),
#         output_path=Path(DEID_DATASET_ROOT, 'output_data_retry'),
#     )

#     anonymizer.run()

    # dcm_path = anonymizer.get_dcm_path_from_idx(9687)
    # print(dcm_path)