import os
import glob
from pathlib import Path
import tqdm
import pydicom

from dicomanonymizer.anonymizer import isDICOMType

from dcm_anonymizers.ps_3_3 import DCMPS33Anonymizer, DcmPHIDetector, replace_with_value, format_action_dict


DEID_DATASET_ROOT = '/home/r079a/Desktop/de-identification/dataset'
SHIFT_DATE_OFFSET = 120

def ensure_dir(path: Path):
    return path.mkdir(parents=True, exist_ok=True)

def list_all_files(target: str, format: str = '.dcm'):
    targetdcm_path = f"{target}/*{format}"
    return glob.glob(targetdcm_path)


class Anonymizer:
    def __init__(self, input_path: str, output_path: str) -> None:
        self.input_path = input_path
        self.output_path = output_path
        self.total_dcms = 0
        self.dcm_dirs = []
        self.output_id_map = {}
        self.anonymizer = None

        ensure_dir(self.output_path)

        self.initialize()
        self.create_output_dirs()

    def initialize(self):

        # List all the directory and dicoms
        alldirs = [x[0] for x in os.walk(self.input_path)]

        for dir in alldirs:
            dcms = list_all_files(dir)
            if len(dcms) > 0:
                self.total_dcms += len(dcms)
                self.dcm_dirs.append(dir)
        
        print(f"Total dicoms to be anonymized: {self.total_dcms}, Total series to be anonymized: {len(self.dcm_dirs)}")

        # initialize model
        phi_detector = DcmPHIDetector()
        self.anonymizer = DCMPS33Anonymizer(phi_detector=phi_detector)

    
    def create_output_dirs(self):
        for idx, dir in enumerate(self.dcm_dirs):
            
            patientid = self.get_patient_id_from_series_path(dir)           
            
            targetdcm_dir = dir.removeprefix(str(self.input_path))
            targetdcm_dir = targetdcm_dir.removeprefix("/")
            output_path = Path(self.output_path, targetdcm_dir)
            anonymized_id = f"Pseudo-PHI-{str(idx+1).rjust(3, '0')}"

            if patientid != "":
                output_path = Path(str(output_path).replace(patientid, anonymized_id))


            ensure_dir(output_path)

            self.output_id_map[dir] = {
                'patiend_id': patientid,
                'anonymized_id': anonymized_id,
                'output_path': str(output_path)
            }    

    def get_patient_id_from_series_path(self, series_path: str):
        patientid = ""
        dcms = list_all_files(series_path)

        if isDICOMType(dcms[0]):
            ds = pydicom.read_file(dcms[0])
            patietid_tag = (0x0010, 0x0020)
            element = ds.get(patietid_tag)
            if element:
                patientid = element.value
        else:
            raise Warning(f"Dicoms in the following directory might be corrupted, hence can not be anonymized. {dir}")
        
        return patientid
    
    def run_on_file(filepath: str):
        
        patient_attrs_action = {
            "(0x0010, 0x0010)": replace_with_value(['Pseudo-PHI-008']),
            "(0x0010, 0x0020)": replace_with_value(['Pseudo-PHI-008']),
        }

        patient_attrs_action = format_action_dict(patient_attrs_action)

        uid_map, id_map, history = anonymizer.anonymize(
            input_path=str(sample_img_path),
            output_path=str(sample_out_path),
            custom_actions=patient_attrs_action,
        )

        print(uid_map)
        print(id_map)
        print(history)

    def run(self):        
        progress_bar = tqdm.tqdm(total=self.total_dcms)

        for idx, dir in enumerate(self.dcm_dirs):
            dcms = list_all_files(dir)
            for dcm in dcms:
                print(dcm)
                progress_bar.update(1)

            if idx > 2:
                break
            


if __name__ == "__main__":
    sample_img_path = Path(DEID_DATASET_ROOT, 'images/manifest-1617826555824/Pseudo-PHI-DICOM-Data/6451050561/07-28-1961-NA-NA-56598/PET IR CTAC WB-48918/1-001.dcm')
    sample_out_path = Path('/home/r079a/Desktop/de-identification/dicom-output', '6451050561/07-28-1961-NA-NA-56598/PET IR CTAC WB-48918')
    
    ensure_dir(sample_out_path)

    anonymizer = Anonymizer(
        input_path=Path(DEID_DATASET_ROOT, 'images/manifest-1617826555824/Pseudo-PHI-DICOM-Data'),
        output_path=Path(DEID_DATASET_ROOT, 'anonymizer-output/Pseudo-PHI-DICOM-Data')
    )

    anonymizer.run()

    # phi_detector = DcmPHIDetector()

    # anonymizer = DCMPS33Anonymizer(phi_detector=phi_detector)
    # # patient_attrs_action = {}

    # patient_attrs_action = {
    #     "(0x0010, 0x0010)": replace_with_value(['Pseudo-PHI-008']),
    #     "(0x0010, 0x0020)": replace_with_value(['Pseudo-PHI-008']),
    # }

    # patient_attrs_action = format_action_dict(patient_attrs_action)

    # uid_map, id_map, history = anonymizer.anonymize(
    #     input_path=str(sample_img_path),
    #     output_path=str(sample_out_path),
    #     custom_actions=patient_attrs_action,
    # )

    # print(uid_map)

    # print(id_map)

    # print(history)