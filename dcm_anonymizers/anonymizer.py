import os
import tqdm
import pydicom
from pathlib import Path
from datetime import datetime
import logging
logger = logging.getLogger(__name__)

from dicomanonymizer.anonymizer import isDICOMType

from dcm_anonymizers.utils import ensure_dir, list_all_files
from dcm_anonymizers.phi_detectors import DcmPHIDetector
from dcm_anonymizers.img_anonymizers import DCMImageAnonymizer
from dcm_anonymizers.ps_3_3 import DCMPS33Anonymizer, replace_with_value, format_action_dict

class Anonymizer:
    def __init__(self, input_path: str, output_path: str) -> None:
        self.input_path = input_path
        self.output_path = output_path
        self.total_dcms = 0
        self.dcm_dirs = []
        self.series_props = {}
        self.anonymizer = None
        self.img_anonymizer = None

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
        self.img_anonymizer = DCMImageAnonymizer(phi_detector=phi_detector)

    
    def create_output_dirs(self):
        patient_id_map = {}
        count = 1

        for idx, dir in enumerate(self.dcm_dirs):
            
            patientid = self.get_patient_id_from_series_path(dir)           
            
            targetdcm_dir = dir.removeprefix(str(self.input_path))
            targetdcm_dir = targetdcm_dir.removeprefix("/")
            output_path = Path(self.output_path, targetdcm_dir)
            anonymized_id = f"Pseudo-PHI-{str(count).rjust(3, '0')}"

            if patientid != "":
                if patientid in patient_id_map:
                    anonymized_id = patient_id_map[patientid]
                else:
                    patient_id_map[patientid] = anonymized_id
                    count += 1

                output_path = Path(str(output_path).replace(patientid, anonymized_id))


            ensure_dir(output_path)

            self.series_props[dir] = {
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
    
    def create_patient_attrs_action(self, parentdir: str):
        series_info = self.series_props[parentdir]
        
        patient_attrs_action = {
            "(0x0010, 0x0010)": replace_with_value([series_info['anonymized_id']]),
            "(0x0010, 0x0020)": replace_with_value([series_info['anonymized_id']]),
        }

        patient_attrs_action = format_action_dict(patient_attrs_action)

        self.anonymizer.id_dict.update({
            series_info['patiend_id']: series_info['anonymized_id']
        })
        
        return patient_attrs_action
    
    
    def anonymize_metadata_on_file(self, filepath: str, parentdir: str, patient_attrs_action: dict = None):

        if not patient_attrs_action:
            patient_attrs_action = self.create_patient_attrs_action(parentdir)
        
        series_info = self.series_props[parentdir]

        filename = os.path.basename(filepath)
        output_file = f"{series_info['output_path']}/{filename}"

        history = self.anonymizer.anonymize(
            input_path=str(filepath),
            output_path=str(output_file),
            custom_actions=patient_attrs_action,
        )

        return history, output_file
    
    def anonymize_image_data_on_file(self, filepath: str, replace: bool = True):

        output_file = filepath
        if not replace:
            path = Path(filepath)
            parent_path = path.parent.absolute()

            filename = path.stem
            suffix = datetime.now().strftime("%d%m%Y_%H%M%S")
            output_file = str( parent_path / f"{filename}_{suffix}.dcm")

        self.img_anonymizer.anonymize_dicom_file(
            dcm_file=filepath,
            out_file=output_file,
        )

    def run(self):        
        progress_bar = tqdm.tqdm(total=self.total_dcms)
        for idx, dir in enumerate(self.dcm_dirs):
            
            patient_attrs_action = self.create_patient_attrs_action(dir)
            
            dcms = list_all_files(dir)
            for dcm in dcms:
                history, outfile = self.anonymize_metadata_on_file(dcm, dir, patient_attrs_action)
                print(f"{history}")
                #self.anonymize_image_data_on_file(outfile, replace=True)
                progress_bar.update(1)
            
            # if idx > 1:
            #     break
        
        # print(self.anonymizer.uid_dict)
        print(self.anonymizer.id_dict)
        print(self.anonymizer.series_uid_dict)
            
            
DEID_DATASET_ROOT = 'C:/src/midi_b_challange'

anonymizer = Anonymizer(
    input_path=Path(DEID_DATASET_ROOT, 'data/validation_data/input_data'),
    output_path=Path(DEID_DATASET_ROOT, 'anonymizer-output/Pseudo-PHI-DICOM-Data')
)

anonymizer.run()
