import os
import tqdm
import csv
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
        
        print(f"Total dicoms in the input data: {self.total_dcms}, Total series in the input data: {len(self.dcm_dirs)}")

        # initialize model
        phi_detector = DcmPHIDetector()
        self.anonymizer = DCMPS33Anonymizer(phi_detector=phi_detector)
        self.img_anonymizer = DCMImageAnonymizer(phi_detector=phi_detector)

    
    def create_output_dirs(self):
        patient_id_map = {}
        count = 1

        for idx, dir in enumerate(self.dcm_dirs):
            
            patientid, studyuid, seriesuid = self.get_series_infos_from_file(dir)           
            
            targetdcm_dir = dir.removeprefix(str(self.input_path))
            targetdcm_dir = targetdcm_dir.removeprefix("/")
            output_path = Path(str(self.output_path) + targetdcm_dir)
            anonymized_id = f"Pseudo-PHI-{str(count).rjust(3, '0')}"
            anonymized_study_uid = self.anonymizer.get_UID(studyuid)
            anonymized_series_uid = self.anonymizer.get_UID(seriesuid)

            if patientid != "":
                if patientid in patient_id_map:
                    anonymized_id = patient_id_map[patientid]
                else:
                    patient_id_map[patientid] = anonymized_id
                    count += 1

                output_path = Path(str(output_path).replace(patientid, anonymized_id))


            ensure_dir(output_path)
            
            already_anonymized_dcms = list_all_files(output_path)
            if len(already_anonymized_dcms) > 0:
                logger.debug(f"Already anonymized dicoms in the following directory: {output_path}, {len(already_anonymized_dcms)}")
            self.total_dcms -= len(already_anonymized_dcms)

            self.series_props[dir] = {
                'series_uid': seriesuid,
                'patient_id': patientid,
                'anonymized_id': anonymized_id,
                'output_path': str(output_path)
            }    

    def get_series_infos_from_file(self, series_filepath: str):
        patientid = ""
        studyuid = ""
        seriesuid = ""
        dcms = list_all_files(series_filepath)

        if isDICOMType(dcms[0]):
            ds = pydicom.read_file(dcms[0])

            patietid_tag = (0x0010, 0x0020)
            studyuid_tag = (0x0020, 0x000D)
            seriesuid_tag = (0x0020, 0x000E)

            patientid_element = ds.get(patietid_tag)
            if patientid_element:
                patientid = patientid_element.value
            
            studyuid_element = ds.get(studyuid_tag)
            if studyuid_element:
                studyuid = studyuid_element.value

            seriesuid_element = ds.get(seriesuid_tag)
            if seriesuid_element:
                seriesuid = seriesuid_element.value
            
        else:
            raise Warning(f"Dicoms in the following directory might be corrupted, hence can not be anonymized. {dir}")
        
        return patientid, studyuid, seriesuid
    
    def create_patient_attrs_action(self, parentdir: str):
        series_info = self.series_props[parentdir]
        
        patient_attrs_action = {
            "(0x0010, 0x0010)": replace_with_value([series_info['anonymized_id']]),
            "(0x0010, 0x0020)": replace_with_value([series_info['anonymized_id']]),
        }

        patient_attrs_action = format_action_dict(patient_attrs_action)

        self.anonymizer.id_dict.update({
            series_info['patient_id']: series_info['anonymized_id']
        })
        
        return patient_attrs_action
    
    def anonymized_file_exists(self, filepath: str, parentdir: str):
        series_info = self.series_props[parentdir]
        filename = os.path.basename(filepath)
        output_file = f"{series_info['output_path']}/{filename}"

        return os.path.exists(output_file)
    
    
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

    def get_series_output_path_map(self):
        series_output_map = {}
        for series_path in self.series_props:
            series_prop = self.series_props[series_path]
            series_uid = series_prop['series_uid']
            output_path = series_prop['output_path']
            output_path = output_path.replace(str(self.output_path), '.')
            series_output_map[series_uid] = output_path
        return series_output_map

    def export_csv_from_id_map(self, id_map: dict, filename: str, fields: list = []):
        if len(fields) == 0:
            fields = ['id_old', 'id_new']
        
        data = [{fields[0]: key, fields[1]: val} for key, val in id_map.items()]
        csvfile = os.path.join(self.output_path, f"{filename}.csv")

        with open(csvfile, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fields)
            writer.writeheader()
            writer.writerows(data)


    def run(self):        
        print(f"Total dicoms to be anonymized: {self.total_dcms}")
        progress_bar = tqdm.tqdm(total=self.total_dcms)
        for idx, dir in enumerate(self.dcm_dirs):
            
            patient_attrs_action = self.create_patient_attrs_action(dir)
            
            dcms = list_all_files(dir)
            for dcm in dcms:
                if not self.anonymized_file_exists(dcm, dir):
                    history, outfile = self.anonymize_metadata_on_file(dcm, dir, patient_attrs_action)
                    logger.debug(f"{history}")
                    #self.anonymize_image_data_on_file(outfile, replace=True)
                    progress_bar.update(1)
            
        progress_bar.close()
        
        series_output_map = self.get_series_output_path_map()
       
        self.export_csv_from_id_map(self.anonymizer.id_dict, filename="patient_id_mapping")
        self.export_csv_from_id_map(self.anonymizer.uid_dict, filename="uid_mapping")
        self.export_csv_from_id_map(series_output_map, filename='path_mapping', fields=['id_old', 'path'])

        print(self.img_anonymizer.change_log)
        
        
