import os
import shutil
import tqdm
import csv
import pydicom
from pathlib import Path
from datetime import datetime
import logging

timestamp = datetime.now().strftime("%Y_%m_%d_%I_%M%p")
logging.basicConfig(
    format='%(name)s :: %(levelname)-8s :: %(message)s',
    level=logging.DEBUG,
    handlers=[
        logging.FileHandler(filename=".logs/debug_{}.log".format(timestamp), mode = "a"),
        logging.StreamHandler()
    ],
    force=True
)


from dicomanonymizer.anonymizer import isDICOMType

from dcm_anonymizers.utils import ensure_dir, list_all_files
from dcm_anonymizers.phi_detectors import DcmPHIDetector, DcmRobustPHIDetector
from dcm_anonymizers.img_anonymizers import DCMImageAnonymizer
from dcm_anonymizers.ps_3_3 import DCMPS33Anonymizer, replace_with_value, format_action_dict
from dcm_anonymizers.tcia_deid import DCMTCIAAnonymizer
from dcm_anonymizers.private_tags_extractor import PrivateTagsExtractorV2
from dcm_validator.dciodvfy import DCIodValidator

class Anonymizer:
    """
    A class for anonymizing DICOM files by modifying metadata, replacing sensitive 
    patient identifiers, and validating the processed files.

    Features:
        - Extracts series-level information (Patient ID, Study UID, Series UID).
        - Generates anonymized patient attributes and maps identifiers.
        - Anonymizes metadata and image data within DICOM files.
        - Validates DICOM files post-anonymization and ensures compliance.
        - Supports debugging through custom checks and retrieving specific DICOM files.
        - Exports mappings of anonymized identifiers and file paths to CSV.

    Key Methods:
        - `get_series_infos_from_file(series_filepath)`: Extracts patient and series-related metadata.
        - `create_patient_attrs_action(parentdir)`: Generates replacement actions for patient attributes.
        - `anonymized_file_exists(filepath, parentdir)`: Checks if a file has already been anonymized.
        - `anonymize_metadata_on_file(filepath, parentdir, patient_attrs_action)`: Applies metadata anonymization.
        - `validate_dicom_file(target_dcm_path, source_dcm_path, ensure_validation)`: Compares and validates files.
        - `anonymize_image_data_on_file(filepath, replace)`: Anonymizes pixel data in a DICOM file.
        - `get_series_output_path_map()`: Returns a mapping of series UIDs to output paths.
        - `export_csv_from_id_map(id_map, filename, fields)`: Exports anonymized ID mappings to CSV.
        - `run(debug_item)`: Executes the anonymization pipeline on all DICOM files.
        - `get_dcm_path_from_idx(target_idx)`: Retrieves the DICOM file path at a given index for debugging.

    This class ensures the anonymization process is robust, efficient, and compliant with 
    medical data privacy regulations.
    """

    def __init__(
            self, 
            input_path: str,
            output_path: str,
            preserve_dir_struct: bool = False, 
            detector_logging: bool = False
        ) -> None:
        """
        Initializes all the class attributes and create the appropriate output directory
        for the all provided DICOM series.

        Attributes:
            input_path (str): Path to the input directory containing DICOM files.
            output_path (Path): Path to the output directory where anonymized files will be stored.
            preserve_dir_struct (bool): Whether to preserve the original directory structure in the output.
            detector_logging (bool): Whether to enable logging for the PHI detector.
            data_output_dir (Path): Path to the directory for storing anonymized DICOM data.
            mappings_output_dir (Path): Path to the directory for storing anonymization mappings.
            total_dcms (int): Total number of DICOM files in the input dataset.
            dcm_dirs (list): List of directories containing DICOM files.
            series_props (dict): Dictionary mapping series directories to their anonymization properties.
            detector (DcmRobustPHIDetector): Instance of the PHI detection model.
            anonymizer (DCMTCIAAnonymizer): Anonymizer for handling PHI in DICOM metadata.
            img_anonymizer (DCMImageAnonymizer): Anonymizer for handling PHI in DICOM images.
            validator (DCIodValidator): Validator for checking compliance of anonymized DICOM files.
            logger (logging.Logger): Logger for tracking anonymization progress and issues.
        """
        output_path = Path(output_path)

        self.input_path = input_path
        self.data_output_dir = output_path / 'data'
        self.mappings_output_dir = output_path / 'mappings'
        self.total_dcms = 0
        self.dcm_dirs = []
        self.series_props = {}
        self.detector: DcmRobustPHIDetector = None
        self.anonymizer = None
        self.img_anonymizer = None
        self.preserve_dir_struct = preserve_dir_struct
        self.detector_logging = detector_logging
        self.validator: DCIodValidator = None

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        ensure_dir(self.data_output_dir)
        ensure_dir(self.mappings_output_dir)

        self.initialize()
        self.create_output_dirs()

    def initialize(self):
        """
        Initializes the anonymization process by:
        - Scanning the input directory to list all DICOM files and directories.
        - Counting the total number of DICOM files and series.
        - Setting up the PHI detection and anonymization models.
        - Initializing the DICOM validator.

        Raises:
            Exception: If an error occurs while initializing the PHI detector or anonymization models.
        """

        # List all the directory and dicoms
        alldirs = [x[0] for x in os.walk(self.input_path)]

        for dir in alldirs:
            dcms = list_all_files(dir)
            
            if len(dcms) > 0:
                self.total_dcms += len(dcms)
                self.dcm_dirs.append(dir)
        
        print(f"Total dicoms in the input data: {self.total_dcms}, Total series in the input data: {len(self.dcm_dirs)}")

        # initialize model
        phi_detector = DcmRobustPHIDetector(
            logging=self.detector_logging
        )
        ptags_extr = PrivateTagsExtractorV2()
        # self.anonymizer = DCMPS33Anonymizer(phi_detector=phi_detector)
        self.anonymizer = DCMTCIAAnonymizer(
            phi_detector=None,
            notes_phi_detector=phi_detector,
            soft_detection=True,
            private_tags_extractor=ptags_extr
        )
        self.img_anonymizer = DCMImageAnonymizer(phi_detector=phi_detector)
        self.detector = phi_detector
        self.validator = DCIodValidator()

    
    def create_output_dirs(self):
        """
        Creates output directories for anonymized DICOM files while ensuring unique anonymized identifiers.
        - Generates pseudonymized patient IDs and UIDs for anonymization.
        - Maps original patient IDs to anonymized IDs while maintaining consistency.
        - Creates necessary directories and handles existing anonymized files.

        Raises:
            Exception: If an error occurs while creating output directories.
        """
        patient_id_map = {}
        count = 1

        for idx, dir in enumerate(self.dcm_dirs):
            
            patientid, studyuid, seriesuid = self.get_series_infos_from_file(dir)           
            
            targetdcm_dir = dir.removeprefix(str(self.input_path))
            targetdcm_dir = targetdcm_dir.removeprefix("/")
            output_path = Path(str(self.data_output_dir), targetdcm_dir)
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

            if not self.preserve_dir_struct:
                output_path = Path(str(self.data_output_dir), anonymized_id, anonymized_study_uid, anonymized_series_uid)

            ensure_dir(output_path)
            
            already_anonymized_dcms = list_all_files(output_path)
            if len(already_anonymized_dcms) > 0:
                self.logger.debug(f"Already anonymized dicoms in the following directory: {output_path}, {len(already_anonymized_dcms)}")
            self.total_dcms -= len(already_anonymized_dcms)

            self.series_props[dir] = {
                'series_uid': seriesuid,
                'patient_id': patientid,
                'anonymized_id': anonymized_id,
                'output_path': str(output_path)
            }    

    def get_series_infos_from_file(self, series_filepath: str):
        """
        Extracts patient ID, study UID, and series UID from the first DICOM file in the given directory.

        Args:
            series_filepath (str): Path to the directory containing DICOM files.

        Returns:
            tuple: A tuple containing (patient ID, study UID, series UID).

        Raises:
            Warning: If the DICOM files in the directory are corrupted or unreadable.
        """
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
        """
        Creates a dictionary of patient attributes to be modified during anonymization.
        Use the randomly generated patient id to anonymize the patient id (0010,0010) and 
        Patient's name (0010,0020) attribute in final anonymized dicom. Pass the value of 
        these two attributes to the anonymizer's `id_dict` attribute.

        Args:
            parentdir (str): Directory path corresponding to a DICOM series.

        Returns:
            dict: A dictionary mapping DICOM tags to their replacement values for anonymization.
        """
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
        """
        Checks if an anonymized version of a given DICOM file already exists.

        Args:
            filepath (str): Path to the original DICOM file.
            parentdir (str): Directory path corresponding to a DICOM series.

        Returns:
            bool: True if the anonymized file exists, False otherwise.
        """
        series_info = self.series_props[parentdir]
        filename = os.path.basename(filepath)
        output_file = f"{series_info['output_path']}/{filename}"

        return os.path.exists(output_file)
    
    
    def anonymize_metadata_on_file(self, filepath: str, parentdir: str, patient_attrs_action: dict = None):
        """
        Anonymizes metadata of a given DICOM file and saves the anonymized version to the output directory.

        Args:
            filepath (str): Path to the original DICOM file.
            parentdir (str): Directory path corresponding to a DICOM series.
            patient_attrs_action (dict, optional): Custom attribute modification actions. 
                If not provided, it will be generated based on the series properties.

        Returns:
            tuple: A tuple containing (anonymization history, output file path).
        """

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
    
    def validate_dicom_file(self, target_dcm_path: str, source_dcm_path: str = "", ensure_validation: bool = True):
        """
        Validates a DICOM file after anonymization by comparing it with the source file 
        and populating missing attributes if necessary.

        Args:
            target_dcm_path (str): Path to the anonymized DICOM file.
            source_dcm_path (str, optional): Path to the original DICOM file for validation comparison. Defaults to "".
            ensure_validation (bool, optional): Whether to enforce validation by comparing with the source file. Defaults to True.
        """
        valid_after_anonymization = True
        if ensure_validation and source_dcm_path != "":
            valid_after_anonymization = self.validator.compare_dicom_validations(source_dcm_path, target_dcm_path)
        
        if valid_after_anonymization:
            self.validator.populate_missing_attributes(target_dcm_path)
        else:
            print(f"Validation Failed after anonymization on file: {source_dcm_path}")

    
    def anonymize_image_data_on_file(self, filepath: str, replace: bool = True):
        """
        Anonymizes the pixel data of a DICOM image file and saves the anonymized version to the output directory.

        Args:
            filepath (str): Path to the DICOM file to be anonymized.
            replace (bool, optional): If True, replaces the original file; otherwise, saves it with a timestamped name. Defaults to True.
        """

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
        """
        Creates a mapping of series UIDs to their corresponding anonymized output directory paths.

        Returns:
            dict: A dictionary mapping series UIDs to relative anonymized output paths.
        """
        series_output_map = {}
        for series_path in self.series_props:
            series_prop = self.series_props[series_path]
            series_uid = series_prop['series_uid']
            output_path = series_prop['output_path']
            output_path = output_path.replace(str(self.data_output_dir), '.')
            series_output_map[series_uid] = output_path
        return series_output_map

    def export_csv_from_id_map(self, id_map: dict, filename: str, fields: list = []):
        """
        Exports a mapping of old and new IDs to a CSV file.

        Args:
            id_map (dict): A dictionary mapping original IDs to anonymized IDs.
            filename (str): Name of the output CSV file (without extension).
            fields (list, optional): Column names for the CSV file. Defaults to ['id_old', 'id_new'].
        """
        if len(fields) == 0:
            fields = ['id_old', 'id_new']
        
        data = [{fields[0]: key, fields[1]: val} for key, val in id_map.items()]
        csvfile = os.path.join(self.mappings_output_dir, f"{filename}.csv")

        with open(csvfile, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fields)
            writer.writeheader()
            writer.writerows(data)

    def run_custom_checks_on_dcm(self, dcmpath, tag, name):
        """
        Runs custom validation checks on a DICOM file by examining a specific tag. 
        Used mostly for debugging after the anonymization.

        Args:
            dcmpath (str): Path to the DICOM file.
            tag (tuple): The DICOM tag to check (group, element).
            name (str): Expected name of the tag.

        Returns:
            tuple: A tuple containing (element_found, non_empty), where:
                - element_found (int): Number of occurrences of the specified tag.
                - non_empty (int): Number of non-empty values found for the tag.
        """
        dataset = pydicom.dcmread(dcmpath)
        
        element_found = 0
        non_empty = 0

        for element in dataset.elements():
            if element.tag == tag:
                if isinstance(element, pydicom.dataelem.RawDataElement):
                    element = pydicom.dataelem.DataElement_from_raw(element)
                assert element.name == name
                element_val = element.value
                if isinstance(element_val, bytes):
                    element_val = element_val.decode('utf-8')
                element_found += 1
                if element_val != '':
                    non_empty += 1
                    # element_val = element_val.encode('ISO-8859–1')
                    print(element_val)


        return element_found, non_empty

    def run(self, debug_item: tuple = None):   
        """
        Executes the full anonymization process for all DICOM files in the dataset.

        This method iterates over all directories containing DICOM files, applies metadata and 
        image anonymization, and validates the anonymized files. It also provides a debugging mode 
        to check specific DICOM tags.

        Args:
            debug_item (tuple, optional): A tuple containing a DICOM tag and its expected name. 
                                        If provided, the method runs a custom check on the tag 
                                        instead of performing anonymization. Defaults to None.

        Behavior:
            - If `debug_item` is provided, it counts the occurrences of the specified DICOM tag 
            and checks for non-empty values.
            - Otherwise, it anonymizes metadata and image data for each DICOM file, validates the 
            anonymized output, and exports mapping files.

        Processing Steps:
            1. Initializes a progress bar for tracking anonymization progress.
            2. Iterates over all DICOM directories and retrieves patient-specific anonymization actions.
            3. For each DICOM file:
            - If in debug mode, runs custom tag checks.
            - Otherwise:
                - Applies metadata anonymization.
                - Applies image anonymization.
                - Validates the anonymized file.
            4. Exports CSV mappings for patient IDs, UIDs, and output paths.
            5. Logs changes made by the image anonymizer and validator.
            6. If detector logging is enabled, exports logs of detected and whitelisted entities.

        Outputs:
            - Progress tracking for anonymization.
            - CSV files storing ID mappings, anonymization paths, and detected entities.
            - Logs detailing anonymization actions and validations.
        """     
        print(f"Total dicoms to be anonymized: {self.total_dcms}")
        progress_bar = tqdm.tqdm(total=self.total_dcms)
        
        start_from = 0
        count = 0

        total_found = 0
        total_non_empty = 0
        for idx, dir in enumerate(self.dcm_dirs):            
            patient_attrs_action = self.create_patient_attrs_action(dir)
            
            dcms = list_all_files(dir)
            for dcm in dcms:
                if count < start_from:
                    progress_bar.update(1)
                    count += 1
                    continue
                    
                if not debug_item:
                    # if not self.anonymized_file_exists(dcm, dir):
                    _, outfile = self.anonymize_metadata_on_file(dcm, dir, patient_attrs_action)
                    # self.logger.debug(f"{history}")                    
                    self.anonymize_image_data_on_file(outfile, replace=True)
                    # validate the output file and add if missing attributes found
                    self.validate_dicom_file(outfile, dcm, ensure_validation=False)
                else:
                    n_element, n_non_empty = self.run_custom_checks_on_dcm(dcm, debug_item[0], debug_item[1])
                    total_found += n_element
                    total_non_empty += n_non_empty

                progress_bar.update(1)
                count += 1
            
        progress_bar.close()

        if debug_item:
            print(f"Total {total_found} series found with tag {debug_item[1]} {total_non_empty} with non empty value.")
        else:   
            series_output_map = self.get_series_output_path_map()
        
            self.export_csv_from_id_map(self.anonymizer.id_dict, filename="patient_id_mapping")
            self.export_csv_from_id_map(self.anonymizer.uid_dict, filename="uid_mapping")
            self.export_csv_from_id_map(series_output_map, filename='path_mapping', fields=['id_old', 'path'])
            self.logger.info(self.img_anonymizer.change_log)
            self.logger.info(self.validator.added_attr_log)
            
            if self.detector_logging:
                self.export_csv_from_id_map(
                    self.detector.detected_entity_log, 
                    filename='detected_entities', 
                    fields=['entitity', 'count']
                )
                self.export_csv_from_id_map(
                    self.detector.missed_by_whitelist, 
                    filename='whitelisted_entities', 
                    fields=['entitity', 'count']
                )
    
    def get_dcm_path_from_idx(self, target_idx: int):
        """
        Retrieves the file path of a DICOM file at a specified index within the dataset.

        This method iterates through all DICOM directories and files, keeping track of the 
        file count until it reaches the specified index. It is primarily used for debugging 
        to locate a specific DICOM file.

        Args:
            target_idx (int): The index of the desired DICOM file in the dataset.

        Returns:
            str: The file path of the DICOM file at the specified index, or an empty string 
                if the index is out of range.
        """
        progress_bar = tqdm.tqdm(total=target_idx)
        
        count = 0
        target_dcm_path = ''
        
        for idx, dir in enumerate(self.dcm_dirs):                        
            dcms = list_all_files(dir)
            for dcm in dcms:
                if count == target_idx:
                    target_dcm_path = dcm
                    break
                
                progress_bar.update(1)
                count += 1
            
            if target_dcm_path != '':
                break
        
        progress_bar.close()
        
        return target_dcm_path
                