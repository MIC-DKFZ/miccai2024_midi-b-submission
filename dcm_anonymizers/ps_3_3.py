import json
from typing import Union
from ast import literal_eval
from decimal import Decimal
from datetime import timedelta

import pydicom
from pydicom.multival import MultiValue
from pydicom.uid import generate_uid

from dicomanonymizer import simpledicomanonymizer
from dicomanonymizer.simpledicomanonymizer import (
    replace, empty_or_replace, delete_or_replace,
    delete_or_empty_or_replace, delete_or_empty_or_replace_UID,
    replace_element_UID, replace_element,
    empty, delete, delete_or_empty
)

from dicomanonymizer.format_tag import tag_to_hex_strings

from dcm_anonymizers.utils import int_tuple_to_basetag, get_hashid, parse_date_string
from dcm_anonymizers.phi_detectors import DcmPHIDetector, DcmRobustPHIDetector


import logging

PS_3_3_ATTRS_JSON = '../dcm_anonymizers/ps3.3_profile_attrs.json'
SHIFT_DATE_OFFSET = 120

def format_action_dict(actions):
    formatted = {}

    for tag in actions.keys():
        formatted[literal_eval(tag)] = actions[tag]

    return formatted

def load_ps3_tags(json_path: str):
    tags = {}    
    with open(json_path) as f:
        tags = json.load(f)
    
    for tag in tags:
        items = tags[tag]
        tags[tag] = [literal_eval(i) for i in items]

    return tags

def replace_with_value(options: Union[list, dict]):
    """
    Replace the given tag with a predefined value.

    :param options: contains one value:
        - value: the string used to replace the tag value
    If options is a list, value is expected to be the first value.
    """

    def apply_replace_with_value(dataset, tag):
        if isinstance(options, dict):
            try:
                value = options["value"]
            except KeyError as e:
                logging.warning(f"Missing field in tag dictionary {tag}: {e.args[0]}")
        else:
            value = options[0]

        element = dataset.get(tag)
        if element is not None:
            element.value = value

    return apply_replace_with_value


class DCMPS33Anonymizer:
    def __init__(self, phi_detector: DcmRobustPHIDetector = None):
        super().__init__()
        self.uid_dict = {}
        self.series_uid_dict = {}
        self.id_dict = {}
        self.history = {}
        self.shift_date_offset = SHIFT_DATE_OFFSET
        self.uid_prefix = '1.2.826.0.1.3680043.8.498.'
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        
        self.detector = phi_detector
        self.ignored_tags = []

        self._override_simpledicomanonymizer()

        self.logger.debug("PS3.3 init")
    
    def _override_simpledicomanonymizer(self):
        simpledicomanonymizer.dictionary = self.uid_dict

        simpledicomanonymizer.replace_date_time_element = self.custom_replace_date_time_element
        simpledicomanonymizer.replace_element = self.custom_replace_element
        simpledicomanonymizer.replace_element_UID = self.replace_element_UID
        simpledicomanonymizer.empty_element = self.custom_empty_element

        simpledicomanonymizer.anonymize_dataset = self.anonymize_dataset


    def shift_date(self, date_string, days=0, hours=0, minutes=0, seconds=0, date_only=True):
        if date_string == '':
            return date_string

        # Parse the date string
        original_date = parse_date_string(date_string)

        # extract the miliseconds digits if provided
        d = Decimal(date_string)
        milisecs_digits = abs(d.as_tuple().exponent)
        
        # Create a timedelta object based on the provided offset values
        offset = timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)
        
        # Shift the date by the offset
        new_date = original_date + offset

        if date_only:
            return new_date.date().strftime("%Y%m%d")
        elif milisecs_digits > 0:        
            return new_date.strftime("%Y%m%d%H%M%S.%f")
        else:
            return new_date.strftime("%Y%m%d%H%M%S")

    
    def get_UID(self, old_uid: str) -> str:
        """
        Lookup new UID in cached dictionary or create new one if none found
        """
        if old_uid not in self.uid_dict:
            self.uid_dict[old_uid] = generate_uid(prefix=self.uid_prefix)
        return self.uid_dict.get(old_uid)


    def replace_element_UID(self, element):
        """
        Replace UID(s) with random UID(s)
        The replaced value is kept in a dictionary link to the initial element.value in order to automatically
        apply the same replaced value if we have an other UID with the same value
        """
        if isinstance(element.value, MultiValue):
            # Example of multi-value UID situation: IrradiationEventUID, (0008,3010)
            for k, v in enumerate(element.value):
                element.value[k] = self.get_UID(v)
        else:
            element.value = self.get_UID(element.value)


    def get_ID(self, old_id: str) -> str:
        """
        Lookup new ID in cached dictionary or create new one if none found
        """
        if old_id not in self.id_dict:
            self.id_dict[old_id] = get_hashid(old_id)
        return self.id_dict.get(old_id)

    def replace_element_id(self, element):
        if element.value != '':
            element.value = self.get_ID(element.value)

    def replace_UID(self, dataset, tag):
        """
        U - replace with a non-zero length UID that is internally consistent within a set of Instances
        Lazy solution : Replace with empty string
        """
        element = dataset.get(tag)
        if element is not None:
            self.replace_element_UID(element)

    def replace_ID(self, dataset, tag):
        """
        D - replace with a non-zero length value that may be a dummy value and consistent with the
        VR
        """
        element = dataset.get(tag)
        if element is not None:
            self.replace_element_id(element)

    def custom_replace_date_time_element(self, element):
        """
        Handle the anonymization of date and time related elements.

        Date and time elements are all handled in the same way, whether they are emptied or removed.
        """
        if element.VR == "DA":
            element.value = self.shift_date(element.value, days=self.shift_date_offset)
        elif element.VR == "DT":
            element.value = self.shift_date(element.value, days=self.shift_date_offset, date_only=False)
        elif element.VR == "TM":
            # Do noting for TIME only
            pass
        else:
            pass

    def custom_replace_element(self, element: pydicom.DataElement):
        """
        Replace element's value according to it's VR:
        - LO, LT, SH, PN, CS, ST, UT: replace with 'Anonymized'
        - UI: cf replace_element_UID
        - DS and IS: value will be replaced by '0'
        - FD, FL, SS, US, SL, UL: value will be replaced by 0
        - DA: value will be replaced by '00010101'
        - DT: value will be replaced by '00010101010101.000000+0000'
        - TM: value will be replaced by '000000.00'
        - UN: value will be replaced by b'Anonymized' (binary string)
        - SQ: call replace_element for all sub elements

        See https://laurelbridge.com/pdf/Dicom-Anonymization-Conformance-Statement.pdf
        """
        if element.VR in ("LO", "LT", "SH", "PN", "CS", "ST", "UT"):
            if self.detector:
                element.value = self.detector.deidentified_element_val(element)
            else:
                self.ignored_tags.append((element.tag, 'replace'))
        elif element.VR == "UI":
            replace_element_UID(element)
        elif element.VR in ("DS", "IS"):            
            element.value = "0"
        elif element.VR in ("FD", "FL", "SS", "US", "SL", "UL"):
            element.value = 0
        elif element.VR in ("DT", "DA", "TM"):
            self.custom_replace_date_time_element(element)
        elif element.VR == "UN":
            element.value = b"Anonymized"
        elif element.VR == "AS":
            # ignore AS AgeString
            pass
        elif element.VR == "SQ":
            for sub_dataset in element.value:
                for sub_element in sub_dataset.elements():
                    if isinstance(sub_element, pydicom.dataelem.RawDataElement):
                        # RawDataElement is a NamedTuple, so cannot set its value attribute.
                        # Convert it to a DataElement, replace value, and set it back.
                        e2 = pydicom.dataelem.DataElement_from_raw(sub_element)
                        self.custom_replace_element(e2)
                        sub_dataset.add(e2)
                    else:
                        self.custom_replace_element(sub_element)
        else:
            self.logger.warning(
                "Element {}={} not anonymized. VR {} not yet implemented.".format(element.name, element.value, element.VR)
            )
        

    def custom_empty_element(self, element):
        """
        Clean element according to the element's VR:
        - SH, PN, UI, LO, LT, CS, AS, ST and UT: value will be set to ''
        - DA: value will be replaced by '00010101'
        - DT: value will be replaced by '00010101010101.000000+0000'
        - TM: value will be replaced by '000000.00'
        - UL, FL, FD, SL, SS and US: value will be replaced by 0
        - DS and IS: value will be replaced by '0'
        - UN: value will be replaced by: b'' (binary string)
        - SQ: all subelement will be called with "empty_element"

        Date and time related VRs are not emptied by replacing their values with a empty string to keep
        the consistency with some software who expect a non null value for those VRs.

        See: https://laurelbridge.com/pdf/Dicom-Anonymization-Conformance-Statement.pdf
        """
        if element.VR in ("SH", "PN", "UI", "LO", "LT", "CS", "ST", "UT"):
            # print(element.name, element.VR, element.value)
            if self.detector:
                entities = self.detector.detect_entities_from_element(element)
                if len(entities) > 0:
                    element.value = ""
            else:
                # self.ignored_tags.append((element.tag, 'empty'))
                element.value = ""
        elif element.VR in ("DT", "DA", "TM"):
            # self.custom_replace_date_time_element(element)
            element.value = ""
        elif element.VR in ("UL", "FL", "FD", "SL", "SS", "US"):
            element.value = 0
        elif element.VR in ("DS", "IS"):
            element.value = "0"
        elif element.VR == "UN":
            element.value = b""
        elif element.VR == "AS":
            # ignore AS AgeString
            pass
        elif element.VR == "SQ":
            for sub_dataset in element.value:
                for sub_element in sub_dataset.elements():
                    self.custom_empty_element(sub_element)
        else:
            self.logger.warning(
                "Element {}={} not anonymized. VR {} not yet implemented.".format(element.name, element.value, element.VR)
            )


    def custom_init_actions(self):
        """
        Initialize anonymization actions with DICOM standard values

        :return Dict object which map actions to tags
        """

        ps3_tags = load_ps3_tags(json_path=PS_3_3_ATTRS_JSON)

        # anonymization_actions = {tag: replace for tag in D_TAGS}
        anonymization_actions = {tag: self.replace_UID for tag in ps3_tags['UID_TAGS']}
        anonymization_actions.update({tag: replace for tag in ps3_tags['DATES_TAGS']})
        anonymization_actions.update({tag: self.replace_ID for tag in ps3_tags['ID_TAGS']})
        anonymization_actions.update({tag: empty for tag in ps3_tags['Z_TAGS']})
        anonymization_actions.update({tag: delete for tag in ps3_tags['X_TAGS']})
        anonymization_actions.update({tag: self.replace_UID for tag in ps3_tags['U_TAGS']})
        anonymization_actions.update({tag: replace for tag in ps3_tags['D_TAGS']})
        anonymization_actions.update({tag: delete_or_empty for tag in ps3_tags['X_Z_TAGS']})
        anonymization_actions.update({tag: delete_or_replace for tag in ps3_tags['X_D_TAGS']})
        anonymization_actions.update({tag: empty_or_replace for tag in ps3_tags['Z_D_TAGS']})
        anonymization_actions.update(
            {tag: delete_or_empty_or_replace for tag in ps3_tags['X_Z_D_TAGS']}
        )
        anonymization_actions.update(
            {tag: delete_or_empty_or_replace_UID for tag in ps3_tags['X_Z_U_STAR_TAGS']}
        )
        return anonymization_actions


    def extract_sequence_element_datasets(self, dataset: pydicom.Dataset):
        sequence_datasets: list = []
        for elem in dataset:
            if elem.VR == 'SQ':
                elem_dataset = elem.value[0]
                sequence_datasets.append(elem_dataset)
                # sequence_datasets = self.extract_sequence_element_datasets(elem_dataset, sequence_datasets)
        return sequence_datasets


    def anonymize_dataset(
        self,
        dataset: pydicom.Dataset,
        extra_anonymization_rules: dict = None,
        delete_private_tags: bool = True,
    ) -> None:
        """
        Anonymize a pydicom Dataset by using anonymization rules which links an action to a tag

        :param dataset: Dataset to be anonymize
        :param extra_anonymization_rules: Rules to be applied on the dataset
        :param delete_private_tags: Define if private tags should be delete or not
        """
        current_anonymization_actions = self.custom_init_actions()

        if extra_anonymization_rules is not None:
            current_anonymization_actions.update(extra_anonymization_rules)
        
        if delete_private_tags:
            private_tags_actions = {}
            
            # Iterate through the data elements and check for private tags
            for element in dataset:
                # A tag is a tuple of (group, element)
                group, _ = element.tag.group, element.tag.element
                if group % 2 != 0 and (element.name.startswith('[') and element.name.endswith("]")):
                    private_tags_actions[(element.tag.group, element.tag.element)] = delete_or_replace
                    # print(f"Private Tag Found: {element.tag} - {element.name}, Private: {element.tag.is_private}")

            if private_tags_actions:
                current_anonymization_actions.update(private_tags_actions)

        action_history = {}
        private_tags = []
        sequences = []

        def walk_sequences(dataset):
            for elem in dataset:
                if elem.VR == 'SQ':            
                    for i, item in enumerate(elem.value):
                        sequences.append(item)
                        walk_sequences(item) 

        walk_sequences(dataset)   

        for tag, action in current_anonymization_actions.items():
            # check current tag already exists in the 
            # processed tags history
            basetag = int_tuple_to_basetag(tag)
            if basetag in self.history:
                continue

            def range_callback(dataset, data_element):
                if (
                    data_element.tag.group & tag[2] == tag[0]
                    and data_element.tag.element & tag[3] == tag[1]
                ):   
                    action(dataset, (data_element.tag.group, data_element.tag.element))

            element = None
            # We are in a repeating group
            if len(tag) > 2:
                dataset.walk(range_callback)
            # Individual Tags
            else:
                try:
                    element = dataset.get(tag)
                    if element:
                        earliervalue = element.value
                except KeyError:
                    logging.warning("Cannot get element from tag: ", tag_to_hex_strings(tag))

                if tag[0] == 0x0002:
                    if not hasattr(dataset, "file_meta"):
                        continue
                    # Apply rule to meta information header
                    action(dataset.file_meta, tag)
                else:
                    action(dataset, tag)

                # also check if the tag exists inside
                # the sequences.
                for s in sequences:
                    if tag in s:
                        action(s, tag)                                                 
                
                if element:
                    # check if Series Instance UID, then store the id in series id map
                    if tag == (0x0020, 0x000E):
                        if earliervalue in self.series_uid_dict:
                            earlieruid = self.series_uid_dict.get(earliervalue)
                            assert earlieruid == element.value
                        else:
                            self.series_uid_dict[earliervalue] = element.value

                    if earliervalue != element.value:
                        action_history[element.tag] = action.__name__

                # Get private tag to restore it later
                # check if the element is already not deleted
                if element and (action != delete and action.__name__ != 'tcia_delete') and element.tag.is_private:                                   
                    private_tags.append(simpledicomanonymizer.get_private_tag(dataset, tag))



        self.history = action_history

        # X - Private tags = (0xgggg, 0xeeee) where 0xgggg is odd
        if delete_private_tags:
            dataset.remove_private_tags()

            # Adding back private tags if specified in dictionary
            for privateTag in private_tags:
                creator = privateTag["creator"]
                element = privateTag["element"]
                block = dataset.private_block(
                    creator["tagGroup"], creator["creatorName"], create=True
                )
                if element is not None:
                    block.add_new(
                        element["offset"], element["element"].VR, element["element"].value
                    )
        
    
    def anonymize(self, input_path: str, output_path: str, opt_history: dict = {}, custom_actions: dict = {}):
        self.history = opt_history

        simpledicomanonymizer.anonymize_dicom_file(
            in_file=str(input_path),
            out_file=str(output_path),
            extra_anonymization_rules=custom_actions,
            delete_private_tags=True,
        )

        return self.history, self.ignored_tags