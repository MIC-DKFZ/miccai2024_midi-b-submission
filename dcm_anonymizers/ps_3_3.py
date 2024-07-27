import hashlib
import base64
from typing import List, Union
from ast import literal_eval
from datetime import datetime, timedelta
import pydicom
from pydicom.tag import BaseTag
from pydicom.multival import MultiValue
import json
import numpy as np
from transformers import pipeline

from dicomanonymizer import anonymize, simpledicomanonymizer
from dicomanonymizer.simpledicomanonymizer import (
    replace, empty_or_replace, delete_or_replace,
    delete_or_empty_or_replace, delete_or_empty_or_replace_UID,
    replace_element_UID, replace_element,
    empty, delete, delete_or_empty
)

from dicomanonymizer.format_tag import tag_to_hex_strings

PS_3_3_ATTRS_JSON = './docs/ps3.3_profile_attrs.json'
SHIFT_DATE_OFFSET = 120


def int_tuple_to_basetag(tag: tuple):
    # Combine the group and element into a single integer
    combined_tag = (tag[0] << 16) + tag[1]
    return BaseTag(combined_tag)


def load_ps3_tags(json_path: str):
    tags = {}    
    with open(json_path) as f:
        tags = json.load(f)
    
    for tag in tags:
        items = tags[tag]
        tags[tag] = [literal_eval(i) for i in items]

    return tags

def get_hashid(key: str, method: str = 'sha256', nchar: int =16):
    """
    Generate a hash identifier.
    :param key: input string for the hash algorithm.
    :param method: an algorithm name. Select from hashlib.algorithms_guaranteed 
    or hashlib.algorithms_available.
    :param nchar: number of the first character to return. Set 0 for all characters.
    :return: a string.
    """
    h = hashlib.new(method)
    h.update(key.encode())

    # for shake, the digest size is variable in length, let's just put it 32 bytes
    if h.digest_size == 0:
        hash_id = base64.b32encode(h.digest(32)).decode().replace("=", "")
    else:
        hash_id = base64.b32encode(h.digest()).decode().replace("=", "")

    if nchar > 0:
        hash_id = hash_id[:nchar]

    return hash_id

def parse_date_string(date_string):
        # Define possible formats
        date_formats = [
            "%Y%m%d%H%M%S",  # Full format with hours, minutes, and seconds
            "%Y%m%d"         # Format with only date
        ]
        
        # Try to parse the date string using the appropriate format
        for date_format in date_formats:
            try:
                return datetime.strptime(date_string, date_format)
            except ValueError:
                continue
        
        # If no format matches, raise an error
        raise ValueError(f"Date string '{date_string}' does not match any known format")


class DcmPHIDetector:
    def __init__(self) -> None:
        self.model_name = "obi/deid_roberta_i2b2"
        self.model = None
        self.min_confidence = np.float32(0.6)

        self._init_model()
    
    def _init_model(self):
        self.model = pipeline("ner", model=self.model_name)

    def entity_name(self, entityvalue: str):
        return entityvalue[2:]

    def process_enitity_val(self, entityvalue: str):
        spacechar = 'Ġ'
        if entityvalue[0] == spacechar:
            entityvalue = entityvalue[1:]
        entityvalue = entityvalue.replace(spacechar, ' ')
        return entityvalue

    def process_element_val(self, element):
        elementval = ""
        if str(element.value).strip() == "":
            return elementval
        
        if element.VM > 1:
            elementval = ', '.join([str(item) for item in element.value])
        elif element.VM == 1:
            elementval = str(element.repval)
        elementval = elementval.replace("'", '')
        if element.VR == 'PN':
            elementval = elementval.replace("^", ' ')
        return elementval.strip()
    
    def filter_outputs(self, outputs: list):
        filtered = [o for o in outputs if o['score'] > self.min_confidence]
        return filtered
    
    def process_outputs(self, outputs: list):
        entities = []
        entitytype = None
        entitystart = -1
        temp = ""
        for idx, item in enumerate(outputs):
            if idx == 0:
                temp = item['word']
                entitytype = self.entity_name(item['entity'])
                entitystart = item['start']
                continue
            previtem = outputs[idx-1]
            currententity = self.entity_name(item['entity'])
            if (item['index'] == previtem['index'] + 1) and (currententity == entitytype):
                temp += item['word']
            else:
                entities.append((self.process_enitity_val(temp), entitytype, entitystart))
                temp = item['word']
                entitytype = self.entity_name(item['entity'])
                entitystart = item['start']

        if temp != "":
            entities.append((self.process_enitity_val(temp), entitytype, entitystart))
        return entities

    def detect_entities(self, text: str):
        if text == "":
            return []
        
        # dcmnote = f"{element.name}: {self.process_element_val(element)}"
        outputs = self.model(text)
        outputs = self.filter_outputs(outputs)
        entities = self.process_outputs(outputs)
        # for e in entities:
        #     print(f"{e[1]}: {self.process_enitity_val(e[0])}")
        return entities
    
    def deidentified_element_val(self, element: pydicom.DataElement) -> Union[str, list[str]]:
        if str(element.value).strip() == "":
            return ""

        element_text = f"{element.name}: {self.process_element_val(element)}"
        entities = self.detect_entities(element_text)
        if len(entities) == 0:
            return element.value
        
        processed = element_text[:]
        for e in entities:
            target_substr = element_text[e[2]: (e[2]+len(e[0]))]
            processed = processed.replace(target_substr, '')
        
        deidentified_val = processed.replace(f"{element.name}: ", '')

        if element.VM > 1:
            return deidentified_val.split(', ')
        
        return deidentified_val.strip()



class DCMPS33Anonymizer:
    def __init__(self, phi_detector: DcmPHIDetector = None):
        super().__init__()
        self.uid_dict = {}
        self.id_dict = {}
        self.history = {}

        self.detector = phi_detector

        self._override_simpledicomanonymizer()

    
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
        
        # Create a timedelta object based on the provided offset values
        offset = timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)
        
        # Shift the date by the offset
        new_date = original_date + offset

        if date_only:
            return new_date.date().strftime("%Y%m%d")
        
        return new_date.strftime("%Y%m%d%H%M%S")
    
    def get_UID(self, old_uid: str) -> str:
        """
        Lookup new UID in cached dictionary or create new one if none found
        """
        from pydicom.uid import generate_uid

        if old_uid not in self.uid_dict:
            self.uid_dict[old_uid] = generate_uid(None)
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
        # print(element.name, element.VR, element.value)
        if element.VR == "DA":
            element.value = self.shift_date(element.value, days=SHIFT_DATE_OFFSET)
        elif element.VR == "DT":
            element.value = self.shift_date(element.value, days=SHIFT_DATE_OFFSET, date_only=False)
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
            # print(element.name, element.VR, element.value)
            # element.value = "ANONYMIZED"  # CS VR accepts only uppercase characters
            if self.detector:
                element.value = self.detector.deidentified_element_val(element)
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
        elif element.VR == "SQ":
            for sub_dataset in element.value:
                for sub_element in sub_dataset.elements():
                    if isinstance(sub_element, pydicom.dataelem.RawDataElement):
                        # RawDataElement is a NamedTuple, so cannot set its value attribute.
                        # Convert it to a DataElement, replace value, and set it back.
                        # Found in https://github.com/KitwareMedical/dicom-anonymizer/issues/63
                        e2 = pydicom.dataelem.DataElement_from_raw(sub_element)
                        replace_element(e2)
                        sub_dataset.add(e2)
                    else:
                        replace_element(sub_element)
        else:
            raise NotImplementedError(
                "Not anonymized. VR {} not yet implemented.".format(element.VR)
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
        if element.VR in ("SH", "PN", "UI", "LO", "LT", "CS", "AS", "ST", "UT"):
            # print(element.name, element.VR, element.value)
            if self.detector:
                element.value = self.detector.deidentified_element_val(element)
            # pass
        elif element.VR in ("DT", "DA", "TM"):
            self.custom_replace_date_time_element(element)
        elif element.VR in ("UL", "FL", "FD", "SL", "SS", "US"):
            element.value = 0
        elif element.VR in ("DS", "IS"):
            element.value = "0"
        elif element.VR == "UN":
            element.value = b""
        elif element.VR == "SQ":
            for sub_dataset in element.value:
                for sub_element in sub_dataset.elements():
                    self.custom_empty_element(sub_element)
        else:
            raise NotImplementedError(
                "Not anonymized. VR {} not yet implemented.".format(element.VR)
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
        
        private_tags = []

        action_history = {}

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
                    print("Cannot get element from tag: ", tag_to_hex_strings(tag))

                if tag[0] == 0x0002:
                    if not hasattr(dataset, "file_meta"):
                        continue
                    # Apply rule to meta information header
                    action(dataset.file_meta, tag)
                else:
                    action(dataset, tag)            
                
                if element:
                    if earliervalue != element.value:
                        action_history[element.tag] = action.__name__

                # Get private tag to restore it later
                # if element and element.tag.is_private:
                #    private_tags.append(get_private_tag(dataset, tag))

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
        
    
    def anonymize(self, input_path: str, output_path: str, opt_history: dict = {}):
        self.history = opt_history

        anonymize(
            input_path=str(input_path),
            output_path=str(output_path),
            anonymization_actions={},
            delete_private_tags=False,
        )

        return self.uid_dict, self.id_dict, self.history