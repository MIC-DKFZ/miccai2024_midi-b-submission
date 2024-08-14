import re
import json
from ast import literal_eval

import pydicom

from dicomanonymizer import simpledicomanonymizer, actions_map_name_functions
from dicomanonymizer.simpledicomanonymizer import (
    replace, empty_or_replace, delete_or_replace,
    delete_or_empty_or_replace, delete_or_empty_or_replace_UID,
    replace_UID, replace_element,
    empty, delete, delete_or_empty,
    keep
)

from dcm_anonymizers.phi_detectors import DcmPHIDetector
from dcm_anonymizers.ps_3_3 import DCMPS33Anonymizer, replace_with_value

TCIA_DEID_ATTRS_JSON = 'dcm_anonymizers/tcia_deid_attrs.json'
SHIFT_DATE_OFFSET = 120

def load_ps3_tags(json_path: str):
    tags = {}    
    with open(json_path) as f:
        tags = json.load(f)
    
    for tag in tags:
        items = tags[tag]
        tags[tag] = [literal_eval(i) for i in items]

    return tags

def keep_element(element):
    pass

def tcia_keep(dataset, tag):
    element = dataset.get(tag)
    return keep_element(element)


class DCMTCIAAnonymizer(DCMPS33Anonymizer):
    def __init__(self, phi_detector: DcmPHIDetector = None):        
        super().__init__(phi_detector)       

        self.logger.debug("TCIA anonymizer init")

        self.tcia_to_ps3_actions_map = {
            'remove': self.tcia_delete,
            'keep': self.tcia_keep,
            'incrementdate': replace,
            'hashuid': replace_UID,
            'time': self.tcia_keep,
            'empty': empty,
            'replace': replace,
            'process': replace,
            'lookup': replace,
            'hashname': replace,
        }
        
        actions_map_name_functions.update({
            "delete": self.tcia_delete,
            "keep": self.tcia_keep,
        })

    def tcia_keep_element(self, element):
        if element.VR in ("LO", "LT", "SH", "PN", "CS", "ST", "UT"):
            if self.detector:
                element.value = self.detector.deidentified_element_val(element)
        else:
            pass

    def tcia_keep(self, dataset, tag):
        element = dataset.get(tag)
        if element is not None:
            self.tcia_keep_element(element)
    
    def tcia_delete_element(self, dataset, element):
        # dont delete if it does not contain any value
        if str(element.value) == "":
            pass
        else:
            simpledicomanonymizer.delete_element(dataset, element)

    def tcia_delete(self, dataset, tag):
        element = dataset.get(tag)
        if element is not None:
            self.tcia_delete_element(dataset, element)


    def custom_init_actions(self):
        """
        Initialize anonymization actions with DICOM standard values

        :return Dict object which map actions to tags
        """

        tcia_tags = load_ps3_tags(json_path=TCIA_DEID_ATTRS_JSON)

        # anonymization_actions = {tag: replace for tag in D_TAGS}
        anonymization_actions = {tag: self.tcia_to_ps3_actions_map['remove'] for tag in tcia_tags['remove']}
        anonymization_actions.update({tag: self.tcia_to_ps3_actions_map['keep'] for tag in tcia_tags['keep']})
        anonymization_actions.update({tag: self.tcia_to_ps3_actions_map['incrementdate'] for tag in tcia_tags['incrementdate']})
        anonymization_actions.update({tag: self.tcia_to_ps3_actions_map['hashuid'] for tag in tcia_tags['hashuid']})
        anonymization_actions.update({tag: self.tcia_to_ps3_actions_map['time'] for tag in tcia_tags['time']})
        anonymization_actions.update({tag: self.tcia_to_ps3_actions_map['empty'] for tag in tcia_tags['empty']})
        anonymization_actions.update({tag: self.tcia_to_ps3_actions_map['replace'] for tag in tcia_tags['replace']})
        anonymization_actions.update({tag: self.tcia_to_ps3_actions_map['process'] for tag in tcia_tags['process']})
        anonymization_actions.update({tag: self.tcia_to_ps3_actions_map['lookup'] for tag in tcia_tags['lookup']})
        anonymization_actions.update({tag: self.tcia_to_ps3_actions_map['hashname'] for tag in tcia_tags['hashname']})
        

        return anonymization_actions
    
    def extract_private_tags(self, dataset: pydicom.Dataset):
        all_texts = ''
        tag_position_map = {}
        id_tags = []

        empty_tags_substr = ["id", "uid", "date"]

        empty_tags_substr = [f"\\b{ts}\\b" for ts in empty_tags_substr]

        empty_tags_pattern = '|'.join(empty_tags_substr)
        empty_tags_pattern = r'{}'.format(empty_tags_pattern)

        for element in dataset:
            if element.VR == 'OW':
                    continue
            if element.tag.is_private:
                if re.search(empty_tags_pattern, str(element.name.lower())):
                    id_tags.append(element.tag)
                elif element.VR in ("LO", "LT", "SH", "PN", "CS", "ST", "UT", "UN") and element.value != "":
                    element_val = self.detector.process_element_val(element)
                    element_name = self.detector.processed_element_name(element.name)
                    element_text = f"{element_name}: {element_val}"
                    start = len(all_texts) + len(element_name) + 1
                    end = start + len(element_name)
                    tag_position_map[element.tag] = (start, end)
                    if all_texts == '':
                        all_texts += f"{element_text}"
                    else:
                        all_texts += f", {element_text}"
        
        return all_texts, tag_position_map, id_tags


    def anonymize_dataset(
        self,
        dataset: pydicom.Dataset,
        extra_anonymization_rules: dict = None,
        delete_private_tags: bool = False,
    ) -> None:
        private_tags_actions = {}
        all_private_texts, text_tag_mapping, id_tags = self.extract_private_tags(dataset)
        for tag in id_tags:
            element = dataset.get(tag)
            private_tags_actions[(element.tag.group, element.tag.element)] = delete

        tags_w_entities = self.detector.detect_enitity_tags_from_text(all_private_texts, text_tag_mapping)
        for tag in tags_w_entities:
            element = dataset.get(tag)
            deid_val = self.detector.deid_element_values_from_entity_values(element.value, tags_w_entities[tag])
            private_tags_actions[(element.tag.group, element.tag.element)] = replace_with_value([deid_val])
        
        extra_anonymization_rules.update(private_tags_actions)

        super().anonymize_dataset(dataset, extra_anonymization_rules, delete_private_tags=False)
        
