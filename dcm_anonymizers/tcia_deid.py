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

from dcm_anonymizers.phi_detectors import DcmPHIDetector, DcmRobustPHIDetector
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

def replace_element_value(dataset, tag, value):
    element = dataset.get(tag)
    if element is not None:
        element.value = value

class DCMTCIAAnonymizer(DCMPS33Anonymizer):
    def __init__(
            self, 
            phi_detector: DcmRobustPHIDetector = None,
            # Phi detector detects from notes
            notes_phi_detector: DcmRobustPHIDetector = None,
            rules_json_path: str = TCIA_DEID_ATTRS_JSON,
            apply_custom_actions: bool = True,
        ):        
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
        self.notes_phi_detector = notes_phi_detector
        self.tag_seperator = ',\n'
        self.rules_json_path = rules_json_path
        self.apply_custom_actions = apply_custom_actions

        self.custom_actions = {
            "(0x0008, 0x2111)": self.tcia_to_ps3_actions_map['replace'],    # Derivation Description
            "(0x0010, 0x2180)": self.tcia_to_ps3_actions_map['keep'],       # Occupation
            "(0x0012, 0x0010)": self.tcia_to_ps3_actions_map['remove'], 	# Clinical Trial Sponsor Name	
            "(0x0012, 0x0020)": self.tcia_to_ps3_actions_map['remove'],	    # Clinical Trial Protocol ID	
            "(0x0012, 0x0021)": self.tcia_to_ps3_actions_map['remove'],	    # Clinical Trial Protocol Name	
            "(0x0012, 0x0030)": self.tcia_to_ps3_actions_map['remove'],	    # Clinical Trial Site ID	
            "(0x0012, 0x0031)": self.tcia_to_ps3_actions_map['remove'],	    # Clinical Trial Site Name	
            "(0x0012, 0x0040)": self.tcia_to_ps3_actions_map['remove'],	    # Clinical Trial Subject ID	
            "(0x0012, 0x0042)": self.tcia_to_ps3_actions_map['remove'],	    # Clinical Trial Subject Reading ID	
            "(0x0012, 0x0051)": self.tcia_to_ps3_actions_map['remove'],	    # Clinical Trial Time Point Description	
            "(0x0012, 0x0060)": self.tcia_to_ps3_actions_map['remove'],	    # Clinical Trial Coordinating Center Name	
            "(0x0012, 0x0071)": self.tcia_to_ps3_actions_map['remove'],	    # Clinical Trial Series ID	
            "(0x0012, 0x0072)": self.tcia_to_ps3_actions_map['remove'],	    # Clinical Trial Series Description	
            "(0x0012, 0x0081)": self.tcia_to_ps3_actions_map['remove'],	    # Clinical Trial Protocol Ethics Committee Name	
            "(0x0012, 0x0082)": self.tcia_to_ps3_actions_map['remove'],	    # Clinical Trial Protocol Ethics Committee Approval Number	
            "(0x0012, 0x0083)": self.tcia_to_ps3_actions_map['remove'],	    # Consent for Clinical Trial Use Sequence	
            "(0x0010, 0x4000)": self.tcia_to_ps3_actions_map['replace'],    # Patient Comments
            "(0x0040, 0x0009)": self.tcia_to_ps3_actions_map['keep'],       # Scheduled Procedure Step ID
            "(0x0020, 0x4000)": self.tcia_to_ps3_actions_map['replace'],    # Image Comments            
            "(0x0018, 0x700A)": self.tcia_to_ps3_actions_map['empty'] ,     # Detector ID
            "(0x0018, 0x700C)": self.tcia_to_ps3_actions_map['empty'],      # Date of Last Detector Calibration
            "(0x0018, 0x1200)": self.tcia_to_ps3_actions_map['replace'] ,   # Date of Last Calibration
            "(0x0018, 0x1201)": self.tcia_to_ps3_actions_map['keep'] ,      # Time of Last Calibration
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
                self.ignored_tags.append((element.tag, 'keep'))
        else:
            pass

    def tcia_keep(self, dataset, tag):
        element = dataset.get(tag)
        if element is not None:
            self.tcia_keep_element(element)
    
    def tcia_delete_element(self, dataset, element):
        # dont delete if it does not contain any value
        # if str(element.value) == "":
        #     pass
        # handle tags of DA, since simpledicomanonymizer replace this 
        # with 00010101
        if element.VR == "DA":
            element.value = ''
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

        tcia_tags = load_ps3_tags(json_path=self.rules_json_path)

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
    
    def is_utf8(self, element):
        elementval = element.value
        if isinstance(elementval, str):
            return True
        
        try:
            _ = elementval.decode('utf-8')
            return True
        except UnicodeError:
            return False
        
    def tags_to_note(self, tags: list, dataset: pydicom.Dataset):
        note = ''
        tag_note_map = {}

        for tag in tags:
            element = dataset.get(tag)
            if not element:
                continue

            if element.VR in ("LO", "LT", "SH", "PN", "CS", "ST", "UT", "UN") and element.value != "":
                if element.VR == "UN" and not self.is_utf8(element):
                    continue
                
                element_val = DcmRobustPHIDetector.process_element_val(element)
                element_name = DcmRobustPHIDetector.processed_element_name(element.name)
                element_text = DcmRobustPHIDetector.element_to_text(element)

                # prev text + element_name + colon
                start = len(note) + len(element_name) + 1
                # if not the first item, then additional 2 chars for comma+space
                if note != '':
                    start += len(self.tag_seperator)
                    
                end = start + len(element_val) + 1
                
                tag_note_map[element.tag] = (start, end)                    
                if note == '':
                    note += f"{element_text}"
                else:
                    note += f"{self.tag_seperator}{element_text}"

                assert note[start:end] == f" {element_val}", f"{note[start:end]} != {element_val}"
        
        return note, tag_note_map

    
    def extract_private_tags(self, dataset: pydicom.Dataset):
        tags_to_empty = []
        tags_to_replace = []

        empty_tags_substr = ["id", "uid", "date"]

        empty_tags_substr = [f"\\b{ts}\\b" for ts in empty_tags_substr]

        empty_tags_pattern = '|'.join(empty_tags_substr)
        empty_tags_pattern = r'{}'.format(empty_tags_pattern)

        for element in dataset:
            if element.VR == 'OW':
                    continue
            if element.tag.is_private:
                if re.search(empty_tags_pattern, str(element.name.lower())):
                    tags_to_empty.append(element.tag)
                elif element.VR in ("LO", "LT", "SH", "PN", "CS", "ST", "UT", "UN") and element.value != "":
                    if element.VR == "UN" and not self.is_utf8(element):
                        continue
                    tags_to_replace.append(element.tag)
                    
        
        return tags_to_empty, tags_to_replace
    

    def deidentify_all_ignored_tags_as_note(self, dataset: pydicom.Dataset):
        tags_action_dict = dict((tag, action) for tag, action in self.ignored_tags)
        tags_list = list(tags_action_dict.keys())
        note, tags_note_map = self.tags_to_note(tags_list, dataset)
        tags_w_entities = self.notes_phi_detector.detect_enitity_tags_from_text(note, tags_note_map)
        for tag in tags_w_entities:
            element = dataset.get(tag)
            deid_val = self.notes_phi_detector.deid_element_from_entity_values(element, tags_w_entities[tag])
            if element.value != deid_val:
                if tags_action_dict[element.tag] == 'empty':
                    element.value = ""
                else:
                    element.value = deid_val

                self.history[element.tag] = tags_action_dict[element.tag]
        
        self.ignored_tags = []

    def get_custom_actions(self):
        custom_actions = {}

        for tag in self.custom_actions.keys():
            taghex = literal_eval(tag)
            custom_actions[taghex] = self.custom_actions[tag]

        return custom_actions

    def empty_remaining_pn_vr(self, dataset: pydicom.Dataset):
        for element in dataset:
            if element.VR == 'OW':
                    continue
            if element.VR == 'PN':
                if element.tag not in self.history:
                    element.value = ''
                    self.history[element.tag] = empty.__name__
        
        return



    def anonymize_dataset(
        self,
        dataset: pydicom.Dataset,
        extra_anonymization_rules: dict = None,
        delete_private_tags: bool = False,
    ) -> None:
        private_tags_actions = {}

        if self.apply_custom_actions:
            custom_actions = self.get_custom_actions()
            private_tags_actions.update(custom_actions)

        tags_to_delete, tags_to_replce = self.extract_private_tags(dataset)
        for tag in tags_to_delete:
            element = dataset.get(tag)
            private_tags_actions[(element.tag.group, element.tag.element)] = self.tcia_delete

        if self.detector:
            all_private_texts, text_tag_mapping = self.tags_to_note(tags_to_replce, dataset)
            tags_w_entities = self.detector.detect_enitity_tags_from_text(all_private_texts, text_tag_mapping)
            for tag in tags_w_entities:
                element = dataset.get(tag)
                deid_val = self.detector.deid_element_from_entity_values(element, tags_w_entities[tag])
                private_tags_actions[(element.tag.group, element.tag.element)] = replace_with_value([deid_val])
        else:
            for tag in tags_to_replce:
                self.ignored_tags.append((tag, 'replace'))
        
        extra_anonymization_rules.update(private_tags_actions)

        super().anonymize_dataset(dataset, extra_anonymization_rules, delete_private_tags=False)

        if self.notes_phi_detector:
            self.deidentify_all_ignored_tags_as_note(dataset)
        
        if self.apply_custom_actions:
            self.empty_remaining_pn_vr(dataset)
            

        
