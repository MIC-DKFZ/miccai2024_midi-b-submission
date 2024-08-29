import re
import json
from ast import literal_eval

import pydicom
from pydicom import Sequence

from dicomanonymizer import simpledicomanonymizer, actions_map_name_functions
from dicomanonymizer.simpledicomanonymizer import (
    replace, empty_or_replace, delete_or_replace,
    delete_or_empty_or_replace, delete_or_empty_or_replace_UID,
    replace_UID, replace_element,
    empty, delete, delete_or_empty,
    keep
)
import pydicom.tag

from dcm_anonymizers.phi_detectors import DcmPHIDetector, DcmRobustPHIDetector
from dcm_anonymizers.ps_3_3 import DCMPS33Anonymizer, replace_with_value
from dcm_anonymizers.utils import parse_date_string
from dcm_anonymizers.private_tags_extractor import PrivateTagsExtractorV2

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

def count_words(s):
    # Use a regular expression to split the string on spaces or separator characters
    words = re.split(r'[ \-_^\n]+', s)
    
    # Filter out empty strings that might occur due to consecutive separators
    words = [word for word in words if word]
    
    return len(words)

class DCMTCIAAnonymizer(DCMPS33Anonymizer):
    def __init__(
            self, 
            phi_detector: DcmRobustPHIDetector = None,
            # Phi detector detects from notes
            notes_phi_detector: DcmRobustPHIDetector = None,
            rules_json_path: str = TCIA_DEID_ATTRS_JSON,
            apply_custom_actions: bool = True,
            soft_detection: bool = True,
            private_tags_extractor: PrivateTagsExtractorV2 = None,
        ):        
        super().__init__(phi_detector)       

        self.logger.debug("TCIA anonymizer init")

        self.tcia_to_ps3_actions_map = {
            'remove': self.tcia_delete,
            'keep': self.tcia_keep,
            'incrementdate': replace,
            'incrementdateforced': self.tcia_replace_date_time,
            'hashuid': self.replace_UID,
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
        self.soft_detection = soft_detection
        self.private_tags_extractor = private_tags_extractor

        self.disposition_val_to_tcia_actions = {
            'k': 'keep',
            'd': 'remove',
            'h': 'hashuid',
            'o': 'incrementdate',
            'oi': 'incrementdateforced',
        }

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
            "(0x0020, 0x0011)": self.tcia_to_ps3_actions_map['replace'] ,   # Series Number
            "(0x0018, 0x1000)": self.tcia_to_ps3_actions_map['keep'],       # Device Serial Number
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
    
    def tcia_replace_date_time(self, dataset, tag):
        element = dataset.get(tag)
        if element is not None:
            if element.VR in ('DA', 'DT', 'TM'):
                self.custom_replace_date_time_element(element)
            else:
                parsed = False
                try:
                    parsed_date, _ = parse_date_string(str(element.value))
                    parsed = True
                except Exception as e:
                    self.logger.warning(f"Could not able to parse as date {element.value}")

                if parsed:
                    newval = self.shift_date(str(element.value), days=self.shift_date_offset)
                    if element.VR in ('SL', 'UL', 'US', 'SS'):
                        element.value = int(newval)
                    else:
                        element.value = newval
    
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
    
    def action_empty_pn(self, element:pydicom.DataElement):
        if element.VR == 'OW':
            pass
        elif element.VR == 'PN':
            if element.tag not in self.history:
                element.value = ''
                self.history[element.tag] = empty.__name__
        elif element.VR == 'SQ':
            for sub_dataset in element.value:
                for sub_element in sub_dataset.elements():
                    if isinstance(sub_element, pydicom.dataelem.RawDataElement):
                        # RawDataElement is a NamedTuple, so cannot set its value attribute.
                        # Convert it to a DataElement, replace value, and set it back.
                        e2 = pydicom.dataelem.DataElement_from_raw(sub_element)
                        self.action_empty_pn(e2)
                        sub_dataset.add(e2)
                    else:
                        self.action_empty_pn(sub_element)


    def empty_remaining_pn_vr(self, dataset: pydicom.Dataset):
        for element in dataset:
            self.action_empty_pn(element)
    
    def check_and_replace_dates(self, dataset: pydicom.Dataset):
        accept_vr_only = ("DS", "IS")

        for tag_tuple in self.ignored_tags:
            tag =  tag_tuple[0]
            element = dataset.get(tag)

            # skip incase tags already deleted          
            if element is None:
                continue
            elif element.VR in accept_vr_only and element.VM == 1:
                parsed = False
                try:
                    parsed_date, _ = parse_date_string(str(element.value))
                    parsed = True
                except Exception as e:
                    continue
                
                # replace date if found
                if parsed:
                    element.value = "0"

    
    def apply_soft_detections(self, dataset: pydicom.Dataset):
        accept_vr_only = ("LO", "ST", "LT")
        ignore_tags_name = ["Manufacturer"]
        filtered_tags = []

        for tag_tuple in self.ignored_tags:
            tag =  tag_tuple[0]
            element = dataset.get(tag)
            # skip incase tags already deleted          
            if element is None:
                continue
            if element.VR in accept_vr_only and element.name not in ignore_tags_name and element.VM == 1:
                n_words = count_words(element.value)
                if n_words > 1:
                    filtered_tags.append(tag)                
        
        all_texts, text_tag_mapping = self.tags_to_note(filtered_tags, dataset)
        
        if len(all_texts) > 0:
            tags_w_entities = self.notes_phi_detector.detect_enitity_tags_from_text(all_texts, text_tag_mapping)
            for tag in tags_w_entities:
                element = dataset.get(tag)
                deid_val = self.notes_phi_detector.deid_element_from_entity_values(element, tags_w_entities[tag])

                # if changed do the processing
                if deid_val != element.value:
                    # replace leftover `for` after removing names at the end
                    if deid_val.lower().endswith(" for"):
                        deid_val = deid_val[:-3].strip()                    
                    # replace if leftover Dr. substring after deidentification
                    deid_val = re.sub(r'(?i)\bdr\b\.?', '', deid_val)
                    # replace multiple whitespaces created by deidentification
                    deid_val = re.sub(r'\s+', ' ', deid_val)

                    deid_val = deid_val.strip()
                            
                    element.value = deid_val

                self.history[element.tag] = replace.__name__
        return
    
    @staticmethod
    def extract_private_groups_n_creators(dataset):
        creators = []
        groups = []
        for element in dataset:
            if element.VR == 'OW':
                continue
            if element.tag.is_private:
                groups.append(f"{element.tag.group:04x}")
                if element.name == 'Private Creator' and element.value not in creators:                
                    creators.append(element.value)

        groups = list(set(groups))

        return groups, creators

    # def get_private_tags_anonymize_actions(self, dataset:pydicom.Dataset):
    #     groups, creators = DCMTCIAAnonymizer.extract_private_groups_n_creators(dataset)
    #     df_filtered_by_active_groups = self.private_tags_extractor.filter_by_tag_group(groups)
        
    #     self.private_tags_extractor.filtered_private_tag_df = df_filtered_by_active_groups

    #     private_tags_actions = {}

    #     disposition_val_to_tcia_actions = {
    #         'k': 'keep',
    #         'd': 'remove',
    #         'h': 'hashuid',
    #         'o': 'incrementdate',
    #         'oi': 'incrementdateforced',
    #     }

    #     for element in dataset:
    #         if element.VR == 'OW':
    #             continue
    #         if element.tag.is_private:
    #             # tag_string = f"(0x{element.tag.group:04x}, 0x{element.tag.element:04x})"

    #             all_patterns = self.private_tags_extractor.search_patterns_from_element(element, creators)        
    #             filtered = self.private_tags_extractor.get_filtered_rows_from_patterns(all_patterns, element)

    #             # For debugging, can be deleted later
    #             if len(filtered) == 0 and element.name != 'Private Creator':
    #                 self.logger.debug(f"{element.name} {element.tag} {element.VR} {str(element.value)}\nNo rules can be extracted for the above private tag.")

    #             disposition_val = self.private_tags_extractor.get_private_disposition_from_rows(filtered, element)
    #             if disposition_val in disposition_val_to_tcia_actions.keys():
    #                 private_tags_actions[element.tag] = self.tcia_to_ps3_actions_map[disposition_val_to_tcia_actions[disposition_val]]
    #             else:
    #                 private_tags_actions[element.tag] = None

    #             # if element.VR == 'SQ':
    #             #     self.logger.debug(f"Sequence Private Tag found {str(element)}")
        
    #     self.private_tags_extractor.filtered_private_tag_df = None

    #     return private_tags_actions

    def walk_and_anonymize_private_dataset(self, dataset, parent_elements=[], is_root=True):
        private_creator_block_name = None
        for elem in dataset:
            tag = elem.tag
            value = elem.value
            name = elem.name

            # Check if the root element is private
            if is_root:
                if not elem.tag.is_private:
                    continue

            if name == "Private Creator":
                private_creator_block_name = value
            elif len(parent_elements) > 0:
                immidiate_parent = parent_elements[-1]
                private_creator_block_name = immidiate_parent[1]
            
            
            # Process the element
            if isinstance(value, Sequence):
                # If the value is a Sequence, recursively traverse each Dataset in the Sequence
                updated_parent_elements = parent_elements.copy()
                updated_parent_elements.append((elem, private_creator_block_name))
                for i, item in enumerate(value):
                    self.walk_and_anonymize_private_dataset(item, parent_elements=updated_parent_elements, is_root=False)
            else:
                # process the data element
                block_tag = self.private_tags_extractor.get_element_block_tag_with_parents(elem, private_creator_block_name, parent_elements)
                private_disposition = self.private_tags_extractor.get_dispoistion_val_from_block_tag(block_tag, elem)
                action = self.tcia_to_ps3_actions_map[self.disposition_val_to_tcia_actions[private_disposition]]
                if action is not None:
                    action(dataset, tag)
                    self.history[tag] = action.__name__



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

        # tags_to_delete, tags_to_replce = self.extract_private_tags(dataset)
        # for tag in tags_to_delete:
        #     element = dataset.get(tag)
        #     private_tags_actions[(element.tag.group, element.tag.element)] = self.tcia_delete

        # if self.detector:
        #     all_private_texts, text_tag_mapping = self.tags_to_note(tags_to_replce, dataset)
        #     tags_w_entities = self.detector.detect_enitity_tags_from_text(all_private_texts, text_tag_mapping)
        #     for tag in tags_w_entities:
        #         element = dataset.get(tag)
        #         deid_val = self.detector.deid_element_from_entity_values(element, tags_w_entities[tag])
        #         private_tags_actions[(element.tag.group, element.tag.element)] = replace_with_value([deid_val])
        # else:
        #     for tag in tags_to_replce:
        #         self.ignored_tags.append((tag, 'replace'))
        
        if extra_anonymization_rules is not None:
            extra_anonymization_rules.update(private_tags_actions)
        else:
            extra_anonymization_rules = private_tags_actions

        super().anonymize_dataset(dataset, extra_anonymization_rules, delete_private_tags=False)

        # if not self.soft_detection:
        #     self.deidentify_all_ignored_tags_as_note(dataset)
        
        if self.apply_custom_actions:
            self.empty_remaining_pn_vr(dataset)

        if self.soft_detection:
            self.apply_soft_detections(dataset)
            self.check_and_replace_dates(dataset)

        if self.private_tags_extractor:
            # private_tag_actions = self.get_private_tags_anonymize_actions(dataset)
            # for tag in private_tag_actions.keys():
            #     action = private_tag_actions[tag]
            #     if action is not None:
            #         action(dataset, tag)
            #         self.history[tag] = action.__name__
            self.walk_and_anonymize_private_dataset(dataset)


            

        
