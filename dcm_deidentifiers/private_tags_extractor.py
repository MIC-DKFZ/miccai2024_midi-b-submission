import re
import json
import pydicom
from difflib import SequenceMatcher

import pandas as pd


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()
    
def replace_non_alphanumeric_with_space(input_string):
    # Replace all non-alphanumeric characters with a space
    result = re.sub(r'[^a-zA-Z0-9]', ' ', input_string)
    return result

PRIVATE_TAGS_DICT = 'dcm_deidentifiers/tcia_private_tags_dict.json'

class PrivateTagsExtractor:
    def __init__(self, private_tags_dict_path: str):
        self.private_tag_dict_path = private_tags_dict_path
        self.private_tag_df = None
        self.filtered_private_tag_df = None

        self._load_private_tag_dict()

    def _load_private_tag_dict(self):
        self.private_tag_df = pd.read_csv(self.private_tag_dict_path)
        self.private_tag_df['vr'] = self.private_tag_df['vr'].astype('category')
        self.private_tag_df['private_disposition'] = self.private_tag_df['private_disposition'].astype('category')

        self.private_tag_df['tag_group'] = self.private_tag_df['element_sig_pattern'].str[1:5]
        self.private_tag_df['tag_group'] = self.private_tag_df['tag_group'].astype('category')

    def filter_by_tag_group(self, tag_groups: list):
        return self.private_tag_df[self.private_tag_df['tag_group'].isin(tag_groups)]

    def filter_by_pattern_n_vr(self, filter_pattern, vr):
        df = self.private_tag_df
        if self.filtered_private_tag_df is not None:
            df = self.filtered_private_tag_df
        return df[
            (df['element_sig_pattern'].str.match(filter_pattern)) & 
            (df['vr'] == vr)
        ]
        
    def search_patterns_from_element(self, element, creators: list = [], include_base_tag_pattrn: bool = False):
        # Get the group and element separately
        group = element.tag.group
        dataelement = element.tag.element
        
        # Convert to hexadecimal string format
        group_str = f"{group:04x}"   # Output: '0010'
        element_str = f"{dataelement:04x}" # Output: '0010'

        all_patterns = []

        for creator in creators:
            pttrn = fr'\({group_str},{creator}.*,{element_str[2:]}\)'
            all_patterns.append(pttrn)

        all_patterns.append(fr'\({group_str},.*,{element_str[2:]}\)')
        if include_base_tag_pattrn:
            all_patterns.append(fr'\({group_str},.*{element_str[2:]}\)')

        return all_patterns

    def get_filtered_rows_from_patterns(self, patterns: list, element, strict: bool = True):
        filtered = []
        for pttrn in patterns:
            filtered_rules = self.filter_by_pattern_n_vr(pttrn, element.VR)
            if not strict:
                filtered = filtered_rules
            if len(filtered_rules) == 1:
                filtered = filtered_rules
                break
        return filtered

    def get_private_disposition_from_rows(self, filtered_rows, element, match_names: bool = True):
        if len(filtered_rows) == 0:
            return 'k'
        disposition_val = 'k'
        
        first_row = filtered_rows.iloc[0]

        if match_names:
            # get the similarity of the tag name with the element name
            row_name = first_row['tag_name']
            similarity = similar(
                element.name.lower(), 
                replace_non_alphanumeric_with_space(row_name.lower())
            )            
            if similarity > 0.5:
                disposition_val = first_row['private_disposition']
        else:
            disposition_val = first_row['private_disposition']
            
        return disposition_val.lower().strip()
    

class PrivateTagsExtractorV2:
    def __init__(self, private_tags_dict_path: str = PRIVATE_TAGS_DICT):
        self.private_tag_dict_path = private_tags_dict_path
        self.private_tag_dict = None
        self.override_dict = {
            "(0009,gems_petd_01,0f)_ST": {
                "pattern": "(0009,GEMS_PETD_01\",0f)",
                "tag_name": "Scan Description",
                "vr": "ST",
                "private_disposition": "k"
            },
            "(0009,gems_petd_01,37)_LO": {
                "pattern": "(0009,GEMS_PETD_01\",37)",
                "tag_name": "Batch Description",
                "vr": "LO",
                "private_disposition": "d"
            },
            "(0117,ucsf birp private creator 011710xx,c5)_UN": {
                "pattern": "(0117,UCSF BIRP PRIVATE CREATOR 011710xx\",c5)",
                "tag_name": "Protocol compliance",
                "vr": "UN",
                "private_disposition": "k"
            }
        }

        self._load_private_tag_dict()

    def _load_private_tag_dict(self):
        with open(self.private_tag_dict_path) as f:
            self.private_tag_dict = json.load(f)
        
        if len(self.override_dict) > 0:
            self.private_tag_dict.update(self.override_dict)
    
    @staticmethod
    def get_element_block_tag(element, private_block_name=None):
        group_str = f"{element.tag.group:04x}"   # Output: '0010'
        element_str = f"{element.tag.element:04x}" # Output: '0010'

        if private_block_name is None:
            return f"({group_str},{element_str})"
        else:
            # strip if block name contains `,`, since dict key also stripped by `,`
            private_block_splits = private_block_name.split(',')
            private_block_splits = [s.strip() for s in private_block_splits]
            private_block_name = ','.join(private_block_splits).lower()

            return f"({group_str},{private_block_name},{element_str[-2:]})"

    @staticmethod
    def get_element_block_tags_with_parents(element, private_blocks: list, parent_blocks: list = []):
        parent_block_str_list = []

        for private_block_name in reversed(private_blocks):
            parent_block_str = ""
            if len(parent_blocks) > 0:
                for idx, parent_elem in enumerate(parent_blocks):
                    parent_block_tag = PrivateTagsExtractorV2.get_element_block_tag(parent_elem, private_block_name)
                    parent_block_str += f"{parent_block_tag}[<{idx}>]"
                
                parent_block_str_list.append(parent_block_str)

        if len(parent_block_str_list) == 0:
            parent_block_str_list.append("")

        block_tags = []
        for parent_block_str in parent_block_str_list:
            for private_block_name in reversed(private_blocks):
                element_block_tag = PrivateTagsExtractorV2.get_element_block_tag(element, private_block_name)

                vr_val = element.VR

                # if vr value like VR.UN, split and take last one
                vr_splits = vr_val.split('.')
                if len(vr_splits) > 0:
                    vr_val = vr_splits[-1]

                block_tag = f"{parent_block_str}{element_block_tag}_{vr_val}"
                block_tags.append(block_tag)

        return block_tags

    def get_dispoistion_val_from_block_tags(self, block_tags: list[str], element: pydicom.DataElement):
        if element.name.lower() == "private creator":
            return 'k'
        
        private_rules = None
        for block_tag in block_tags:
            private_rules = self.private_tag_dict.get(block_tag)
            if private_rules is not None:
                break

        if private_rules is None:
            print(f"Warning!!! '{block_tags[0]}': {element.name} not found in private tags dictionary")
            return 'k'

        return private_rules['private_disposition']