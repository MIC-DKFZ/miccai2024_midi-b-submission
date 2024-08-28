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

PRIVATE_TAGS_DICT = 'dcm_anonymizers/tcia_private_tags_dict.json'

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

        self._load_private_tag_dict()

    def _load_private_tag_dict(self):
        with open(self.private_tag_dict_path) as f:
            self.private_tag_dict = json.load(f)
    
    @staticmethod
    def get_element_block_tag(element, private_block_name=None):
        group_str = f"{element.tag.group:04x}"   # Output: '0010'
        element_str = f"{element.tag.element:04x}" # Output: '0010'

        if private_block_name is None:
            return f"({group_str},{element_str})"
        else:
            return f"({group_str},{private_block_name.lower()},{element_str[-2:]})"

    @staticmethod
    def get_element_block_tag_with_parents(element, private_block_name=None, parent_blocks: list = []):
        parent_block_str = ""
        for idx, parent_tuple in enumerate(parent_blocks):
            parent_block_tag = PrivateTagsExtractorV2.get_element_block_tag(parent_tuple[0], parent_tuple[1])
            parent_block_str += f"{parent_block_tag}[<{idx}>]"

        element_block_tag = PrivateTagsExtractorV2.get_element_block_tag(element, private_block_name)

        vr_val = element.VR

        # if vr value like VR.UN, split and take last one
        vr_splits = vr_val.split('.')
        if len(vr_splits) > 0:
            vr_val = vr_splits[-1]

        return f"{parent_block_str}{element_block_tag}_{vr_val}"

    def get_dispoistion_val_from_block_tag(self, block_tag: str, element: pydicom.DataElement):
        if element.name.lower() == "private creator":
            return 'k'
        
        private_rules = self.private_tag_dict.get(block_tag)
        if private_rules is None:
            print(f"Warning!!! '{block_tag}': {element.name} not found in private tags dictionary")
            return 'k'

        return private_rules['private_disposition']