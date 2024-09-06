import re
from pprint import pprint
import subprocess
from typing import List
import pydicom
from pydicom.datadict import keyword_for_tag, dictionary_VR
from pydicom.uid import generate_uid


class ValidationItem:
    def __init__(
        self, tag, type, message, name="", module="", index=None, raw=""
    ) -> None:
        if not tag or tag == "":
            raise ValueError("Invalid Tag provided")
        self.tag = tag
        self.type = type
        self.message = message
        self.name = name
        self.module = module
        self.index = index
        self.raw = raw
        self.list_of_dicoms = []
        return

    def __str__(self):
        if self.raw != "":
            return self.raw
        return f'{self.type}:\nTag: {self.tag}\nName: {self.name}\nIndex: {self.index}\nMessage: {self.message}\n'

    def add_dicom(self, dicom_name: str):
        self.list_of_dicoms.append(dicom_name)


class MissingAttributeItem(ValidationItem):
    def __init__(self, tag: pydicom.tag.Tag, type, message, name="", module="", index=None, raw="", missing_type="", parents=[]) -> None:
        super().__init__(tag, type, message, name, module, index, raw)

        self.missing_type = missing_type
        self.parents: list[pydicom.tag.Tag] = parents

    def __str__(self):
        if self.raw != "":
            return self.raw
        if len(self.parents) > 0:
            parents_list = [str(item) for item in self.parents]
            return f"""\
{self.type}:\nTag: {self.tag}\nName: {self.name}\nIndex: {self.index}\n\
Parents: {','.join(parents_list)}\nMessage: {self.message}\nType: {self.missing_type}\nModule: {self.module}\
"""
        else:
            return f"""\
{self.type}:\n\
Tag: {self.tag}\nName: {self.name}\nIndex: {self.index}\n\
Message: {self.message}\nType: {self.missing_type}\nModule: {self.module}\
"""



class DCIodValidator():
    def __init__(self) -> None:
        super().__init__()
        self.added_attr_log = {}
        self.ignore_list = [
            'CodeValue', 'CodeMeaning', 
            'LongCodeValue', 'URNCodeValue',
            'ValueType', 'Manufacturer', 
            'ClinicalTrialSubjectID', 'ClinicalTrialSubjectReadingID', 'ClinicalTrialSiteName',
            'Laterality', 'PurposeOfReferenceCodeSequence',
            'DoubleFloatRealWorldValueFirstValueMapped', 'DoubleFloatRealWorldValueLastValueMapped',
            'CodingSchemeDesignator',
            'ReferencedSOPClassUID', "ReferencedSOPInstanceUID",
        ]

    @staticmethod
    def process_dciodvfy_output(output: str):
        """
        Process the output from the dciodvfy command.

        Args:
            output (str): The raw output string from the dciodvfy command.

        Returns:
            List[Tuple[str]]: A list of tuples, where each tuple contains parts of an error or warning message.
        """
        lines = output.split("\n")
        outputs = []

        for l in lines:
            if l.strip() == "":
                continue

            splitted = tuple(l.split(" - "))

            outputs.append(splitted)

        return outputs

    @staticmethod
    def get_validataion_item_from_err_tuple(source: tuple, ref: int = -1):
        """
        Extract a ValidationItem from an error tuple.

        Args:
            source (Tuple[str]): A tuple containing parts of an error or warning message.
            ref (int, optional): A reference index used for naming general tags. Defaults to -1.

        Returns:
            Union[ValidationItem, None]: A ValidationItem instance if extraction is successful, otherwise None.
        """
        valid_tag_regex = re.compile(r"<\/|[A-Za-z0-9]*\([0-9]{4},[0-9]{4}\)\[\d\]>")

        def extract_tag(target: str):
            extracted = re.search(r"\([0-9a-fA-F]{4},[0-9a-fA-F]{4}\)", target)
            if extracted:
                return extracted.group(0)
            extracted_wout_brakets = re.search(r"[0-9a-fA-F]{4},[0-9a-fA-F]{4}", target)
            if extracted_wout_brakets:
                return f"({extracted_wout_brakets.group(0)})"

            # print(f"Could not able to extract tag from {target}")
            return ""

        def extract_name(target: str):
            isvalid = re.match(valid_tag_regex, target)
            if not isvalid:
                # print(f'{target} is not a valid tag')
                return ""

            name_extracted = re.search(r"<\/[A-Za-z0-9]+\(", target)
            if name_extracted:
                extract = name_extracted.group(0)
                return extract[2:-1]

            return ""

        def extract_index(target: str):
            isvalid = re.match(valid_tag_regex, target)
            if not isvalid:
                return None

            idx_extracted = re.search(r"\[\d\]", target)
            if idx_extracted:
                extract = idx_extracted.group(0)
                return int(extract[1:-1])

            return None

        if len(source) > 1:
            vtype = source[0]
            tag = extract_tag(source[1])
            if tag == "" and ref != -1:
                tag = f"general-{ref}"
            elif tag == "":
                tag = "general"
            name = extract_name(source[1])
            index = extract_index(source[1])
            message = source[2]
            module = ""
            if len(source) > 3:
                module = source[3]
            return ValidationItem(
                tag,
                vtype,
                message,
                name,
                module,
                index=index,
                raw=' - '.join(source[1:]),
            )

        return None
    
    @staticmethod
    def extract_tag_name_and_tag(plaintag: str):
        tag_extract_rgx = re.compile(r'([A-Za-z]+)\(([0-9a-fA-F]{4}),([0-9a-fA-F]{4})\)')

        extracted_tag_groups = re.search(tag_extract_rgx, plaintag)

        tag = pydicom.tag.Tag(int(extracted_tag_groups[2], 16), int(extracted_tag_groups[3], 16))
        assert extracted_tag_groups[1] == keyword_for_tag(tag)
        return extracted_tag_groups[1], tag

    @staticmethod
    def extract_missing_type(errmsg: str):
        type_extrct_rgx = re.compile(r'\b[Tt]ype\s+(\d[A-Z]?)\b')
        extracts = re.search(type_extrct_rgx, errmsg)
        if extracts:
            return extracts.group(1)
        else:
            return ""
    
    @staticmethod
    def validation_item_to_missing_attribute_item(vitem: ValidationItem):
        mssgparts = vitem.raw.split(' - ')
        tag_rgx = re.compile(r'<([^>]*)>')

        element_tags = re.search(tag_rgx, mssgparts[0])
        element_tags_splits = element_tags.group(1).split('/')
        element_tags_splits = [elem for elem in element_tags_splits if elem.strip() != '']

        parents = []
        if len(element_tags_splits) > 1:
            for split in element_tags_splits[:-1]:
                _, tag = DCIodValidator.extract_tag_name_and_tag(split)
                parents.append(tag)
        
        name, tag = DCIodValidator.extract_tag_name_and_tag(element_tags_splits[-1])

        module = ""
        if vitem.module != "":
            module_extracts = re.search(tag_rgx, vitem.module)
            module = module_extracts.group(1)
        
        return MissingAttributeItem(
            type = vitem.type,
            tag = tag,
            name = name,
            message = vitem.message,           
            missing_type=DCIodValidator.extract_missing_type(vitem.message),
            index = len(parents),
            parents = parents,
            module = module,
            # raw = vitem.raw,
        )


    @staticmethod
    def filter_missing_attributes_errors(errors: List[ValidationItem], filter_mssg: str = 'Missing attribute'):
        filtered = [err for err in errors if filter_mssg in err.message]
        filtered = [DCIodValidator.validation_item_to_missing_attribute_item(err) for err in filtered]
        return filtered

    def validate_dicom(self, dicom_path: str) -> tuple:
        """
        Validate a DICOM file using the dciodvfy tool.

        Args:
            dicom_path (str): The file path of the DICOM file to be validated.

        Returns:
            Tuple[List[ValidationItem], List[ValidationItem]]: Two lists containing error and warning ValidationItems respectively.
        """
        cmd = ["dciodvfy", "-new", dicom_path]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        _, err = process.communicate()
        errs = self.process_dciodvfy_output(err.decode("utf-8"))
        vitems = [
            self.get_validataion_item_from_err_tuple(item, idx)
            for idx, item in enumerate(errs)
            if self.get_validataion_item_from_err_tuple(item)
        ]
        errors, warnings = [], []

        for item in vitems:
            if item.type == "Error":
                errors.append(item)
            else:
                warnings.append(item)

        return errors, warnings

    @staticmethod
    def get_empty_element_value_for_tag(tag):
        elem_vr = dictionary_VR(tag)
        # elem_name = keyword_for_tag(tag)
        elem_val = None
        if elem_vr in ("SH", "PN", "LO", "LT", "CS", "ST", "UT"):          
            elem_val = ""
        elif elem_vr == "UI":
            elem_val = generate_uid()
        elif elem_vr in ("DT", "DA", "TM"):
            elem_val = ""
        elif elem_vr in ("UL", "FL", "FD", "SL", "SS", "US"):
            elem_val = 0
        elif elem_vr in ("DS", "IS"):
            elem_val = "0"
        elif elem_vr == "UN":
            elem_val = b""
        else:
            pass
        return elem_val

    def create_element_from_tag(self, tag, forced: bool = False):
        elem_vr = dictionary_VR(tag)
        elem_name = keyword_for_tag(tag)
        elem_val = DCIodValidator.get_empty_element_value_for_tag(tag)

        if elem_name in self.ignore_list and not forced:
            return None
        elif elem_val is None:
            return None
        else:
            elem = pydicom.dataelem.DataElement(tag, elem_vr, elem_val)
            return elem
        
    def create_empty_element(self, ds, element_tag, parents: list = [], forced: bool = False):
        created = False
        selected_ds = None
        if len(parents) > 1:
            sub_dataset = ds
            for ptag in parents:
                sq_elem = sub_dataset.get(ptag)
                sub_dataset = sq_elem.value[0]
            selected_ds = sub_dataset
        elif len(parents) == 1:
            element = ds.get(parents[0])
            if element is not None:
                if len(element.value) > 0:
                    selected_ds = element.value[0]
                else:
                    selected_ds = pydicom.dataset.Dataset()
                    element.value.append(selected_ds) 
        else:
            selected_ds = ds

        if selected_ds is not None:
            # if element already exists in dataset return
            element = selected_ds.get(element_tag)
            if element is not None:
                print(f"{keyword_for_tag(element_tag)} already present in the dataset")
                return created
            new_element = self.create_element_from_tag(element_tag, forced)
            if new_element is not None:
                selected_ds.add(new_element)
                created = True
                elem_name = keyword_for_tag(element_tag)
                if elem_name in self.added_attr_log:
                    self.added_attr_log[elem_name] += 1
                else:
                    self.added_attr_log[elem_name] = 1   
            else:
                elem_name = keyword_for_tag(element_tag)
                if elem_name not in self.ignore_list:
                    print(f"Element can not be created for tag {element_tag} {keyword_for_tag(element_tag)}") 
        
        return created
    
    # def get_missing_attribute_errors_from_dcm_path(self, dcm_path: str):
    #     errors, _ = self.validate_dicom(dcm_path)
    #     missing_attribute_errors = DCIodValidator.filter_missing_attributes_errors(errors)
    #     return missing_attribute_errors

    @staticmethod
    def print_valitation_item_list(items: list):
        for item in items:
            print(item)

    def populate_missing_attributes(self, dcm_path: str):
        errors, _ = self.validate_dicom(dcm_path)
        missing_attribute_errors = DCIodValidator.filter_missing_attributes_errors(errors)
        ds = pydicom.dcmread(dcm_path)

        attribute_created = 0
        added_tags = []
        for error in missing_attribute_errors:
            created = self.create_empty_element(ds, error.tag, error.parents)
            if created:
                attribute_created += 1
                added_tags.append(error.tag)

        missing_sequence_number_attr_errs = DCIodValidator.filter_missing_attributes_errors(errors, filter_mssg='Bad Sequence number')
        for error in missing_sequence_number_attr_errs:
            # forcefully add ReferencedPatientSequence(0008,1120)[1]ReferencedSOPInstanceUID(0008,1155)
            if error.tag == (0x0008, 0x1120):
                created = self.create_empty_element(ds, pydicom.tag.Tag(0x0008, 0x1150), [error.tag], forced=True)
                if created:
                    attribute_created += 1
                    added_tags.append(error.tag)

        # update dicom if new attribute added
        if attribute_created > 0:
            ds.save_as(dcm_path)
            # verify if errors increased
            newerrs, _ = self.validate_dicom(dcm_path)
            if len(errors) != len(newerrs) + attribute_created:
                tags_str = ', '.join([keyword_for_tag(item) for item in added_tags])
                print(f"New errors created after empty attribute addition for tags: {tags_str}")

    
    @staticmethod
    def get_error_differences(error_list_1, error_list_2):
        if len(error_list_1) > len(error_list_2):
            list_1, list_2 = error_list_1, error_list_2
        else:
            list_1, list_2 = error_list_2, error_list_1
        
        smaller_set = set((x.tag,x.name) for x in list_2)
        difference = [x for x in list_1 if (x.tag,x.name) not in smaller_set]
        return difference
    
    def compare_dicom_validations(self, source_dcm_path: str, target_dcm_path: str):
        source_errs, _ = self.validate_dicom(source_dcm_path)
        target_errs, _ = self.validate_dicom(target_dcm_path)

        if len(source_errs) != len(target_errs):
            return False

        difference = DCIodValidator.get_error_differences(source_errs, target_errs)

        return len(difference) == 0


        

