import re
from typing import Union
import numpy as np

import pydicom
from transformers import pipeline

from deid_app.robust_app import RobustDeID as RobustDeIDPipeline

class DcmPHIDetector:
    def __init__(self) -> None:
        self.model_name = "obi/deid_roberta_i2b2"
        self.model = None
        self.min_confidence = np.float32(0.6)

        self._init_model()
    
    def _init_model(self):
        self.model = pipeline("ner", model=self.model_name, aggregation_strategy="simple")

    def entity_name(self, entityvalue: str):
        return entityvalue[2:]

    def process_enitity_val(self, entityvalue: str):
        spacechar = 'Ġ'
        if entityvalue[0] == spacechar:
            entityvalue = entityvalue[1:]
        entityvalue = entityvalue.replace(spacechar, ' ')
        return entityvalue
    
    def safe_str(self, elementval: str):
        if isinstance(elementval, bytes):
            try:
                decoded = elementval.decode("utf-8")
            except UnicodeError:
                decoded = str(elementval)
            return decoded
        return str(elementval)

    def process_element_val(self, element):
        elementval = ""
        if self.safe_str(element.value).strip() == "":
            return elementval
        
        if element.VM > 1:
            elementval = ', '.join([self.safe_str(item) for item in element.value])
        elif element.VM == 1:
            elementval = self.safe_str(element.value)
        elementval = elementval.replace("'", '')
        if element.VR == 'PN':
            elementval = elementval.replace("^", ' ')
        return elementval.strip()

    def processed_element_name(self, element_name: str):
        element_name = element_name.strip()
        return element_name.lstrip("[").rstrip("]") 
    
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
        """
        input -> Scott Community Hospital
        outputs -> (Scott Community Hospital, hospital)
        """
        if text == "":
            return []
        
        outputs = self.model(text)
        outputs = self.filter_outputs(outputs)
        entities = [(entity["word"].strip(), entity["entity_group"], entity["start"]) for entity in outputs]
        # entities = self.process_outputs(outputs)
   
        return entities
    
    def detect_entities_from_element(self, element: pydicom.DataElement) -> list:
        element_name = self.processed_element_name(element.name)
        element_text = f"{element_name}: {self.process_element_val(element)}"
        entities = self.detect_entities(element_text)
        # filter entitites, if those are detected from entity name
        entities = [entity for entity in entities if entity[2] > len(element_name)]
        return entities
    
    def deidentified_element_val(self, element: pydicom.DataElement) -> Union[str, list[str]]:
        if self.safe_str(element.value).strip() == "":
            return element.value
        
        isbyte = isinstance(element.value, bytes)
        
        processed_element_val = self.process_element_val(element)
        element_text = f"{self.processed_element_name(element.name)}: {processed_element_val}"
        entities = self.detect_entities(element_text)
        if len(entities) == 0:
            return element.value
        
        all_entity_vals = ''.join([e[0] for e in entities])
        # ignore if the proposed entity is too small compared to 
        # the real value
        if len(all_entity_vals) <= 3 and len(processed_element_val) > 6:
            return element.value
        
        processed = element_text[:]
        for e in entities:
            target_substr = element_text[e[2]: (e[2]+len(e[0]))]
            processed = processed.replace(target_substr, '')
        
        deidentified_val = processed.replace(f"{self.processed_element_name(element.name)}: ", '')

        if element.VM > 1:
            splitted = deidentified_val.split(', ')
            if isbyte:
                splitted = [str.encode(s) for s in splitted]
            return splitted
        
        remaining_value_prcnt = len(deidentified_val) / len(processed_element_val)
        # return empty string in case of value almost stripped by anonymizer
        if remaining_value_prcnt <= 0.2:
            deidentified_val = ""
        
        deidentified_val = deidentified_val.strip()

        if isbyte:
            return str.encode(deidentified_val)

        return deidentified_val
    
    def detect_enitity_tags_from_text(self, all_texts: str, text_tag_mapping: dict):
        entities = self.detect_entities(all_texts)
        element_target = {}

        for e in entities:
            e_start = e[2]
            for t in text_tag_mapping:
                if e_start >= text_tag_mapping[t][0] and e_start <= text_tag_mapping[t][1]:
                    if t in element_target:
                        element_target[t].append(e[0])
                    else:
                        element_target[t] = [e[0]]
        
        return element_target
    
    def deid_element_values_from_entity_values(self, elemval, entity_values: list):
        isbyte = isinstance(elemval, bytes)

        elemval = self.safe_str(elemval)
        n_words = len(elemval.split())
        deid_val = elemval[:]
        if n_words == 1:
            elem_n_chars = len(elemval)
            entity_n_chars = len(''.join(entity_values))
            if elem_n_chars < 2*entity_n_chars:
                deid_val = ""
        else:
            for entity_val in entity_values:
                deid_val = deid_val.replace(entity_val, '', 1)
                
            remaining_value_prcnt = len(deid_val) / len(elemval)
            # return empty string in case of value almost stripped by anonymizer
            if remaining_value_prcnt <= 0.2:
                deid_val = ""

        deid_val = deid_val.strip()

        if isbyte:
            return str.encode(deid_val)

        return deid_val


class DcmRobustPHIDetector:
    def __init__(self, logging: bool = False) -> None:
        self.modelname = "OBI-RoBERTa De-ID"
        self.threshold = "No threshold"
        self.pipeline = None
        
        self.logging = logging
        self.detected_entity_log = {}
        self.missed_by_whitelist = {}

        self._init_pipeline()
    
    def _init_pipeline(self):
        self.pipeline = RobustDeIDPipeline(
            self.modelname, 
            self.threshold,
            disable_logs=True
        )
    
    @staticmethod
    def safe_str(elementval: str):
        if isinstance(elementval, bytes):
            try:
                decoded = elementval.decode("utf-8")
            except UnicodeError:
                decoded = str(elementval)
            return decoded
        return str(elementval)

    @staticmethod
    def process_element_val(element):
        elementval = ""
        if DcmRobustPHIDetector.safe_str(element.value).strip() == "":
            return elementval
        
        if element.VM > 1:
            items = [DcmRobustPHIDetector.safe_str(item) for item in element.value if item.strip() != '']
            elementval = ', '.join(items)
        elif element.VM == 1:
            elementval = DcmRobustPHIDetector.safe_str(element.value)
        elementval = elementval.replace("'", '')
        if element.VR == 'PN':
            elementval = elementval.replace("^", ' ')
        
        combine_whitespace_pttrn = re.compile(r"\s+")
        elementval = combine_whitespace_pttrn.sub(" ", elementval).strip()
        return elementval

    @staticmethod
    def processed_element_name(element_name: str):
        element_name = element_name.strip()
        return element_name.lstrip("[").rstrip("]") 

    @staticmethod
    def element_to_text(element):
        return f"{DcmRobustPHIDetector.processed_element_name(element.name)}: {DcmRobustPHIDetector.process_element_val(element)}"
    
    @staticmethod
    def clear_mistaken_highlights(textpart: str):
        pattern = re.compile(r'<<(PATIENT|STAFF|AGE|DATE|LOCATION|PHONE|ID|EMAIL|PATORG|HOSPITAL|OTHERPHI):((.)*)?>>', re.DOTALL)
        matches = re.search(pattern, textpart)
        if matches:
            textpart = textpart.replace(matches.group(0), matches.group(2))

        return textpart
    
    def run_deid(self, texts: list[str]):
        notes = []

        for idx, text in enumerate(texts):
            note = {"text": text, "meta": {"note_id": f"note_{idx}", "patient_id": "patient_1"}, "spans": []}
            notes.append(note)

        ner_notes = self.pipeline.get_ner_dataset_from_json_list(notes)

        predictions = self.pipeline.get_predictions_from_generator(ner_notes)

        predictions_list = [item for item in predictions]

        deid_dict_list = list(self.pipeline.get_deid_text_replaced_from_values(notes, predictions_list))

        # Get deid text
        deid_texts = [pred_dict['deid_text'] for pred_dict in deid_dict_list]

        highlight_texts = []
        for deid_text in deid_texts:
            highligted = [highlight_text for highlight_text in RobustDeIDPipeline._get_highlights(deid_text)]
            highlight_texts.append(highligted)

        return highlight_texts


    def detect_entities(self, text: str):
        """
        input -> "Hospital Name: Scott Community Hospital"
        outputs -> [('Scott Community Hospital', 'HOSP', 0)]
        """
        if text.strip() == "":
            return []
        
        outputs = self.run_deid([text])
        
        entities = []
        current = 0

        for item in outputs[0]:
            itemval = item[0]
            itemval = DcmRobustPHIDetector.clear_mistaken_highlights(itemval)

            if item[1]:
                found_in = text[current:].find(itemval)
                start = current + found_in
                entity = (itemval, item[1], start)
                entities.append(entity)

                assert text[start:start+len(item[0])] == itemval, f"segmenting entities from note text mismatch, {text[start:start+len(item[0])]} -> {itemval}"

            current += len(itemval)
   
        return entities

    def detect_entities_from_element(self, element: pydicom.DataElement) -> list:
        element_name = self.processed_element_name(element.name)
        element_text = f"{element_name}: {self.process_element_val(element)}"
        entities = self.detect_entities(element_text)
        # filter entitites, if those are detected from entity name
        entities = [entity for entity in entities if entity[2] > len(element_name)]
        return entities
    
    def append_entity_count_to_log_dict(self, log_dict: dict, entity: tuple):
        entity_val = entity[0]
        if entity_val in log_dict:
            log_dict[entity_val] += 1
        else:
            log_dict[entity_val] = 1

    
    def filter_entities_by_whitelist(self, entities):
        
        word_match_tmplt = r'(?i)\b{}\b'
        whole_string_match_templt = r'(?i)^{}$'

        entity_whitelist = [
            'breast?', 'contrast', 'bilateral', 'ressonancia', 
            'magnetica', 'pelve', 'lung', 'chest', 'abdomen', 
            'miednicy',
        ]

        exact_match_whitelist = [
            ',', '-', '\(', '\)', 
            'price', 'short', 'glass', 
            'brzuchmiednica', 'h20', 'day', 
            'jpeglosslessprocfirstorderredict', 
            'kidney_mass_hematuria_bmi_under30', 'thins'
        ]

        filtered = []
        for entity in entities:
            matched = False

            for pattrn in exact_match_whitelist:
                match = re.search(whole_string_match_templt.format(pattrn), entity[0])
                if match is not None:
                    matched = True
                    if self.logging:
                        self.append_entity_count_to_log_dict(self.missed_by_whitelist, entity)
                    break

            if not matched:
                for pattrn in entity_whitelist:
                    match = re.search(word_match_tmplt.format(pattrn), entity[0])
                    if match is not None:
                        matched = True
                        if self.logging:
                            self.append_entity_count_to_log_dict(self.missed_by_whitelist, entity)
                        break

            if not matched:
                filtered.append(entity)

        return filtered

    def detect_enitity_tags_from_text(self, all_texts: str, text_tag_mapping: dict):
        entities = self.detect_entities(all_texts)
        entities = self.filter_entities_by_whitelist(entities)

        if self.logging:
            for e in entities:
                self.append_entity_count_to_log_dict(self.detected_entity_log, e)

        element_target = {} 
        for e in entities:
            e_start = e[2]
            for t in text_tag_mapping:
                if e_start >= text_tag_mapping[t][0] and e_start <= text_tag_mapping[t][1]:
                    if t in element_target:
                        element_target[t].append(e[0])
                    else:
                        element_target[t] = [e[0]]
        
        return element_target
    
    def deid_element_from_entity_values(self, element: pydicom.DataElement, entity_values: list):
        isbyte = isinstance(element.value, bytes)

        elemval = self.process_element_val(element)
        deid_val = elemval[:]
        entity_n_chars = len(''.join(entity_values))

        if len(elemval) == entity_n_chars:
            return ""
        
        for entity_val in entity_values:
            if entity_val in deid_val:
                deid_val = deid_val.replace(entity_val, '', 1)
            else:
                entity_pattern = re.sub(r'\s+', '.*', entity_val)
                match = re.search(entity_pattern, deid_val)
                if match:
                    deid_val = deid_val.replace(match.group(), '', 1)
                else:                
                    print(f"{entity_val} not found in the original value {elemval}")

        deid_val = deid_val.strip()

        if element.VM > 1:
            splitted = deid_val.split(', ')
            if isbyte:
                splitted = [str.encode(s) for s in splitted]
            return splitted
        
        if isbyte:
            return str.encode(deid_val)

        return deid_val
    
    def deidentified_element_val(self, element: pydicom.DataElement) -> Union[str, list[str]]:
        if DcmRobustPHIDetector.safe_str(element.value).strip() == "":
            return element.value
        
        entities = self.detect_entities_from_element(element)
        if len(entities) == 0:
            return element.value
        
        entity_values = [entity[0] for entity in entities]

        return self.deid_element_from_entity_values(element, entity_values)

        