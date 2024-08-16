from typing import Union
import numpy as np

import pydicom
from transformers import pipeline


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
