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
        
        outputs = self.model(text)
        outputs = self.filter_outputs(outputs)
        entities = self.process_outputs(outputs)
   
        return entities
    
    def detect_entities_from_element(self, element: pydicom.DataElement) -> list:
        element_text = f"{element.name}: {self.process_element_val(element)}"
        return self.detect_entities(element_text)
    
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
