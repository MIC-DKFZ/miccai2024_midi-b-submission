import tempfile
from pathlib import Path

from deid_app.robust_app import RobustDeID

temdir = '.tmp'
tempfile.tempdir = '.tmp'


def delete_files(path: Path):
    """
    Delete all files in a folder.
    """
    for file in path.glob('*'):
        if file.is_file():
            file.unlink()



def deid(texts: list, app: RobustDeID):
    notes = []

    for idx, text in enumerate(texts):
        note = {"text": text, "meta": {"note_id": f"note_{idx}", "patient_id": "patient_1"}, "spans": []}
        notes.append(note)

    ner_notes = app.get_ner_dataset_from_json_list(notes)

    predictions = app.get_predictions_from_generator(ner_notes)

    predictions_list = [item for item in predictions]

    deid_dict_list = list(app.get_deid_text_replaced_from_values(notes, predictions_list))
    # Get deid text
    deid_texts = [pred_dict['deid_text'] for pred_dict in deid_dict_list]

    highlight_texts = []

    for deid_text in deid_texts:
        highligted = [highlight_text for highlight_text in RobustDeID._get_highlights(deid_text)]
        highlight_texts.append(highligted)

    return highlight_texts

if __name__ == "__main__":
    text = """\
Private Creator: GEIIS, Private Creator: CTP, Private tag data: Pseudo-PHI-DICOM-Data, Private tag data: 87009668, \
Private Creator: SIEMENS CSA HEADER, Private Creator: SIEMENS MEDCOM HEADER, Private Creator: SIEMENS MEDCOM OOG, \
Private Creator: GEIIS PACS, Private Creator: GEIIS, Private Creator: GEIIS\
"""

    modelname = "OBI-RoBERTa De-ID"
    threshold = "No threshold"

    app = RobustDeID(modelname, threshold)

    loggers = [
        'robust_deid.sequence_tagging.sequence_tagger'
    ]

    texts = [text] * 5
    highlight_texts = deid(texts, app)

    
    for highlighted in highlight_texts:
        print(highlighted)

    # delete_files(Path(temdir))
