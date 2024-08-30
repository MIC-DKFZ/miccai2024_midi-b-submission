import csv
from pathlib import Path
import pydicom
from pprint import pprint
import tqdm
from pydicom import dcmread

from utils.dataloaders import MIDIEvalDataLoader
from dcm_anonymizers.utils import list_all_files
from dcm_anonymizers.phi_detectors import DcmRobustPHIDetector
from dcm_anonymizers.img_anonymizers import DCMImageAnonymizer

def id_map_csv_to_dict(csvfile: str):
    id_map = {}
    with open(path_mapping_file, mode ='r')as file:
      mapping = csv.reader(file)
      for idx, lines in enumerate(mapping):
          if idx == 0:
              continue
          id_map[lines[0]] = lines[1]
    return id_map 

def get_dcm_paths_from_series(seriesUID: str, series_output_map: dict):
    series_path = series_output_map.get(seriesUID, '')
    if series_path == "":
        print(f"No path found for given series id {seriesUID}")
        return
    full_series_path = anonymizer_output_path / 'data' / series_path
    alldcms = list_all_files(full_series_path)
    if len(alldcms) == 0:
        print(f"No dicom found for given series id {seriesUID}")
        return
    return alldcms

def extract_tags(dcm, gt_ds, annon_ds, tagvalues):
    elements = dcm
    gt_elements = gt_ds
    annon_elements = annon_ds
    parent_tag = None
    
    if isinstance(dcm, pydicom.dataelem.DataElement):
        parent_tag = dcm.tag
        if len(dcm.value) == 0:
            return
        
        elements = dcm.value[0]
        gt_elements = None
        if gt_ds and len(gt_ds.value) > 0:
            gt_elements = gt_ds.value[0]
        
        annon_elements = None
        if annon_ds and len(annon_ds.value) > 0:
            annon_elements = annon_ds.value[0] if annon_ds else None
        
    for element in elements:
        deidelem = gt_elements.get(element.tag) if gt_elements else None
        dcmannonelem = annon_elements.get(element.tag) if annon_elements else None
        
        if element.VR == 'OW':
            continue
        elif element.VR == 'SQ':
            extract_tags(element, deidelem, dcmannonelem, tagvalues)
            continue
        # targettags.append(element.tag)
        
        deidval = ""
        if deidelem:
            deidval = str(deidelem.value)
        dcmannonval = ""
        if dcmannonelem:
            dcmannonval = str(dcmannonelem.value)
        changed = False
        if dcmannonval != deidval:
            changed = True

        element_tag_str = str(element.tag)
        if parent_tag:
            element_tag_str = f"{str(parent_tag)} - {str(element.tag)}"
        values_tuple = (element_tag_str, element.VR, element.name, str(element.value), deidval, dcmannonval, changed)
        tagvalues.append(values_tuple)


def find_mismatched_tags(tagvalues: list[tuple]):
    n_mismatched = 0
    mismatched_tags = []

    for row in tagvalues:
        gt_val = row[4]
        target_val = row[5]

        if gt_val != target_val:
            if row[1] == 'UI' and not (gt_val == "" or target_val == ""):
                continue
            elif row[1] in ('DA', 'DT', 'TM') and not (gt_val == "" or target_val == ""):
                if len(gt_val) != len(target_val):
                    pass
                continue
            elif row[0] in ('(0010, 0010)', '(0010, 0020)'):
                continue
            n_mismatched += 1
            mismatched_tags.append(row[2])

    return n_mismatched, mismatched_tags


def find_mismatched_in_pixel_data(imganonymizer: DCMImageAnonymizer, dcm_deid_gt, dcm_deid):
    gt_note, _, _ = imganonymizer.extract_texts_as_note(dcm_deid_gt.pixel_array)
    gt_texts = gt_note.split('\n')

    deidentified_note, _, _ = imganonymizer.extract_texts_as_note(dcm_deid.pixel_array)
    deidentified_texts = deidentified_note.split('\n')

    diff = abs(len(deidentified_texts) - len(gt_texts))
    # if diff > 0:
    #     print(gt_note)
    #     print(deidentified_note)

    return diff, len(gt_texts)


def evaluate_series_by_index(
        series_idx, loader, series_output_path_map, imganonymizer: DCMImageAnonymizer,
        evaluate_pixel_data: bool = True
    ):

    (rawdcm, metadata), (deiddcm, deiddcm_metadata) = loader.get_raw_n_deid_patient(series_idx, include_metadata=True)
    deidentfied_dcm_paths = get_dcm_paths_from_series(metadata['Series UID'], series_output_path_map)

    anonymized_dcms = []

    for dcmpath in deidentfied_dcm_paths:
        with open(dcmpath, 'rb') as infile:
            deidentfied_dcm = dcmread(infile)
            anonymized_dcms.append(deidentfied_dcm)

    total_elements = 0
    total_mismatched = 0
    mismatching_tags = {}

    if len(rawdcm) != len(deiddcm) or len(rawdcm) != len(anonymized_dcms):
        print(f"{metadata['Series UID']} Skipped, raw and deidentifed dicoms number mismatch.")
        return total_elements, total_mismatched, mismatching_tags

    for idx, dcm in enumerate(rawdcm):
        deid_gt = deiddcm[idx]
        anonymized = anonymized_dcms[idx]
        tagvalues = []
        extract_tags(dcm, deid_gt, anonymized, tagvalues)
        n_mismatched, mismatched_tags = find_mismatched_tags(tagvalues)

        total_elements += len(dcm)
        total_mismatched += n_mismatched

        for tag in mismatched_tags:
            if tag in mismatching_tags:
                mismatching_tags[tag] += 1
            else:
                mismatching_tags[tag] = 1

        # image anonymization evaluation
        if evaluate_pixel_data:
            n_img_mismatched, total_img_txts = find_mismatched_in_pixel_data(imganonymizer, deid_gt, anonymized)
            total_elements += total_img_txts
            total_mismatched += n_img_mismatched

            if n_img_mismatched > 0:
                mismatching_tags['text_from_image'] = n_img_mismatched

    return total_elements, total_mismatched, mismatching_tags


if __name__ == "__main__":
    root_data_dir = '/home/r079a/Desktop/de-identification/dataset'

    loader = MIDIEvalDataLoader(
        rawimagespath=Path(root_data_dir, 'images/manifest-1617826555824'),
        deidimagespath=Path(root_data_dir, 'images-2/manifest-1617826161202'),
        uidsmappath=Path(root_data_dir, 'Pseudo-PHI-DICOM-Dataset-uid_crosswalk.csv'),
    )

    detector = DcmRobustPHIDetector()
    img_anonymizer = DCMImageAnonymizer(phi_detector=detector)

    anonymizer_output_path = Path(root_data_dir, 'anonymizer-output/Pseudo-PHI-DICOM-Data-10-removed-ctp-custom')

    path_mapping_file = Path(anonymizer_output_path, 'mappings/path_mapping.csv')

    series_output_map = id_map_csv_to_dict(path_mapping_file)

    total_series = 26

    total_elements = 0
    total_mismatched = 0
    mismatching_tags = {}
    mismatching_tags_idx = {}
    progress_bar = tqdm.tqdm(total=total_series)

    for i in range(total_series):
        current_elements, current_mismatched, current_mismatching_tags = evaluate_series_by_index(
            i, loader, series_output_map, img_anonymizer, evaluate_pixel_data=False
        )
        total_elements += current_elements
        total_mismatched += current_mismatched
        
        for tag in current_mismatching_tags.keys():
            if tag in mismatching_tags:
                mismatching_tags[tag] += 1
                mismatching_tags_idx[tag].append(i)
            else:
                mismatching_tags[tag] = 1
                mismatching_tags_idx[tag] = [i]
        
        progress_bar.update(1)

    
    progress_bar.close()

    matching_accuracy = ((total_elements - total_mismatched) / total_elements) * 100
    print('Evaluation of the anonymization process complete')
    print("=================================================")
    print(f"Anonymization Closeness Score {round(matching_accuracy, 3)}%")
    print("=================================================")
    print("Mismatched Tags Summary:")
    print("--------------------------------------------")
    for tag in mismatching_tags.keys():
        mismatched_idx_str = ','.join(str(x) for x in mismatching_tags_idx[tag])
        print(f"\t{tag}: {mismatching_tags[tag]} -> {mismatched_idx_str}")
    print("---------------------------------------------")


# VR which needs to be replaced by AI
# LO, ST, LT
# PN -> empty
# Custom Rules
# (0x0008, 0x2111) | Derivation Description | remove -> replace / AI
# (0x0010, 0x2180) | Occupation | remove -> keep
# (0x0012, 0x0051) | Clinical Trial Time Point Description | keep -> remove
# (0x0012, 0x0010), (0x0012, 0x0020) | Clinical Trial Sponsor Name/Protocol ID | replace -> remove
# (0x0012, 0x0021), (0x0012, 0x0030), (0x0012, 0x0031) | Clinical Trial Site .. | empty -> remove
# (0x0012, 0x0042) | Clinical Trial Subject Reading ID | replace -> remove
# (0x0010, 0x4000) | Patient Comments | remove -> replace
# (0x0040, 0x0009) | Scheduled Procedure Step ID | remove -> keep
# (0x0020, 0x4000) | Image Comments | remove -> replace
# (0x0018, 0x700C) | Date of Last Detector Calibration | incrementdate -> empty
# (0x0018, 0x700A) | Detector ID | remove -> empty
# ?? (0x0028, 0x0034) | Pixel Aspect Ratio | remove

