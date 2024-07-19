from pathlib import Path
from ast import literal_eval
import pydicom
import json
from dicomanonymizer import anonymize, simpledicomanonymizer
from dicomanonymizer.simpledicomanonymizer import (
    replace, replace_UID, empty_or_replace, delete_or_replace,
    delete_or_empty_or_replace, delete_or_empty_or_replace_UID
)

from dicomanonymizer.format_tag import tag_to_hex_strings

from dicomanonymizer.dicomfields import (
    D_TAGS, U_TAGS, Z_D_TAGS, X_D_TAGS, X_Z_D_TAGS, X_Z_U_STAR_TAGS
)


DEID_DATASET_ROOT = '/home/r079a/Desktop/de-identification/dataset'


def ensure_dir(path: Path):
    return path.mkdir(parents=True, exist_ok=True)

def load_ps3_tags():
    json_path = './docs/ps3.3_profile_attrs.json'

    tags = {}    
    with open(json_path) as f:
        tags = json.load(f)
    
    for tag in tags:
        items = tags[tag]
        tags[tag] = [literal_eval(i) for i in items]

    return tags


def custom_init_actions():
    """
    Initialize anonymization actions with DICOM standard values

    :return Dict object which map actions to tags
    """

    ps3_tags = load_ps3_tags()

    # anonymization_actions = {tag: replace for tag in D_TAGS}
    anonymization_actions = {tag: replace_UID for tag in ps3_tags['UID_TAGS']}
    anonymization_actions.update({tag: empty_or_replace for tag in ps3_tags['Z_D_TAGS']})
    anonymization_actions.update({tag: delete_or_replace for tag in ps3_tags['X_D_TAGS']})
    anonymization_actions.update(
        {tag: delete_or_empty_or_replace for tag in ps3_tags['X_Z_D_TAGS']}
    )
    anonymization_actions.update(
        {tag: delete_or_empty_or_replace_UID for tag in ps3_tags['X_Z_U_STAR_TAGS']}
    )
    return anonymization_actions

def anonymize_dataset(
    dataset: pydicom.Dataset,
    extra_anonymization_rules: dict = None,
    delete_private_tags: bool = True,
) -> None:
    """
    Anonymize a pydicom Dataset by using anonymization rules which links an action to a tag

    :param dataset: Dataset to be anonymize
    :param extra_anonymization_rules: Rules to be applied on the dataset
    :param delete_private_tags: Define if private tags should be delete or not
    """
    current_anonymization_actions = custom_init_actions()

    if extra_anonymization_rules is not None:
        current_anonymization_actions.update(extra_anonymization_rules)
    
    private_tags = []

    action_history = {}

    for tag, action in current_anonymization_actions.items():
        # print(tag, action)

        def range_callback(dataset, data_element):
            if (
                data_element.tag.group & tag[2] == tag[0]
                and data_element.tag.element & tag[3] == tag[1]
            ):
                action(dataset, (data_element.tag.group, data_element.tag.element))

        element = None
        # We are in a repeating group
        if len(tag) > 2:
            dataset.walk(range_callback)
        # Individual Tags
        else:
            try:
                element = dataset.get(tag)
                if element:
                    earliervalue = element.value
            except KeyError:
                print("Cannot get element from tag: ", tag_to_hex_strings(tag))

            if tag[0] == 0x0002:
                if not hasattr(dataset, "file_meta"):
                    continue
                # Apply rule to meta information header
                action(dataset.file_meta, tag)
            else:
                action(dataset, tag)            
            
            if element:
                if earliervalue != element.value:
                    action_history[element.tag] = action.__name__

            # Get private tag to restore it later
            # if element and element.tag.is_private:
            #    private_tags.append(get_private_tag(dataset, tag))

    print(action_history)

    # X - Private tags = (0xgggg, 0xeeee) where 0xgggg is odd
    if delete_private_tags:
        dataset.remove_private_tags()

        # Adding back private tags if specified in dictionary
        for privateTag in private_tags:
            creator = privateTag["creator"]
            element = privateTag["element"]
            block = dataset.private_block(
                creator["tagGroup"], creator["creatorName"], create=True
            )
            if element is not None:
                block.add_new(
                    element["offset"], element["element"].VR, element["element"].value
                )


if __name__ == "__main__":
    sample_img_path = Path(DEID_DATASET_ROOT, 'images/manifest-1617826555824/Pseudo-PHI-DICOM-Data/292821506/07-13-2013-NA-XR CHEST AP PORTABLE for Douglas Davidson-46198/1002.000000-NA-53238')
    sample_out_path = Path('/home/r079a/Desktop/de-identification/dicom-output', '292821506/07-13-2013-NA-XR CHEST AP PORTABLE for Douglas Davidson-46198/1002.000000-NA-53238')
    
    ensure_dir(sample_out_path)

    simpledicomanonymizer.anonymize_dataset = anonymize_dataset

    anonymize(
        input_path=str(sample_img_path),
        output_path=str(sample_out_path),
        anonymization_actions={},
        delete_private_tags=False,
    )
