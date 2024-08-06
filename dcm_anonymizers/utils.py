import glob
import hashlib
import base64
from pathlib import Path
from datetime import datetime

from pydicom.tag import BaseTag


def ensure_dir(path: Path):
    return path.mkdir(parents=True, exist_ok=True)

def list_all_files(target: str, format: str = '.dcm'):
    targetdcm_path = f"{target}/*{format}"
    return glob.glob(targetdcm_path)


def int_tuple_to_basetag(tag: tuple):
    if isinstance(tag, BaseTag):
        return tag
    
    # Combine the group and element into a single integer
    combined_tag = (tag[0] << 16) + tag[1]
    return BaseTag(combined_tag)

def get_hashid(key: str, method: str = 'sha256', nchar: int =16):
    """
    Generate a hash identifier.
    :param key: input string for the hash algorithm.
    :param method: an algorithm name. Select from hashlib.algorithms_guaranteed 
    or hashlib.algorithms_available.
    :param nchar: number of the first character to return. Set 0 for all characters.
    :return: a string.
    """
    h = hashlib.new(method)
    h.update(key.encode())

    # for shake, the digest size is variable in length, let's just put it 32 bytes
    if h.digest_size == 0:
        hash_id = base64.b32encode(h.digest(32)).decode().replace("=", "")
    else:
        hash_id = base64.b32encode(h.digest()).decode().replace("=", "")

    if nchar > 0:
        hash_id = hash_id[:nchar]

    return hash_id

def parse_date_string(date_string):
    # Define possible formats
    date_formats = [
        "%Y%m%d%H%M%S",  # Full format with hours, minutes, and seconds
        "%Y%m%d%H%M%S.%f",
        "%Y%m%d"         # Format with only date
    ]
    
    # Try to parse the date string using the appropriate format
    for date_format in date_formats:
        try:
            return datetime.strptime(date_string, date_format)
        except ValueError:
            continue
    
    # If no format matches, raise an error
    raise ValueError(f"Date string '{date_string}' does not match any known format")