import os.path
import zipfile
import requests 
from io import BytesIO

scriptdir = os.path.dirname(os.path.realpath(__file__))
deid_dir = os.path.join(scriptdir, "deid")
raw_dir = os.path.join(scriptdir, "raw")

if not os.path.exists(deid_dir):
    os.makedirs(deid_dir)

if not os.path.exists(raw_dir):
    os.makedirs(raw_dir)

with open(os.path.join(scriptdir, "Pseudo-Phi-DICOM-Evaluation-dataset-April-7-2021.tcia"), "r") as manifest_raw:
    series_ids_raw = [id.strip() for id in manifest_raw.readlines()[6:]]
    for id in series_ids_raw:
        zip_file_url = f"https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage?SeriesInstanceUID={id}"
        r = requests.get(zip_file_url, stream=True)
        z = zipfile.ZipFile(BytesIO(r.content))
        z.extractall(path = raw_dir)


with open(os.path.join(scriptdir, "Pseudo-PHI-DICOM-De-id-Evaluation-dataset-April-7-2021.tcia"), "r") as manifest_deid:
    series_ids_deid = [id.strip() for id in manifest_deid.readlines()[6:]]
    for id in series_ids_deid:
        zip_file_url = f"https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage?SeriesInstanceUID={id}"
        r = requests.get(zip_file_url, stream=True)
        z = zipfile.ZipFile(BytesIO(r.content))
        z.extractall(path = deid_dir)


