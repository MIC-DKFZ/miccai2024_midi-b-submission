import csv
import random
from os import listdir
from os.path import isfile, join
from pathlib import Path

import pydicom
import pandas as pd


def load_metadata(filename: str):
    return pd.read_csv(filename)

def load_dicoms_from_path(dicompath: str):
    alldicompaths = [f for f in listdir(dicompath) if isfile(join(dicompath, f))]
    alldicoms = []
    for dcm in alldicompaths:
        dcmpath = Path(dicompath, dcm)
        ds = pydicom.dcmread(dcmpath)
        alldicoms.append(ds)
    return alldicoms


class MIDIEvalDataLoader:
    
    def __init__(self, rawimagespath: str, deidimagespath: str, uidsmappath: str):
        self.rawimagespath: str = rawimagespath
        self.deidimagespath: str = deidimagespath
        self.uidsmappath: str = uidsmappath

        self.rawmetadata: Path = Path(self.rawimagespath, 'metadata.csv')
        self.deidmetadata: Path = Path(self.deidimagespath, 'metadata.csv')

        self.raw_2_deid = {}
        self.deid_2_raw = {}

        self.n_patients = 0

        self._map_uids(self.uidsmappath)


    def _map_uids(self, uidsmappath: str):
        linecount = 0

        with open(uidsmappath, newline='') as csvfile:
            mapreader = csv.reader(csvfile, delimiter=',', quotechar='|')            
            for row in mapreader:
                # ignore first line, column headers
                if linecount == 0:
                    linecount += 1
                    continue
                self.raw_2_deid[row[0]] = row[1]
                self.deid_2_raw[row[1]] = row[0]
                linecount += 1
        
        self.n_patients = linecount
        

    def _load_series_by_index(self, metadatafile: str, seriesidx: int):
        metadata = load_metadata(metadatafile)
        assert seriesidx < len(metadata), f"Patient index greater than available patient dicoms. Available patients {len(metadata)}"
        
        target_row = metadata.loc[seriesidx]
        targetdcmdir = Path(Path(metadatafile).parent, str(target_row['File Location']))
        alldicoms = load_dicoms_from_path(targetdcmdir)
        
        return alldicoms, target_row.to_dict()

    def _get_series_index_from_id(self, metadatafile: str, seriesid: str):
        metadata_df = load_metadata(metadatafile)
        indexes = metadata_df[metadata_df['Series UID'] == seriesid].index.tolist()
        if len(indexes) == 0:
            return -1
        else:
            return indexes[0]
    
    def load_raw_patient(self, idx: int = -1):
        if idx == -1:
            idx = random.randint(0, self.n_patients)
        
        return self._load_series_by_index(self.rawmetadata, idx)

    def load_deid_patient(self, idx: int = -1):
        if idx == -1:
            idx = random.randint(0, self.n_patients)
        
        return self._load_series_by_index(self.deidmetadata, idx)
    
    def get_raw_n_deid_patient(self, idx: int, include_metadata: bool = False):
        assert idx < self.n_patients, f"Patient index greater than available patient dicoms. Available patients {self.n_patients}"

        rawdicoms, rawmetadata = self.load_raw_patient(idx)
        print(rawmetadata)

        deid_series = self.raw_2_deid[rawmetadata['Series UID']]
        deid_idx = self._get_series_index_from_id(self.deidmetadata, deid_series)

        deiddicoms, deidmetadata = self.load_deid_patient(deid_idx)

        if include_metadata:
            return (rawdicoms, rawmetadata), (deiddicoms, deidmetadata)
        else:
            return rawdicoms, deiddicoms
    
    
