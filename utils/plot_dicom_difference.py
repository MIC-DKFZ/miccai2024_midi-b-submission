"""
=======================================
Analyse differences between DICOM files
=======================================

This examples illustrates how to find the differences between two DICOM files.

"""

# authors : Guillaume Lemaitre <g.lemaitre58@gmail.com>
# license : MIT

import difflib
from pathlib import Path
import pydicom
from pydicom.data import get_testdata_file

print(__doc__)

filename_mr = Path(r"C:\src\midi_b_challange\data\validation_data\input_data\31780971\2.5.698.0.0.5917906.8.541.1019580938801511034\2.5.698.0.0.5917906.8.541.3398391956045238764\00000002.dcm")
filename_ct = Path(r"C:\src\midi_b_challange\data\validation_data\input_data\31780971\2.5.698.0.0.5917906.8.541.1019580938801511034\2.5.698.0.0.5917906.8.541.3398391956045238764\00000036.dcm")

datasets = tuple([pydicom.dcmread(filename, force=True)
                  for filename in (filename_mr, filename_ct)])

# difflib compare functions require a list of lines, each terminated with
# newline character massage the string representation of each dicom dataset
# into this form:
rep = []
for dataset in datasets:
    lines = str(dataset).split("\n")
    lines = [line + "\n" for line in lines]  # add the newline to end
    rep.append(lines)


diff = difflib.Differ()
for line in diff.compare(rep[0], rep[1]):
    if line[0] != "?" and line[0] != " ":
        print(line)
