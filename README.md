# MIC-DICOM-Deidentifier

# A Hybrid AI and Rule-based Approach to DICOM De-identification: A Solution for the MIDI-B Challenge at MICCAI 2024

## Abstract
This project presents a robust de-identification system for DICOM files, developed for the MIDI-B challenge at MICCAI 2024. The solution integrates AI-based and rule-based techniques to de-identify Protected Health Information (PHI) within DICOM metadata and image data, achieving near-perfect accuracy. Through iterative improvements, our approach ensures privacy while preserving image utility, achieving an accuracy of 99.91% on the test dataset.

## Keywords
DICOM De-identification, AI-based de-identification, Rule-based de-identification, Protected Health Information (PHI), RoBERTa model, I2B2 2014 dataset, PaddleOCR

## Introduction
This project addresses the challenge of DICOM file de-identification, crucial for protecting patient privacy in medical imaging. Leveraging the MIDI-B challenge framework, we built a system that removes PHI from both metadata and image data, based on the DICOM PS 3.15 confidentiality guidelines and the Safe Harbor method by TCIA.

## Download the Data
To use this system, download the dataset from The Cancer Imaging Archive (TCIA) using [this link](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=80969777). Place the raw images under an `images` directory and the de-identified images under an `images-2` directory.

## Installation
Install the necessary dependencies using `poetry`:
```bash
poetry install --no-root
```

### Running Jupyter Notebooks
To run Jupyter notebooks, follow these steps:
1. Install Jupyter Notebook globally:
   ```bash
   pip install jupyter
   ```
2. Add the current Poetry environment to the Jupyter notebook kernels:
   ```bash
   poetry run python -m ipykernel install --user --name dcm-deid
   ```
3. Start the Jupyter notebook:
   ```bash
   jupyter notebook
   ```
4. Open the notebooks in your browser and select the `dcm-deid` kernel. Ensure it is selected for running the notebooks, especially if another kernel is chosen by default.

## Dataset Directory
Download the required dataset from TCIA as mentioned above. Ensure the data is organized as follows:
```
dataset_directory/
├── images/         # Raw DICOM images
└── images-2/       # De-identified DICOM images
```

## Usage
To de-identify DICOM files, execute the following command:
```bash
python deidentify.py --input_dir <path_to_dicom_files> --output_dir <output_path>
```

## Results
Our hybrid method achieved a final accuracy of 99.91% on the MIDI-B test dataset, with specific improvements in handling private tags and reducing false positives.

## Acknowledgements
Supported by the Helmholtz Metadata Collaboration, Hub Health, and the RACOON project in NUM 2.0.

