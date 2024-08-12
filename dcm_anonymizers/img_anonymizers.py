import os
import pydicom
from paddleocr import PaddleOCR
import cv2
import numpy as np
import logging

from dcm_anonymizers.phi_detectors import DcmPHIDetector


class DCMImageAnonymizer:
    def __init__(self,  phi_detector: DcmPHIDetector, use_gpu: bool = True) -> None:
        self.ocr = None
        self.detector = phi_detector
        self.change_log = {} # series_id: []
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        self._init_ocr(use_gpu)
        
    def _init_ocr(self, use_gpu):
        self.ocr = PaddleOCR(
            use_angle_cls=True, lang='en', use_gpu=use_gpu, show_log=False
        )

    def normalize_pixel_arr(self, pxl_arr: np.ndarray):
        # Normalize the pixel values to the range 0-255
        pxl_arr = cv2.normalize(pxl_arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Ensure pixel_array is a 3-channel image (required for color drawing)
        if len(pxl_arr.shape) == 2:
            try:
                pxl_arr = cv2.cvtColor(pxl_arr, cv2.COLOR_GRAY2BGR)
            except Exception as e:
                print(e)

        return pxl_arr

    def extract_from_pixel_array(self, img_arr: np.ndarray):
        # img_arr = self.pixel_arr_to_rgb(img_arr)
        result = self.ocr.ocr(img_arr, cls=True)
        return result
    
    # Function to draw polygons
    def draw_filled_polygons(self, img, polygons: list, color=(255, 255, 255)):
        for poly in polygons:
            # Convert the polygon to a NumPy array of int32 type
            pts = np.array(poly, dtype=np.int32)
            pts = pts.reshape((-1, 1, 2))
            
            # Fill the polygon
            img = cv2.fillPoly(img, [pts], color=color)

            # Optionally draw the polygon border
            img = cv2.polylines(img, [pts], isClosed=True, color=color, thickness=5)

        return img

    
    def anonymize_dicom_image_data(self, ds: pydicom.Dataset): 
        try:
            pixel_shape = ds.pixel_array.shape
        except AttributeError as e:
            self.logger.warning(
                str(e)
            )
            return False
        
        if len(pixel_shape) > 3:
            self.logger.warning(
                "DICOM pixel array found with shape {} of Modality {}".format(str(ds.pixel_array.shape), ds.Modality)
            )
            return False
        elif len(pixel_shape) == 3 and pixel_shape[2] != 3:
            self.logger.warning(
                "DICOM pixel array found with shape {} of Modality {}".format(str(ds.pixel_array.shape), ds.Modality)
            )
            return False

        normalized_array = self.normalize_pixel_arr(ds.pixel_array)

        extracted = self.extract_from_pixel_array(normalized_array)
        detected_polygons = []
        updated = False
        if extracted[0] and len(extracted[0]) > 0:
            for e in extracted[0]:
                bbox = e[0]
                text = e[1][0]

                detected_phi = self.detector.detect_entities(text)
                if len(detected_phi) > 0:
                    extracted_txt_list = [entity[0] for entity in detected_phi]
                    extracted_text = "".join(extracted_txt_list)
                    # IGNORE, if extracted text is just 2 character
                    if len(extracted_text) <= 2:
                        continue
                    detected_polygons.append(bbox)
            
            if len(detected_polygons) > 0:
                img = ds.pixel_array.copy()        
                drawn_img = self.draw_filled_polygons(img, detected_polygons)

                ds.PixelData = drawn_img
                updated = True
        
        return updated

    def anonymize_dicom_file(self, dcm_file: str, out_file: str):
        dataset = pydicom.dcmread(dcm_file)
        seriesuid_tag = (0x0020, 0x000E)

        changed = self.anonymize_dicom_image_data(dataset)
        if changed:
            # print(f"Dicom image updated: {dcm_file}")
            seriesuid_element = dataset.get(seriesuid_tag)
            seriesuid = seriesuid_element.value
            basename = os.path.basename(dcm_file)
            if seriesuid in self.change_log:
                self.change_log[seriesuid].append(basename)
            else:
                self.change_log[seriesuid] = [basename]


            # Store modified image
            dataset.save_as(out_file)