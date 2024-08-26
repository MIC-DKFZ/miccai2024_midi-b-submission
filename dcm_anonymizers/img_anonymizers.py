import os
import pydicom
from paddleocr import PaddleOCR
import cv2
import numpy as np
import logging

from dcm_anonymizers.phi_detectors import DcmPHIDetector

# Function to calculate Euclidean distance between two points
def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

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
    
    def get_polygons_color(self, img, polygon):
        cX, cY = img.shape[0]/2, img.shape[1]/2

        polygon_points = np.array(polygon, np.int32)

        closest_dist = 9999999
        selected_point = polygon_points[0]

        for point in polygon_points:
            distance = euclidean_distance([cX, cY], point)
            if distance < closest_dist:
                selected_point = point

        pX, pY = selected_point

        color = img[pY, pX]
        if isinstance(color, list):
            color = [int(c) for c in color]
            color = tuple(color)
        else:
            color = int(color)

        return color
    
    # Function to draw polygons
    def draw_filled_polygons(self, img, polygons: list, color=None):
        for poly in polygons:
            color = self.get_polygons_color(img, poly)

            # print(poly)
            # print(color)

            # Convert the polygon to a NumPy array of int32 type
            pts = np.array(poly, dtype=np.int32)
            pts = pts.reshape((-1, 1, 2))
            
            # Fill the polygon
            img = cv2.fillPoly(img, [pts], color=color)

            # Optionally draw the polygon border
            img = cv2.polylines(img, [pts], isClosed=True, color=color, thickness=5)

        return img
    
    def extract_texts_as_note(self, pxl_arr: np.ndarray, normalize_arr: bool = True):
        texts = ''
        bbox_map = {}
        text_position_map = {}
        if normalize_arr:
            pxl_arr = self.normalize_pixel_arr(pxl_arr)
        extracted = self.extract_from_pixel_array(pxl_arr)

        if extracted[0] and len(extracted[0]) > 0:           
            for e in extracted[0]:
                bbox = e[0]
                text = e[1][0]

                start = len(texts)
                texts += f"{text}\n"
                end = len(texts)

                bbox_map[text] = bbox
                text_position_map[text] = (start, end)
        
        return texts, bbox_map, text_position_map

    def get_associated_bounding_boxes(self, entities: list, bbox_map, text_position_map):
        selected_bbox = []
        added_bbox = []
        for entiity in entities:
            start = entiity[2]
            for text in text_position_map.keys():
                txt_start, txt_end = text_position_map[text]
                if start >= txt_start and start < txt_end:
                    bbox = bbox_map[text]
                    if text not in added_bbox:
                        selected_bbox.append(bbox)
                        added_bbox.append(text)
                    break
        return selected_bbox

    def anonymize_single_slice(self, img_arr: np.ndarray):
        # normalized = self.normalize_pixel_arr(img_arr)
        extracted_note, bbox_map, text_position_map = self.extract_texts_as_note(img_arr)
        detected_polygons = []
        updated = False

        ## Earlier Implementation
        # detected_phi = self.detector.detect_entities(text)
        # if len(detected_phi) > 0:
        #     extracted_txt_list = [entity[0] for entity in detected_phi]
        #     extracted_text = "".join(extracted_txt_list)
        #     # IGNORE, if extracted text is just 2 character
        #     if len(extracted_text) <= 2:
        #         continue
        #     detected_polygons.append(bbox)
        if extracted_note != '':                              
            # might contain \n inside the entity and 
            # next bbox might be overlooked
            detected_phi = self.detector.detect_entities(extracted_note)
            detected_polygons = self.get_associated_bounding_boxes(detected_phi, bbox_map, text_position_map)

            
        if len(detected_polygons) > 0:
            # img = ds.pixel_array.copy()       
            img_arr = self.draw_filled_polygons(img_arr, detected_polygons)
            updated = True
        
        return img_arr, updated
    
    def process_3d_image_array(self, img_arr):
        img_shape = img_arr.shape
        if img_shape[2] == 3:
            return self.anonymize_single_slice(img_arr)

        updated = False
        for i in range(img_shape[0]):
            img_arr[i], isupdated = self.anonymize_single_slice(img_arr[i])
            if isupdated and not updated:
                updated = isupdated

        return img_arr, updated

    
    def anonymize_dicom_image_data(self, ds: pydicom.Dataset):
        pixel_data = ds.get("pixel_array")
        if pixel_data is None:
            return False
        
        pixel_shape = pixel_data.shape        
        if len(pixel_shape) > 3:
            self.logger.warning(
                "Unhandled Shaped DICOM pixel array with shape {} of Modality {}".format(str(ds.pixel_array.shape), ds.Modality)
            )
            return False
        
        img = ds.pixel_array.copy()
        if len(pixel_shape) == 3:
            drawn_img, updated = self.process_3d_image_array(img)
        else:
            drawn_img, updated = self.anonymize_single_slice(img)
            
        if updated > 0:
            ds.PixelData = drawn_img
        
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