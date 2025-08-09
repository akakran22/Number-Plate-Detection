import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Suppress GPU warnings

import cv2
import numpy as np
import imutils
import easyocr
import matplotlib.pyplot as plt

class LicensePlateDetector:
    def __init__(self):
        self.reader = easyocr.Reader(['en'])
        self.MIN_CONTOUR_AREA = 500
        self.MAX_CANDIDATES = 15

    def preprocess_image(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
        return gray, bfilter

    def detect_edges(self, filtered_img):
        edged = cv2.Canny(filtered_img, 30, 200)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
        return edged

    def find_plate_candidates(self, edged):
        keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(keypoints)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:self.MAX_CANDIDATES]
        candidates = []
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.018 * peri, True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / float(h)
                if 2.0 <= aspect_ratio <= 5.5:
                    candidates.append({
                        'contour': approx,
                        'bbox': (x, y, w, h),
                        'area': cv2.contourArea(contour),
                        'aspect_ratio': aspect_ratio
                    })
        return candidates

    def validate_with_ocr(self, gray_img, candidate):
        approx = candidate['contour']
        mask = np.zeros(gray_img.shape, np.uint8)
        cv2.drawContours(mask, [approx], 0, 255, -1)
        (x, y) = np.where(mask == 255)
        if len(x) == 0 or len(y) == 0:
            return None, ""
        (x1, y1), (x2, y2) = (np.min(x), np.min(y)), (np.max(x), np.max(y))
        cropped = gray_img[x1:x2+1, y1:y2+1]
        if cropped.shape[0] < 20 or cropped.shape[1] < 60:
            return None, ""
        cropped = self.preprocess_for_ocr(cropped)
        try:
            result = self.reader.readtext(cropped)
            if result:
                text = result[0][-2]
                if self.is_valid_plate_text(text):
                    return approx, text
        except:
            pass
        return None, ""

    def preprocess_for_ocr(self, img):
        if img.shape[1] < 200:
            scale = 200 / img.shape[1]
            img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)), interpolation=cv2.INTER_CUBIC)
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img = cv2.medianBlur(img, 3)
        return img

    def is_valid_plate_text(self, text):
        clean_text = ''.join(c for c in text if c.isalnum())
        if len(clean_text) >= 4:
            return any(c.isalpha() for c in clean_text) or any(c.isdigit() for c in clean_text)
        return False

    def detect_alternative_method(self, img, gray, edged):
        edged2 = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edged2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        height, width = img.shape[:2]
        min_area, max_area = (width * height) * 0.001, (width * height) * 0.05
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h)
                if 2.0 <= aspect_ratio <= 5.5:
                    rect_area = w * h
                    extent = float(area) / rect_area
                    if extent > 0.5:
                        valid_contours.append((x, y, w, h))
        return valid_contours

    def detect(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Cannot read image from {img_path}")
        original = img.copy()
        detected_plates = []
        gray, bfilter = self.preprocess_image(img)
        edged = self.detect_edges(bfilter)
        candidates = self.find_plate_candidates(edged)

        for candidate in candidates:
            location, text = self.validate_with_ocr(gray, candidate)
            if location is not None:
                detected_plates.append({
                    'location': location,
                    'text': text,
                    'bbox': candidate['bbox']
                })

        if len(detected_plates) == 0:
            alt_boxes = self.detect_alternative_method(img, gray, edged)
            for bbox in alt_boxes:
                x, y, w, h = bbox
                approx = np.array([[[x, y]], [[x + w, y]], [[x + w, y + h]], [[x, y + h]]])
                cropped = gray[y:y+h, x:x+w]
                cropped = self.preprocess_for_ocr(cropped)
                try:
                    result = self.reader.readtext(cropped)
                    if result:
                        text = result[0][-2]
                        if self.is_valid_plate_text(text):
                            detected_plates.append({
                                'location': approx,
                                'text': text,
                                'bbox': bbox
                            })
                except:
                    pass

        output = original.copy()
        for plate in detected_plates:
            cv2.drawContours(output, [plate['location']], -1, (0, 255, 0), 3)
            x, y, _, _ = plate['bbox']
            cv2.putText(output, plate['text'], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return output, detected_plates

def main():
    detector = LicensePlateDetector()
    output_img, plates = detector.detect('img.png')

    # Save to file
    cv2.imwrite('output.jpg', output_img)
    print(f"Output saved as output.jpg. {len(plates)} plate(s) detected.")

    # Display using matplotlib
    plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
