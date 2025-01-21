import cv2
import numpy as np
from ultralytics import YOLO
from ocr import recognize_text_with_easyocr
import matplotlib.pyplot as plt

def preprocess_image(image):
    """
    Applies preprocessing techniques to enhance text visibility for OCR.

    :param image: Cropped image with text.
    :return: Preprocessed image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(gray, -1, sharpen_kernel)


    return sharpened


def detect_and_recognize_text(image_path, model_path, zoom_factor=1.5):
    """
    Wykrywa obszary tekstu za pomocą YOLO i rozpoznaje ich treść za pomocą EasyOCR.
    Każdy wykryty obszar tekstu jest powiększany przed wysłaniem do OCR.

    :param image_path: Ścieżka do obrazu wejściowego.
    :param model_path: Ścieżka do wytrenowanego modelu YOLO.
    :param zoom_factor: Współczynnik powiększenia wyciętych regionów (default 1.5).
    :return: Lista wykrytych tekstów oraz obraz z zaznaczonymi obszarami.
    """
    # Wczytaj model YOLO
    model = YOLO(model_path)
    
    # Wczytaj obraz
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Nie znaleziono obrazu: {image_path}")

    # Konwersja do RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(image_rgb)

    boxes = results[0].boxes.xyxy  
    
    detected_texts = []
    
    h, w, _ = image.shape

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)

        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

        new_width = int((x2 - x1) * zoom_factor)
        new_height = int((y2 - y1) * zoom_factor)

        new_x1 = max(0, center_x - new_width // 2)
        new_y1 = max(0, center_y - new_height // 2)
        new_x2 = min(w, center_x + new_width // 2)
        new_y2 = min(h, center_y + new_height // 2)

        cropped = image[new_y1:new_y2, new_x1:new_x2]

        cropped=preprocess_image(cropped)

        text_results = recognize_text_with_easyocr(cropped)

        for item in text_results:
            detected_texts.append({'text': item['text'], 'confidence': item['confidence'], 'bbox': (new_x1, new_y1, new_x2, new_y2)})

    return detected_texts, image
