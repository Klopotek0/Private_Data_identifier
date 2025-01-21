import easyocr

def recognize_text_with_easyocr(image_path):
    """
    Recognizes text in an image using EasyOCR.

    Args:
        image_path (str): Path to the input image.

    Returns:
        list: Detected texts along with bounding boxes and confidence scores.
    """
    reader = easyocr.Reader(['en', 'pl'])  #
    results = reader.readtext(image_path, text_threshold=0.1,low_text=0.3)

    detected_texts = []
    for (bbox, text, confidence) in results:
        detected_texts.append({'text': text, 'confidence': confidence, 'bbox': bbox})
    return detected_texts
