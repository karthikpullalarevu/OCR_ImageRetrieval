
        
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from symspellpy.symspellpy import SymSpell, Verbosity
import unicodedata
import re
import spacy

import pytesseract

def adjust_tessaract(image):
    info = pytesseract.image_to_osd(image, config="--psm 0")
    angle = int(re.search(r"Orientation in degrees: \d+", info).group().split(":")[-1].strip())
    conf = float(re.search(r"Orientation confidence: \d+\.\d+", info).group().split(":")[-1].strip())
    return angle, conf



def draw_ocr_boxes(image_path, details):
    """
    Draw bounding boxes and text on the original image.
    """
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    for detail in details:
        coordinates = detail['coordinates']
        x0, y0 = min(coordinates[0], coordinates[2]), min(coordinates[1], coordinates[3])
        x1, y1 = max(coordinates[0], coordinates[2]), max(coordinates[1], coordinates[3])
        text = detail['text']
        draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
    return image

def draw_text_boxes(image_size, details):
    """
    Create a blank image and add text matching the bounding box size and location.
    """
    text_image = Image.new("RGB", image_size, "white")
    draw = ImageDraw.Draw(text_image)
    for detail in details:
        coordinates = detail['coordinates']
        x0, y0 = min(coordinates[0], coordinates[2]), min(coordinates[1], coordinates[3])
        x1, y1 = max(coordinates[0], coordinates[2]), max(coordinates[1], coordinates[3])
        text = detail['text']
        bbox_width = x1 - x0
        bbox_height = y1 - y0
        font_size = min(bbox_width, bbox_height)  # Adjust font size to fit
        font = ImageFont.truetype("Arial.ttf", font_size)
        draw.text((x0, y0), text, fill="black", font=font)
    return text_image

def plot_side_by_side(image_path, details, output_path):
    """
    Plot the original image with OCR boxes and a blank text image side-by-side.
    """
    original_image = Image.open(image_path)
    image_with_boxes = draw_ocr_boxes(image_path, details)
    text_only_image = draw_text_boxes(original_image.size, details)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 10), dpi=300)  
    axes[0].imshow(image_with_boxes)
    axes[0].axis("off")
    axes[0].set_title("Original Image with text Boxes")
    axes[1].imshow(text_only_image)
    axes[1].axis("off")
    axes[1].set_title("OCR Extractions")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300) 



def parse_ocr_response(ocr_response):
    words_info = []
    result = ocr_response.get('result', [])
    if not result:
        return words_info
    details = result[0].get('details', [])
    for item in details:
        value = item.get('value', '')
        coordinates = item.get('coordinates', [])
        conf = item.get('conf', 0)

        words_info.append({
            'text': value,
            'coordinates': coordinates,
            'confidence': conf
        })
    return words_info


def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  
    text = text.strip().lower() 
    text = unicodedata.normalize('NFKD', text)  
    return text

def correct_text(sym_spell, text):
    corrected_text = sym_spell.lookup_compound(text, max_edit_distance=2)
    return corrected_text[0].term if corrected_text else text

def clean_and_correct_text(words_info):
    
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    sym_spell.load_dictionary("frequency_dictionary.txt", term_index=0, count_index=1)

    cleaned_and_corrected_text = [correct_text(sym_spell,clean_text(item['text'])) for item in words_info]
    return cleaned_and_corrected_text

def extract_entities_and_pos(model, text):
   
    doc = model(text)
    
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    pos_tags = [(token.text, token.pos_) for token in doc]

    return entities, pos_tags

def extract_entities_from_ocr(ocr_response):
  
    new_ocr_response = []
    model = spacy.load("en_core_web_sm")
    for word_info in ocr_response:
        text = word_info['text']
        # Extract entities from the text
        entities,pos = extract_entities_and_pos(model, text)
        word_info['entities'] = entities
        word_info['pos'] = pos
        new_ocr_response.append(word_info)
    
    
    return new_ocr_response

