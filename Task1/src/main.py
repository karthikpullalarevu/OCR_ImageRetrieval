import os
import sys
import math
import re
import json
import uvicorn
from glob import glob
from typing import Dict
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from fastapi.responses import FileResponse
from tqdm import tqdm
import pytesseract
import requests
import yaml
from fastapi import FastAPI, Form, File, Response, BackgroundTasks, UploadFile
from utils import plot_side_by_side, parse_ocr_response, clean_and_correct_text, extract_entities_from_ocr, adjust_tessaract
# Load config
try:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    raise RuntimeError("config.yaml not found. Please ensure it exists.")

# Initialize FastAPI
app = FastAPI()
headers = {
    'appkey': config['appkey'],
    'appId': config['appid']
}
url = config['generic_ocr_url']

@app.get("/healthcheck")
def healthcheck() -> Dict:
    return {"statusCode": "200"}



@app.post("/predict/getJson")
def run_bot(image: UploadFile = File(...), ocr_engine: str = Form(...)):
    # Read image
    file_bytes = np.frombuffer(image.file.read(), np.uint8)
    if file_bytes.size == 0:
        return {"error": "Uploaded image is empty or corrupted."}

    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "Failed to decode the uploaded image. Please check the file format and content."}

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Adjust orientation
    angle, conf = adjust_tessaract(gray_image)
    if conf >= 3 and angle != 0:
        if angle == 90:
            gray_image = cv2.rotate(gray_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif angle == 270:
            gray_image = cv2.rotate(gray_image, cv2.ROTATE_90_CLOCKWISE)

    # Save the processed image
    output_dir = "/document"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{os.path.basename(image.filename)}.jpg")
    cv2.imwrite(output_path, gray_image)

    # Make OCR request
    payload = {'ocr': ocr_engine}
    files = [
        ('image', (os.path.basename(output_path), open(output_path, 'rb'), 'image/jpeg'))
    ]
    response = requests.post(url, headers=headers, data=payload, files=files).json()

    processed_details = parse_ocr_response(response)
    processed_details = extract_entities_from_ocr(processed_details)
    try:
        json_path = os.path.join(output_dir, f"{os.path.basename(image.filename.split('.')[0])}.json")
        with open(json_path, 'w') as f:
            json.dump(processed_details, f)
    except Exception as e:
        return {"error": str(e)}
    
    return FileResponse(json_path, media_type='application/json', filename=f'{os.path.basename(json_path)}')


@app.post("/predict/readDocument")
def run_bot(image: UploadFile = File(...), ocr_engine: str = Form(...)):
    # Read image
    file_bytes = np.frombuffer(image.file.read(), np.uint8)
    if file_bytes.size == 0:
        return {"error": "Uploaded image is empty or corrupted."}

    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "Failed to decode the uploaded image. Please check the file format and content."}

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Adjust orientation
    angle, conf = adjust_tessaract(gray_image)
    if conf >= 3 and angle != 0:
        if angle == 90:
            gray_image = cv2.rotate(gray_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif angle == 270:
            gray_image = cv2.rotate(gray_image, cv2.ROTATE_90_CLOCKWISE)

    # Save the processed image
    output_dir = "/document"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{os.path.basename(image.filename)}.jpg")
    cv2.imwrite(output_path, gray_image)

    # Make OCR request
    payload = {'ocr': ocr_engine}
    files = [
        ('image', (os.path.basename(output_path), open(output_path, 'rb'), 'image/jpeg'))
    ]
    response = requests.post(url, headers=headers, data=payload, files=files).json()

    processed_details = parse_ocr_response(response)
    processed_details = extract_entities_from_ocr(processed_details)
    # print(processed_details)
    # processed_details_cleaned = clean_and_correct_text(processed_details)
    try:
        json_path = os.path.join(output_dir, f"{os.path.basename(image.filename)}.json")
        with open(json_path, 'w') as f:
            json.dump(processed_details, f)
    except Exception as e:
        return {"error": str(e)}
    

    new_output_path = os.path.join(output_dir, f"{os.path.basename(image.filename)}_ocr.jpg")
    plot_side_by_side(output_path, processed_details, new_output_path)
    return FileResponse(new_output_path, media_type='image/jpeg', filename=f'{os.path.basename(output_path)}')


if __name__ == "__main__":
    PORT = os.environ.get("PORT", 6000)
    uvicorn.run(app, host="0.0.0.0", port=int(PORT))
