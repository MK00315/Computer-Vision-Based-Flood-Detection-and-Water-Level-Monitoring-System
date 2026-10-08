# SPDX-License-Identifier: AGPL-3.0-only

"""Google Colab source export; use the corresponding notebook to run it."""


# # Computer Vision-Based Flood Detection and Water-Level Monitoring System
# ## Team 20 — interactive image demo
# 
# 2023BCS0235 Mahendra K  
# 2023BCS0232 Sanin K  
# 2023BCS0238 Bhukya Gandhi  
# 2022BCD0001 Ankit Subhash
# 
# This is a separate presentation demo for our completed project. It puts an image-upload page around the existing car detector and flood-level classifier. It does not replace the submitted Task 7 notebook and does not retrain either model.
# 
# ### Before running
# 
# 1. Upload this notebook to Google Colab. Start a fresh runtime and select **Runtime → Change runtime type → T4 GPU** if it is available.
# 2. Keep the trained Task 4 checkpoint in `MyDrive/CSE411_Flood_Project/task4_outputs/resnet50_sturm_best_macro_f1.pth`. If yours is elsewhere, change `CHECKPOINT_PATH` in Section 3.
# 3. Existing full-scene images in `MyDrive/CSE411_Flood_Project/task5_scene_images` can be used as examples. This folder is optional: the page also accepts new JPG/PNG images.
# 4. Run Sections 1–8 in order. Approve the Drive-mount prompt. The optional backup and stop cells do nothing by default.
# 5. Section 8 prints a temporary link, username and password. Open the link, sign in, upload or load an image, and click **Analyse image**.
# 
# We do not need to upload the STURM dataset again for this demo. The trained checkpoint is required. The first run needs internet access to install packages and download the pretrained YOLOv8n weights.
# 
# ### What the page demonstrates
# 
# Full-scene image → pretrained YOLOv8n car boxes → padded car crops → trained ResNet-50 → relative labels and annotated output.
# 
# The classes are `Level0`–`Level4`, following the classifier checkpoint. They are relative categories, not water depth in centimetres or metres. Only the COCO **car** class is processed. Trucks, buses, flood segmentation, video monitoring and road-safety decisions are outside this demo. Its confidence scores are model scores, not guaranteed probabilities of a correct or safe result.
# 
# This notebook is supplied without saved execution outputs. Run it in Colab to obtain your own demonstration results.


# ## 1. Install the demo packages
# 
# Use a fresh Colab runtime. We retain Colab's supplied PyTorch/torchvision pair rather than replacing them individually. Do not uninstall Pillow or install the old `Pillow==9.5.0` workaround.
# 
# If Colab explicitly asks for a runtime restart after installation, restart it once and continue from Section 2. If an old Pillow import error persists, disconnect and delete that runtime, then start this notebook in a fresh one.


# Install the notebook dependencies before running this source.


# ## 2. Connect Google Drive
# 
# Drive is used to read our own trained checkpoint and optional scene images. The demo page does not expose the Drive folder or let visitors upload model files.


from google.colab import drive
drive.mount('/content/drive')


# ## 3. Configuration and preflight
# 
# Usually only the two Drive paths need checking. The baseline is detector confidence 0.30, NMS IoU 0.45 and detector image size 640. A crop is expanded by 20% of the box width on each horizontal side and 20% of its height on each vertical side, clipped to the image boundary. Classifier scores below 0.50 are marked uncertain.
# 
# Results are written into a new local session folder, separate from the Task 4–7 outputs. Local files disappear when the Colab runtime is deleted; download them from the page or use the optional backup cell before ending the session.


from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import secrets
import shutil
import time
import traceback
import uuid

PROJECT_DIR = Path('/content/drive/MyDrive/CSE411_Flood_Project')
CHECKPOINT_PATH = PROJECT_DIR / 'task4_outputs' / 'resnet50_sturm_best_macro_f1.pth'
SCENE_DIR = PROJECT_DIR / 'task5_scene_images'  # Optional examples, not required for uploads.

DETECTOR_CONFIDENCE = 0.30
DETECTOR_IOU = 0.45
DETECTOR_IMAGE_SIZE = 640
CLASSIFIER_CONFIDENCE = 0.50
CROP_PADDING_FRACTION = 0.20
MAX_IMAGE_PIXELS = 20_000_000
MAX_UPLOAD_SIZE = '25mb'
MAX_SAMPLE_IMAGES = 12

SESSION_ID = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC') + '_' + uuid.uuid4().hex[:8]
SESSION_DIR = Path('/content/cse411_gradio_demo_files') / SESSION_ID
RESULTS_DIR = SESSION_DIR / 'results'
SAMPLES_DIR = SESSION_DIR / 'samples'
for directory in (RESULTS_DIR, SAMPLES_DIR, SESSION_DIR / 'cache'):
    directory.mkdir(parents=True, exist_ok=True)

# Set these before importing Gradio. Only local result files are downloadable.
os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
os.environ['GRADIO_TEMP_DIR'] = str(SESSION_DIR / 'cache')

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw, ImageFont
import torch
import torch.nn as nn
from torchvision import models, transforms
import ultralytics
from ultralytics import YOLO
import gradio as gr

if not CHECKPOINT_PATH.is_file():
    raise FileNotFoundError(
        f'Our trained Task 4 checkpoint was not found:\n{CHECKPOINT_PATH}\n'
        'Check the Drive path above before continuing. This demo does not train a new classifier.'
    )

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DETECTOR_DEVICE = 0 if device.type == 'cuda' else 'cpu'
print('Device:', device)
if device.type != 'cuda':
    print('GPU is not active. The demo can run on CPU, but image analysis will be slower.')
print('Checkpoint:', CHECKPOINT_PATH)
print('Optional example folder:', SCENE_DIR)
print('Local result folder:', RESULTS_DIR)
print('Gradio:', gr.__version__, '| Ultralytics:', ultralytics.__version__)


# ## 4. Load the existing models
# 
# ResNet-50 uses our selected Task 4 weights and saved preprocessing metadata. YOLOv8n uses its existing COCO-pretrained weights; STURM was not used to train YOLO. Model loading happens once, not for every image.
# 
# Only load the checkpoint produced by our own trusted training notebook. PyTorch checkpoint files can contain executable serialized objects; do not replace this path with an unknown downloaded `.pth` file.


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
required_keys = {'model_state_dict', 'class_names', 'image_size', 'imagenet_mean', 'imagenet_std'}
missing_keys = required_keys - set(checkpoint)
if missing_keys:
    raise KeyError(f'Checkpoint metadata is incomplete: {sorted(missing_keys)}')

class_names = list(checkpoint['class_names'])
image_size = int(checkpoint['image_size'])
imagenet_mean = tuple(float(value) for value in checkpoint['imagenet_mean'])
imagenet_std = tuple(float(value) for value in checkpoint['imagenet_std'])
if class_names != [f'Level{i}' for i in range(5)] or image_size != 224:
    raise ValueError('The checkpoint must use Level0–Level4, in that order, and 224 × 224 inputs.')
if (len(imagenet_mean) != 3 or len(imagenet_std) != 3
        or not all(math.isfinite(value) for value in imagenet_mean + imagenet_std)
        or not all(value > 0 for value in imagenet_std)):
    raise ValueError('Invalid RGB normalization metadata in the checkpoint.')

classifier_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])
classifier = models.resnet50(weights=None)
classifier.fc = nn.Linear(classifier.fc.in_features, len(class_names))
classifier.load_state_dict(checkpoint['model_state_dict'], strict=True)
classifier = classifier.to(device).eval()
CHECKPOINT_SHA256 = file_sha256(CHECKPOINT_PATH)
del checkpoint

detector = YOLO('yolov8n.pt')
if detector.names.get(2) != 'car':
    raise ValueError('The detector must use COCO class 2 = car.')
print('Ready: pretrained YOLOv8n + our trained ResNet-50.')
print('Classifier classes:', ', '.join(class_names))
print('No training is performed in this notebook.')


# ## 5. Image processing and optional examples
# 
# For each detected car, the classifier sees a crop taken from the original image, not from an image with boxes already drawn on it. The crop gallery shows that padded crop before resizing; the model receives its 224 × 224 normalized version.
# 
# The optional examples are copied into the local session folder. Known Task 5 source credits are kept with those examples and included in the result metadata. For a new upload, supply its source credit yourself before including it in a report or distributing it. Do not remove a source image's licence requirements from an annotated copy.


CREDIT_RECORDS = [('Cars_driving_through', 'TCExplorer', 'CC BY-SA 2.0', 'https://commons.wikimedia.org/wiki/File:Cars_driving_through_flood_water_on_the_B4380_near_Atcham_-_geograph.org.uk_-_7390621.jpg'), ('Cars_in_flooded', 'Jiří Sedláček', 'CC BY-SA 4.0', 'https://commons.wikimedia.org/wiki/File:Cars_in_flooded_Novodvorsk%C3%A1_street_in_T%C5%99eb%C3%AD%C4%8D%2C_T%C5%99eb%C3%AD%C4%8D_District.JPG'), ('FEMA_-_32048', 'Marvin Nauman/FEMA', 'Public domain in the US', 'https://commons.wikimedia.org/wiki/File:FEMA_-_32048_-_Red_car_floating_in_flood_waters_in_Oklahoma.jpg'), ('FEMA_-_32096', 'Marvin Nauman/FEMA', 'Public domain in the US', 'https://commons.wikimedia.org/wiki/File:FEMA_-_32096_-_Cars_drives_though_flooded_street_in_an_Oklahoma_neighborhood.jpg'), ('Flooded_street_and_vehicles', 'MarkBuckawicki', 'CC0', 'https://commons.wikimedia.org/wiki/File:Flooded_street_and_vehicles.JPG'), ('Flooded_street_near_13th', 'Don Becker/USGS', 'Public domain in the US', 'https://commons.wikimedia.org/wiki/File:Flooded_street_near_13th_Ave_and_J_Street_Cedar_Rapids_Iowa.jpg'), ('July_2023_flood', 'United States Military Academy', 'Public domain in the US', 'https://commons.wikimedia.org/wiki/File:July_2023_flood_damage_in_West_Point.jpg'), ('Sunken_cars_on', 'Don Becker/USGS', 'Public domain in the US', 'https://commons.wikimedia.org/wiki/File:Sunken_cars_on_flooded_street_in_Cedar_Rapids_Iowa.jpg'), ('Two_cars_stuck', 'Bidgee', 'CC BY-SA 2.5 Australia', 'https://commons.wikimedia.org/wiki/File:Two_cars_stuck_on_the_flooded_Berry_Street.jpg'), ('UAE_Flood', 'CherryPie94', 'CC0', 'https://commons.wikimedia.org/wiki/File:UAE_Flood_-_16_April_2024.jpg')]


RESULT_COLUMNS = [
    'car_id', 'relative_flood_level', 'detector_confidence', 'classifier_confidence',
    'uncertain', 'x1', 'y1', 'x2', 'y2', 'crop_x1', 'crop_y1', 'crop_x2', 'crop_y2',
]
TABLE_COLUMNS = ['Car', 'Relative level', 'Detector score', 'Classifier score', 'Uncertain']


def canonical_image(image):
    if image is None:
        raise ValueError('Upload an image or load an example first.')
    if not isinstance(image, Image.Image):
        raise ValueError('The input must be an image uploaded through the image box.')
    width, height = image.size
    if width < 2 or height < 2 or width * height > MAX_IMAGE_PIXELS:
        raise ValueError('Use an image with at least 2 × 2 pixels and no more than 20 megapixels.')
    return ImageOps.exif_transpose(image).convert('RGB')


def image_fingerprint(image):
    image = canonical_image(image)
    return hashlib.sha256(str(image.size).encode('ascii') + image.tobytes()).hexdigest()


def expanded_crop_box(x1, y1, x2, y2, image_width, image_height):
    pad_x = int(round(CROP_PADDING_FRACTION * max(1, x2 - x1)))
    pad_y = int(round(CROP_PADDING_FRACTION * max(1, y2 - y1)))
    return (max(0, x1 - pad_x), max(0, y1 - pad_y),
            min(image_width, x2 + pad_x), min(image_height, y2 + pad_y))


@torch.inference_mode()
def classify_crop(crop_bgr):
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    tensor = classifier_transform(Image.fromarray(crop_rgb)).unsqueeze(0).to(device)
    probabilities = torch.softmax(classifier(tensor), dim=1)[0]
    class_index = int(probabilities.argmax().item())
    return class_names[class_index], float(probabilities[class_index].item())


def draw_label(image, box, text, uncertain):
    # Green means the classifier threshold was met, not that a road is safe.
    color = (235, 145, 25) if uncertain else (0, 155, 80)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    x1, y1, x2, y2 = box
    draw.rectangle((x1, y1, x2 - 1, y2 - 1), outline=color, width=2)
    bounds = draw.textbbox((0, 0), text, font=font)
    text_width = bounds[2] - bounds[0]
    text_height = bounds[3] - bounds[1]
    left = min(x1, max(0, image.width - text_width - 8))
    top = max(0, y1 - text_height - 10)
    draw.rectangle((left, top, min(image.width - 1, left + text_width + 8),
                    min(image.height - 1, top + text_height + 8)), fill=color)
    draw.text((left + 4, top + 3 - bounds[1]), text, font=font, fill=(0, 0, 0))


def process_image(image, detector_confidence=DETECTOR_CONFIDENCE,
                  detector_imgsz=DETECTOR_IMAGE_SIZE):
    original = canonical_image(image)
    width, height = original.size
    original_bgr = cv2.cvtColor(np.asarray(original), cv2.COLOR_RGB2BGR)
    annotated = original.copy()
    prediction = detector.predict(
        source=original_bgr, classes=[2], conf=float(detector_confidence),
        iou=DETECTOR_IOU, imgsz=int(detector_imgsz), device=DETECTOR_DEVICE, verbose=False,
    )[0]
    records, gallery = [], []
    if prediction.boxes is None:
        return annotated, records, gallery
    for box in prediction.boxes:
        coordinates = box.xyxy[0].detach().cpu().tolist()
        if len(coordinates) != 4 or not all(math.isfinite(value) for value in coordinates):
            continue
        x1, y1, x2, y2 = [int(round(value)) for value in coordinates]
        x1, y1 = max(0, min(width, x1)), max(0, min(height, y1))
        x2, y2 = max(0, min(width, x2)), max(0, min(height, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        detector_score = float(box.conf[0].item())
        crop_box = expanded_crop_box(x1, y1, x2, y2, width, height)
        cx1, cy1, cx2, cy2 = crop_box
        crop_bgr = original_bgr[cy1:cy2, cx1:cx2].copy()
        level_name, classifier_score = classify_crop(crop_bgr)
        uncertain = classifier_score < CLASSIFIER_CONFIDENCE
        car_id = len(records) + 1
        shown_level = f'Uncertain ({level_name})' if uncertain else level_name
        text = f'Car {car_id}: {shown_level} | det {detector_score:.2f} | cls {classifier_score:.2f}'
        draw_label(annotated, (x1, y1, x2, y2), text, uncertain)
        records.append({
            'car_id': car_id, 'relative_flood_level': level_name,
            'detector_confidence': round(detector_score, 6),
            'classifier_confidence': round(classifier_score, 6), 'uncertain': uncertain,
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            'crop_x1': cx1, 'crop_y1': cy1, 'crop_x2': cx2, 'crop_y2': cy2,
        })
        gallery.append((original.crop(crop_box), f'Car {car_id} · {shown_level} · cls {classifier_score:.2f}'))
    return annotated, records, gallery


def unknown_credit():
    return {'source_image': 'Uploaded image', 'author': 'Not supplied',
            'license': 'Check before distributing', 'source_url': '', 'license_url': '',
            'changes': 'Car boxes and labels added; car crops extracted for display.'}


def sample_credit(filename):
    match = next((row for row in CREDIT_RECORDS if filename.startswith(row[0])), None)
    credit = unknown_credit()
    credit['source_image'] = filename
    if match:
        _, credit['author'], credit['license'], credit['source_url'] = match
        credit['license_url'] = {
            'CC BY-SA 2.0': 'https://creativecommons.org/licenses/by-sa/2.0/',
            'CC BY-SA 4.0': 'https://creativecommons.org/licenses/by-sa/4.0/',
            'CC BY-SA 2.5 Australia': 'https://creativecommons.org/licenses/by-sa/2.5/au/',
            'CC0': 'https://creativecommons.org/publicdomain/zero/1.0/',
            'Public domain in the US': 'https://commons.wikimedia.org/wiki/Template:PD-USGov',
        }[credit['license']]
    return credit


def credit_text(credit):
    if not credit.get('source_url'):
        return 'Source credit: not supplied. Check the image licence before sharing this result.'
    return (f"Source: [{credit['source_image']}]({credit['source_url']}) · "
            f"{credit['author']} · [{credit['license']}]({credit['license_url']}). "
            'Boxes/text and displayed car crops are derived from this source. '
            'Keep the credit and applicable licence when distributing these copies.')


SAMPLE_LOOKUP = {}
sample_paths = sorted(
    (path for path in SCENE_DIR.rglob('*')
     if path.is_file() and path.suffix.lower() in {'.jpg', '.jpeg', '.png'}),
    key=lambda path: str(path).casefold(),
) if SCENE_DIR.is_dir() else []
for sample_path in sample_paths:
    if len(SAMPLE_LOOKUP) >= MAX_SAMPLE_IMAGES:
        break
    try:
        with Image.open(sample_path) as opened:
            example = canonical_image(opened)
        number = len(SAMPLE_LOOKUP) + 1
        sample_id = f'Example {number:02d}'
        local_path = SAMPLES_DIR / f'example_{number:02d}.png'
        example.save(local_path)
        SAMPLE_LOOKUP[sample_id] = {'path': local_path, 'credit': sample_credit(sample_path.name)}
    except (OSError, ValueError, Image.DecompressionBombError) as error:
        print(f'Skipping example {sample_path.name}: {error}')
print(f'{len(SAMPLE_LOOKUP)} optional examples ready. Uploads work even if this number is zero.')


# ## 6. Connect the page to the image pipeline
# 
# Every successful analysis creates its own result folder. The table and CSV contain separate detector and classifier scores, box coordinates, padded-crop coordinates and an uncertainty flag. Empty detection results still produce a CSV with its column headers.
# 
# The downloads are the annotated PNG, predictions CSV and settings/source-credit JSON. The original image is kept locally and is included if we make the optional backup. Credentials are not written into result files. Use non-sensitive demo images: a password-protected temporary page is still an internet-accessible service.


INITIAL_MESSAGE = 'Upload an image or load an example, then click Analyse image.'
INITIAL_CREDIT = 'Source credit appears here when a known example is analysed.'


def empty_table():
    return pd.DataFrame(columns=TABLE_COLUMNS)


def clear_outputs():
    return None, empty_table(), [], INITIAL_MESSAGE, [], INITIAL_CREDIT


def reset_page():
    return (None, None, *clear_outputs())


def new_upload():
    return (None, *clear_outputs())


def load_example(sample_id):
    if sample_id not in SAMPLE_LOOKUP:
        gr.Warning('Choose an example from the list first, or upload your own image.')
        return reset_page()
    item = SAMPLE_LOOKUP[sample_id]
    with Image.open(item['path']) as opened:
        example = canonical_image(opened)
    state = {'fingerprint': image_fingerprint(example), 'credit': dict(item['credit'])}
    return example, state, *clear_outputs()


def analyse_image(image, detector_confidence, detector_imgsz, source_state):
    try:
        original = canonical_image(image)
        confidence = float(detector_confidence)
        image_size_setting = int(detector_imgsz)
        if not math.isfinite(confidence) or not 0.05 <= confidence <= 0.80:
            raise ValueError('Detector confidence must be between 0.05 and 0.80.')
        if image_size_setting not in (640, 960, 1280):
            raise ValueError('Choose detector image size 640, 960 or 1280.')
    except (TypeError, ValueError, OverflowError) as error:
        return None, empty_table(), [], f'No result: {error}', [], INITIAL_CREDIT

    try:
        start_time = time.perf_counter()
        annotated, records, gallery = process_image(original, confidence, image_size_setting)
        elapsed = time.perf_counter() - start_time
        credit = unknown_credit()
        # Do not attach an old example's credit to a different uploaded image.
        if (isinstance(source_state, dict)
                and source_state.get('fingerprint') == image_fingerprint(original)):
            credit = dict(source_state['credit'])

        result_id = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]
        result_dir = RESULTS_DIR / result_id
        result_dir.mkdir(parents=True, exist_ok=False)
        original.save(result_dir / 'input_image.png')
        annotated_path = result_dir / 'annotated_image.png'
        predictions_path = result_dir / 'predictions.csv'
        metadata_path = result_dir / 'settings_and_source.json'
        annotated.save(annotated_path)
        frame = pd.DataFrame(records, columns=RESULT_COLUMNS)
        frame.to_csv(predictions_path, index=False)
        uncertain_count = sum(bool(record['uncertain']) for record in records)
        metadata = {
            'project': 'Computer Vision-Based Flood Detection and Water-Level Monitoring System',
            'team': 'Team 20', 'result_id': result_id,
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'input_size': {'width': original.width, 'height': original.height},
            'detector': 'YOLOv8n / COCO-pretrained / car class only',
            'classifier': 'ResNet-50 / our Task 4 checkpoint',
            'checkpoint_filename': CHECKPOINT_PATH.name, 'checkpoint_sha256': CHECKPOINT_SHA256,
            'class_names': class_names,
            'settings': {'detector_confidence': confidence, 'detector_iou': DETECTOR_IOU,
                         'detector_imgsz': image_size_setting, 'classifier_confidence': CLASSIFIER_CONFIDENCE,
                         'crop_padding_per_side': CROP_PADDING_FRACTION,
                         'classifier_image_size': image_size,
                         'normalization_mean': list(imagenet_mean), 'normalization_std': list(imagenet_std)},
            'detected_cars': len(records), 'uncertain_predictions': uncertain_count,
            'processing_seconds': round(elapsed, 3),
            'timing_note': 'Detection, crop classification and annotation; not a benchmark. First run may be slower.',
            'package_versions': {'gradio': gr.__version__, 'ultralytics': ultralytics.__version__,
                                 'torch': torch.__version__},
            'source_credit': credit,
            'limitations': [
                'Relative categories only; no exact water depth or road-safety judgement.',
                'Cars can be missed or false boxes produced, especially with strong submergence or occlusion.',
                'Scene results have no ground-truth accuracy score.',
                'Uncertain means classifier score below 0.50; it is not a sixth learned class.',
                'Lower detector confidence or larger input size is exploratory, not verified improvement.',
            ],
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
        table = pd.DataFrame([
            {'Car': record['car_id'], 'Relative level': record['relative_flood_level'],
             'Detector score': record['detector_confidence'],
             'Classifier score': record['classifier_confidence'], 'Uncertain': record['uncertain']}
            for record in records
        ], columns=TABLE_COLUMNS)
        if not records:
            summary = ('No cars were detected at these settings. This is not evidence that the image '
                       'contains no cars or no flood. A heavily submerged car may be missed.')
        else:
            summary = f'Detected cars: {len(records)} · Uncertain classifications: {uncertain_count}.'
        summary += f' Processing time: {elapsed:.2f} s (not a benchmark).'
        if confidence != DETECTOR_CONFIDENCE or image_size_setting != DETECTOR_IMAGE_SIZE:
            summary += ' Exploratory settings are active; additional boxes may be incorrect.'
        return (annotated, table, gallery, summary,
                [str(annotated_path), str(predictions_path), str(metadata_path)], credit_text(credit))
    except Exception:
        # Keep technical details in our Colab output, not in a public app error.
        traceback.print_exc()
        return (None, empty_table(), [],
                'Analysis failed. Check the Colab cell output; reduce the detector image size if GPU memory is low.',
                [], INITIAL_CREDIT)


# ## 7. Build the Gradio page
# 
# The main controls are upload, load example, analyse and clear. Advanced settings let us explore detector confidence and image size without changing our trained models. Lowering confidence can reveal missed cars but can also add false boxes; it is not proof of higher accuracy.
# 
# Green boxes mean the classifier score meets the 0.50 threshold; amber boxes indicate an uncertain classification. Neither colour says that a road is safe. The car IDs connect boxes, rows and crop-gallery captions.


required_ui_names = [
    'gr', 'SAMPLE_LOOKUP', 'TABLE_COLUMNS', 'DETECTOR_CONFIDENCE',
    'DETECTOR_IMAGE_SIZE', 'INITIAL_MESSAGE', 'INITIAL_CREDIT', 'empty_table',
    'analyse_image', 'load_example', 'new_upload', 'reset_page', 'clear_outputs',
]
missing_ui_names = [name for name in required_ui_names if name not in globals()]
if missing_ui_names:
    raise RuntimeError(
        'Paste this cell into the original connected demo notebook, where Sections 1–6 '
        'have already run. Missing: ' + ', '.join(missing_ui_names)
    )

# System fonts avoid fetching an additional font during the presentation.
DEMO_THEME = gr.themes.Base(
    primary_hue='teal', secondary_hue='slate', neutral_hue='slate',
    font=['Segoe UI', 'Arial', 'sans-serif'],
    font_mono=['Consolas', 'monospace'],
)
theme_values = DEMO_THEME.to_dict()['theme']
# Give the app the same light appearance even when the browser prefers dark mode.
theme_updates = {
    key: theme_values[key[:-5]]
    for key in theme_values
    if key.endswith('_dark') and key[:-5] in theme_values
}
light_tokens = {
    'body_background_fill': '#f6f8fa',
    'body_text_color': '#172b3a',
    'body_text_color_subdued': '#60717f',
    'background_fill_primary': '#ffffff',
    'background_fill_secondary': '#f6f8fa',
    'block_background_fill': '#ffffff',
    'block_border_color': '#dce4e9',
    'block_label_text_color': '#526575',
    'block_label_background_fill': 'transparent',
    'block_title_background_fill': 'transparent',
    'block_title_text_color': '#172b3a',
    'block_info_text_color': '#60717f',
    'block_shadow': 'none',
    'block_radius': '12px',
    'input_background_fill': '#ffffff',
    'input_background_fill_hover': '#f8fafb',
    'input_border_color': '#d6e0e6',
    'input_border_color_focus': '#147d79',
    'input_radius': '8px',
    'input_shadow': 'none',
    'button_primary_background_fill': '#147d79',
    'button_primary_background_fill_hover': '#106561',
    'button_primary_border_color': '#147d79',
    'button_primary_text_color': '#ffffff',
    'button_primary_text_color_hover': '#ffffff',
    'button_primary_shadow': 'none',
    'button_secondary_background_fill': '#ffffff',
    'button_secondary_background_fill_hover': '#f1f5f7',
    'button_secondary_border_color': '#d6e0e6',
    'button_secondary_text_color': '#344d5e',
    'button_secondary_shadow': 'none',
    'button_large_radius': '8px',
    'button_large_text_weight': '600',
    'button_large_padding': '12px 20px',
    'table_even_background_fill': '#ffffff',
    'table_odd_background_fill': '#ffffff',
    'table_border_color': '#e3e9ed',
    'table_text_color': '#243b4a',
    'panel_background_fill': '#ffffff',
    'border_color_primary': '#dce4e9',
    'border_color_accent': '#147d79',
    'link_text_color': '#147d79',
    'layout_gap': '18px',
}
for key, value in light_tokens.items():
    theme_updates[key] = value
    if key + '_dark' in theme_values:
        theme_updates[key + '_dark'] = value
DEMO_THEME = DEMO_THEME.set(**theme_updates)

DEMO_CSS = """
.gradio-container {
    max-width: 1220px !important;
    width: 100% !important;
    margin: 0 auto !important;
    padding: 24px 28px 32px !important;
}
.gradio-container .main { width: 100% !important; max-width: none !important; padding: 0 !important; }
#flood-workspace { gap: 18px; }
#flood-header { padding: 0; }
.flood-topline {
    display: flex; align-items: center; justify-content: space-between;
    gap: 12px; padding: 0 0 16px; border-bottom: 1px solid #dce4e9;
}
.flood-brand { font-size: 14px; font-weight: 700; letter-spacing: .03em; color: #172b3a; }
.flood-brand span { margin-left: 12px; font-weight: 400; letter-spacing: 0; color: #60717f; }
.flood-badge {
    border: 1px solid #dce4e9; border-radius: 20px; padding: 5px 11px;
    font-size: 11px; font-weight: 600; color: #526575; background: #ffffff;
    white-space: nowrap;
}
.flood-intro { padding: 23px 0 5px; }
.flood-intro h1 {
    margin: 0 0 8px; font-size: 30px; line-height: 1.2;
    font-weight: 650; letter-spacing: -.025em; color: #172b3a;
}
.flood-intro p { margin: 0; max-width: 860px; font-size: 14px; line-height: 1.6; color: #60717f; }
.flood-card {
    background: #ffffff !important; border: 1px solid #dce4e9 !important;
    border-radius: 12px !important; padding: 16px !important; overflow: hidden;
    box-shadow: 0 2px 5px rgba(25, 45, 60, .025) !important;
}
.flood-card-heading {
    display: flex; align-items: baseline; justify-content: space-between;
    flex-wrap: wrap; gap: 5px 12px; margin-bottom: 12px;
}
.flood-card-heading h2 { margin: 0; font-size: 16px; font-weight: 600; color: #172b3a; }
.flood-card-heading span { font-size: 12px; color: #738492; }
.flood-card .html-container, #flood-header .html-container { padding: 0 !important; background: transparent !important; }
#flood-input, #flood-result {
    border: 1px solid #e6ebef !important; border-radius: 8px !important;
    background: #fafcfd !important;
}
#flood-toolbar { align-items: end; gap: 14px; }
#flood-example-controls { gap: 10px; align-items: end; }
#flood-example { border: none !important; padding: 0 !important; background: transparent !important; }
#flood-example label { font-size: 12px; }
#flood-actions { align-items: end; gap: 10px; }
#flood-analyse { min-height: 46px; }
#flood-clear, #flood-load { min-height: 46px; }
#flood-load { font-size: 13px; white-space: nowrap; }
#flood-status {
    padding: 13px 16px; border: 1px solid #dce7e6; border-radius: 9px;
    background: #f0f7f6; color: #344d5e;
}
#flood-status p { margin: 0; font-size: 13px; line-height: 1.6; }
#flood-results {
    border: 1px solid #dce4e9; border-radius: 12px;
    background: #ffffff; padding: 0 16px 16px;
}
#flood-results .tab-nav { padding-top: 5px; gap: 8px; }
#flood-results .tab-nav button { font-size: 13px; font-weight: 600; padding: 13px 14px; }
#flood-results .tab-nav button.selected { color: #147d79 !important; }
#flood-results .tabitem { padding: 16px 0 0; }
#flood-predictions { border-radius: 8px; }
#flood-predictions th, #flood-predictions td { font-size: 13px; font-family: 'Segoe UI', Arial, sans-serif; }
#flood-predictions th { white-space: nowrap !important; word-break: normal !important; }
#flood-crops { background: #fafcfd; }
#flood-settings { background: #ffffff; border: 1px solid #dce4e9; border-radius: 10px; }
#flood-settings .label-wrap { font-size: 13px; }
.flood-tab-note p, #flood-credit p { font-size: 13px; line-height: 1.65; color: #60717f; }
#flood-scope p { margin: 0; font-size: 12px; line-height: 1.6; color: #738492; }
.flood-about h3 { font-size: 15px; margin: 0 0 8px; }
.flood-about p, .flood-about li { font-size: 13px; line-height: 1.7; color: #526575; }
button:focus-visible { outline: 2px solid #147d79 !important; outline-offset: 3px; }
@media (max-width: 760px) {
    .gradio-container { padding: 18px 14px 24px !important; }
    .flood-intro h1 { font-size: 25px; }
    .flood-intro { padding-top: 18px; }
    .flood-brand span { display: block; margin: 4px 0 0; font-size: 12px; }
    .flood-badge { font-size: 10px; }
    .flood-card { padding: 13px !important; }
    #flood-images { flex-direction: column; }
    #flood-input, #flood-result { height: 280px !important; min-height: 280px !important; }
    #flood-toolbar { flex-direction: column; align-items: stretch; }
    #flood-toolbar > div { width: 100%; min-width: 0 !important; }
    #flood-results .tab-nav { gap: 0; }
    #flood-results .tab-nav button { padding: 12px 9px; font-size: 12px; }
}
"""

previous_demo = globals().get('demo')
if previous_demo is not None:
    previous_demo.close()

with gr.Blocks(analytics_enabled=False, title='Team 20 | Flood-level analysis') as demo:
    with gr.Column(elem_id='flood-workspace'):
        gr.HTML("""
        <div class="flood-topline">
          <div class="flood-brand">TEAM 20<span>CSE411 · Computer Vision</span></div>
          <div class="flood-badge">IMAGE PROTOTYPE</div>
        </div>
        <div class="flood-intro">
          <h1>Flood-level image analysis</h1>
          <p>Computer Vision-Based Flood Detection and Water-Level Monitoring System.<br>
          Upload a scene, analyse the cars, then inspect their predicted relative flood levels.</p>
        </div>
        """, elem_id='flood-header')
        source_state = gr.State(None)
        with gr.Row(equal_height=True, elem_id='flood-images'):
            with gr.Column(min_width=320, elem_classes='flood-card'):
                gr.HTML('<div class="flood-card-heading"><h2>Input scene</h2>'
                        '<span>Full-scene photograph</span></div>')
                input_image = gr.Image(
                    type='pil', image_mode='RGB', sources=['upload'], format='png',
                    label='Input scene', show_label=False, container=False,
                    height=360, elem_id='flood-input',
                )
            with gr.Column(min_width=320, elem_classes='flood-card'):
                gr.HTML('<div class="flood-card-heading"><h2>Annotated result</h2>'
                        '<span>Car boxes · levels · scores</span></div>')
                output_image = gr.Image(
                    type='pil', format='png', label='Annotated result',
                    show_label=False, container=False, height=360, interactive=False,
                    buttons=['download', 'fullscreen'], elem_id='flood-result',
                )
        with gr.Row(elem_id='flood-toolbar'):
            with gr.Column(scale=3, min_width=320):
                with gr.Row(elem_id='flood-example-controls'):
                    example_choice = gr.Dropdown(
                        choices=list(SAMPLE_LOOKUP), value=next(iter(SAMPLE_LOOKUP), None),
                        label='Or choose an example', interactive=bool(SAMPLE_LOOKUP),
                        scale=3, min_width=150, elem_id='flood-example',
                    )
                    example_button = gr.Button(
                        'Load example', interactive=bool(SAMPLE_LOOKUP),
                        scale=1, min_width=120, elem_id='flood-load',
                    )
            with gr.Column(scale=2, min_width=280):
                with gr.Row(elem_id='flood-actions'):
                    analyse_button = gr.Button(
                        'Analyse image', variant='primary', scale=3, elem_id='flood-analyse',
                    )
                    clear_button = gr.Button('Clear', scale=1, min_width=80, elem_id='flood-clear')
        summary = gr.Markdown(INITIAL_MESSAGE, elem_id='flood-status')

        with gr.Tabs(elem_id='flood-results'):
            with gr.Tab('Predictions'):
                result_table = gr.Dataframe(
                    value=empty_table(), headers=TABLE_COLUMNS,
                    datatype=['number', 'str', 'number', 'number', 'bool'],
                    interactive=False, label='Per-car predictions', show_label=False,
                    wrap=False, max_height=260, column_widths=['70px', '155px', '155px', '160px', '105px'],
                    buttons=[], elem_id='flood-predictions',
                )
                gr.Markdown('Car IDs match the boxes and crop captions. A classifier score below '
                            '0.50 is marked uncertain; scores are not guaranteed correctness.',
                            elem_classes='flood-tab-note')
            with gr.Tab('Car crops'):
                gr.Markdown('These padded crops come from the original scene and are the '
                            'images passed to ResNet-50.', elem_classes='flood-tab-note')
                crop_gallery = gr.Gallery(
                    label='Car crops used for classification', show_label=False, columns=3,
                    height=300, object_fit='contain', format='png', interactive=False,
                    buttons=['fullscreen'], elem_id='flood-crops',
                )
            with gr.Tab('Downloads'):
                gr.Markdown('Save the annotated image (PNG), prediction rows (CSV) and '
                            'settings with source credit (JSON).', elem_classes='flood-tab-note')
                result_files = gr.File(
                    label='Result files', file_count='multiple', interactive=False,
                    elem_id='flood-downloads',
                )
                source_credit = gr.Markdown(INITIAL_CREDIT, elem_id='flood-credit')
            with gr.Tab('Project details'):
                gr.Markdown(
                    '### How the prototype works\n'
                    'The COCO-pretrained YOLOv8n detects cars. We expand each detected box, '
                    'then use our STURM-trained ResNet-50 to predict Level0–Level4. '
                    'YOLO was not trained on STURM.\n\n'
                    '### What to keep in mind\n'
                    'Cars can be missed, boxes can be wrong, and crop-style differences can '
                    'affect classification. Green boxes indicate a classifier score of at least '
                    '0.50; amber boxes indicate uncertainty. Neither colour means a road is safe. '
                    'These scene outputs do not measure whole-system accuracy.\n\n'
                    '### Team 20\n'
                    'Mahendra K (2023BCS0235), Sanin K (2023BCS0232), '
                    'Bhukya Gandhi (2023BCS0238), Ankit Subhash (2022BCD0001).',
                    elem_classes='flood-about',
                )
        with gr.Accordion('Advanced detector settings', open=False, elem_id='flood-settings'):
            with gr.Row():
                detector_confidence = gr.Slider(
                    0.05, 0.80, value=DETECTOR_CONFIDENCE, step=0.05,
                    label='Detector confidence threshold',
                )
                detector_imgsz = gr.Dropdown(
                    [640, 960, 1280], value=DETECTOR_IMAGE_SIZE,
                    label='Detector image size', interactive=True,
                )
            baseline_button = gr.Button('Restore baseline settings', size='sm')
            gr.Markdown('Baseline: confidence 0.30 / size 640. NMS IoU remains 0.45; '
                        'the classifier threshold remains 0.50. Lower confidence or larger '
                        'input size may produce additional incorrect boxes.', elem_classes='flood-tab-note')
        gr.Markdown('Relative flood categories only—not exact water depth or a road-safety '
                    'assessment. Image-based research prototype.', elem_id='flood-scope')

    # Keep the same callbacks, output order and single-model execution queue.
    result_components = [output_image, result_table, crop_gallery, summary, result_files, source_credit]
    analyse_button.click(
        analyse_image, inputs=[input_image, detector_confidence, detector_imgsz, source_state],
        outputs=result_components, concurrency_limit=1, concurrency_id='models', api_visibility='private',
    )
    example_button.click(load_example, inputs=example_choice,
                         outputs=[input_image, source_state, *result_components], api_visibility='private')
    input_image.upload(new_upload, outputs=[source_state, *result_components], api_visibility='private')
    input_image.clear(new_upload, outputs=[source_state, *result_components], api_visibility='private')
    clear_button.click(reset_page, outputs=[input_image, source_state, *result_components],
                       api_visibility='private')
    baseline_button.click(lambda: (DETECTOR_CONFIDENCE, DETECTOR_IMAGE_SIZE),
                          outputs=[detector_confidence, detector_imgsz], api_visibility='private')
    detector_confidence.change(clear_outputs, outputs=result_components, api_visibility='private')
    detector_imgsz.change(clear_outputs, outputs=result_components, api_visibility='private')

demo.queue(max_size=4, default_concurrency_limit=1)
print('Redesigned page built. Run Section 8 next. The loaded models and session results are unchanged.')


# ## 8. Start the temporary demo
# 
# This cell generates the page password and prints it below. Open the `gradio.live` link, then enter the printed username and password. Share them only with your teammates/evaluator. Keep Colab connected while presenting; this is temporary hosting, not a permanent website.
# 
# The launch cell output contains the password. Clear that output before publishing the executed notebook to an open repository. Result PNG/CSV/JSON files do not contain it. Do not display this cell's password on a publicly recorded presentation screen.
# 
# For the presentation, first try one example that produces useful detections and one that shows a limitation. Compare the annotated scene with the original; do not describe the detector's car count as a measured accuracy score.


if 'DEMO_THEME' not in globals() or 'DEMO_CSS' not in globals():
    raise RuntimeError('Run the replacement Section 7 cell first.')

DEMO_USERNAME = 'team20'
DEMO_PASSWORD = secrets.token_urlsafe(16)
print('Temporary demo login')
print('Username:', DEMO_USERNAME)
print('Password:', DEMO_PASSWORD)
print('Open the NEW public link printed below. Keep this runtime connected.')

demo.launch(
    share=True, inline=False, debug=False, show_error=False,
    auth=(DEMO_USERNAME, DEMO_PASSWORD),
    auth_message='Team 20 project demo. Enter the login printed in our Colab notebook.',
    max_file_size=MAX_UPLOAD_SIZE,
    allowed_paths=[str(RESULTS_DIR.resolve())],
    blocked_paths=['/content/drive', str(CHECKPOINT_PATH.resolve())],
    run_history=False, enable_monitoring=False, mcp_server=False,
    theme=DEMO_THEME, css=DEMO_CSS, footer_links=['gradio'],
)


# ## 9. Optional: back up this session to Drive
# 
# The page already lets us download each result. If we also want a single ZIP of all analyses in this session, change `SAVE_SESSION_BACKUP` to `True` and run this cell **after** using the page.
# 
# The ZIP is saved in a new `gradio_demo_outputs` folder, not in the earlier task folders. It contains input images and their result files, but not model weights or login credentials. Treat the backup as private if any uploaded image is sensitive. Running all cells with the default `False` does not create a backup.


SAVE_SESSION_BACKUP = False

if SAVE_SESSION_BACKUP:
    completed_runs = sorted(path for path in RESULTS_DIR.iterdir()
                            if path.is_dir() and (path / 'settings_and_source.json').is_file())
    if not completed_runs:
        print('No completed analyses yet. Use the page first, then rerun this cell.')
    else:
        # Snapshot only complete results, so a failed/unfinished run is not included.
        snapshot_dir = SESSION_DIR / ('backup_snapshot_' + uuid.uuid4().hex[:8])
        snapshot_dir.mkdir()
        for run_dir in completed_runs:
            shutil.copytree(run_dir, snapshot_dir / run_dir.name)
        backup_dir = PROJECT_DIR / 'gradio_demo_outputs'
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_name = 'Team20_Gradio_Demo_' + SESSION_ID + '_' + uuid.uuid4().hex[:6]
        backup_path = shutil.make_archive(str(backup_dir / backup_name), 'zip', root_dir=snapshot_dir)
        print('Saved session backup:', backup_path)
else:
    print('Optional backup skipped. To save after using the demo, set SAVE_SESSION_BACKUP = True.')


# ## 10. Optional: stop the demo after presenting
# 
# Leave the default `False` while using the page. When finished, set it to `True` and run this cell. Disconnecting and deleting the Colab runtime also ends the demo and removes its local session files. Download or back up anything needed first.


STOP_DEMO = False
if STOP_DEMO:
    demo.close()
    print('Demo stopped. Run Section 8 again if the page needs to be reopened.')
else:
    print('The demo remains open. This cell does not stop it with the default setting.')


# ## Quick checks before the presentation
# 
# 1. Confirm that Section 4 loads the selected checkpoint and shows the five classes in order.
# 2. Analyse an image, then compare the scene, its boxes, the prediction rows and the crop gallery. Cars can be missed and boxes can be wrong; keep a limitation example ready.
# 3. Download one PNG, CSV and JSON. Confirm that they correspond to the same image/settings and retain source credits when distributing an example result.
# 4. Keep the original executed Task 7 notebook/report as the experiment evidence. This interface alone is not a new accuracy evaluation and does not change the completed methodology.
# 
# If the link does not open, check that the runtime is still connected and rerun Section 8. If imports fail after a previous package downgrade, use a fresh runtime and run the setup in order. For a missing checkpoint, fix its path in Section 3. For a GPU-memory error, restore detector size 640 and try a smaller input image.
# 
# Gradio reference: [Quickstart](https://www.gradio.app/guides/quickstart), [sharing apps](https://www.gradio.app/guides/sharing-your-app), [file access](https://www.gradio.app/guides/file-access). The demo uses the public packaged Gradio release pinned in Section 1.
