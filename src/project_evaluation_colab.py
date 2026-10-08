# SPDX-License-Identifier: AGPL-3.0-only

"""Google Colab source export; use the corresponding notebook to run it."""


# # CSE411 Project Task 7
# ## Computer Vision-Based Flood Detection and Water-Level Monitoring System
# ### Implementation - Part 4
# 
# **Team 20**
# 
# 2023BCS0235 Mahendra K  
# 2023BCS0232 Sanin K  
# 2023BCS0238 Bhukya Gandhi  
# 2022BCD0001 Ankit Subhash


# ## Final project scope
# 
# This notebook brings together the completed classifier and image pipeline. The classifier is evaluated on the official STURM test split, and the integrated system is tested on our full-scene images. It also provides a detector-setting comparison, an image-testing panel and the saved records needed for the final report.
# 
# COCO-pretrained YOLOv8n detects cars. The Task 4 ResNet-50 checkpoint assigns a relative `Level0`–`Level4` category to each detected car crop. STURM provides cropped vehicles for classifier training; it is not used to train YOLOv8n. The system does not measure water depth in centimetres or metres. Video is an optional extension and is outside this final still-image implementation.
# 
# The existing five-epoch checkpoint was selected by validation macro F1. This final notebook reuses it; classifier training source is included separately for reproducibility. No extra training run is required to use this notebook.
# 
# Classifier metrics refer to the labelled STURM test crops. The full-scene images do not have ground-truth boxes or class labels, so scene counts and confidence scores are diagnostic results. They are not end-to-end accuracy measurements.


# ## 1. Install dependencies and connect Drive


# Install the notebook dependencies before running this source.


from google.colab import drive
drive.mount('/content/drive')


# ## 2. Paths and inference settings


from pathlib import Path
from datetime import datetime, timezone
import json
import re
import shutil
import time

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, UnidentifiedImageError
from IPython.display import display, clear_output

import torch
import torch.nn as nn
from torchvision import models, transforms
import ultralytics
from ultralytics import YOLO

PROJECT_DIR = Path('/content/drive/MyDrive/CSE411_Flood_Project')
CHECKPOINT_PATH = PROJECT_DIR / 'task4_outputs' / 'resnet50_sturm_best_macro_f1.pth'
SCENE_DIR = PROJECT_DIR / 'task5_scene_images'
OUTPUT_ROOT = PROJECT_DIR / 'task7_outputs'
DATASET_DIR = PROJECT_DIR / 'STURM-FloodDepth'
IMAGE_ROOT = DATASET_DIR / 'flooded_cars' / 'upscaled_images'
TRAINING_HISTORY_PATH = PROJECT_DIR / 'task4_outputs' / 'resnet50_macro_f1_training_history.csv'

# Only the COCO 'car' class is in scope. Other vehicles are not classified.
CAR_CLASS_ID = 2
DETECTOR_CONFIDENCE = 0.30
DETECTOR_IOU = 0.45
CLASSIFIER_CONFIDENCE = 0.50
CROP_PADDING_FRACTION = 0.20
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}

# Optional: add links for the photos you use, keyed by their exact filename.
# These links are saved with the output for later image attribution.
SOURCE_URLS = {
    # 'example.jpg': 'https://commons.wikimedia.org/wiki/File:Example.jpg',
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)
print('Checkpoint:', CHECKPOINT_PATH)
print('Scene folder:', SCENE_DIR)
print('Output root:', OUTPUT_ROOT)


# Credits for the ten Wikimedia Commons scenes used in Task 5.
# Filenames are matched by a distinctive prefix, including Unicode filenames.
credit_records = [
    ('Cars_driving_through', 'TCExplorer', 'CC BY-SA 2.0', 'https://commons.wikimedia.org/wiki/File:Cars_driving_through_flood_water_on_the_B4380_near_Atcham_-_geograph.org.uk_-_7390621.jpg'),
    ('Cars_in_flooded', 'Jiří Sedláček', 'CC BY-SA 4.0', 'https://commons.wikimedia.org/wiki/File:Cars_in_flooded_Novodvorsk%C3%A1_street_in_T%C5%99eb%C3%AD%C4%8D%2C_T%C5%99eb%C3%AD%C4%8D_District.JPG'),
    ('FEMA_-_32048', 'Marvin Nauman/FEMA', 'Public domain in the US', 'https://commons.wikimedia.org/wiki/File:FEMA_-_32048_-_Red_car_floating_in_flood_waters_in_Oklahoma.jpg'),
    ('FEMA_-_32096', 'Marvin Nauman/FEMA', 'Public domain in the US', 'https://commons.wikimedia.org/wiki/File:FEMA_-_32096_-_Cars_drives_though_flooded_street_in_an_Oklahoma_neighborhood.jpg'),
    ('Flooded_street_and_vehicles', 'MarkBuckawicki', 'CC0', 'https://commons.wikimedia.org/wiki/File:Flooded_street_and_vehicles.JPG'),
    ('Flooded_street_near_13th', 'Don Becker/USGS', 'Public domain in the US', 'https://commons.wikimedia.org/wiki/File:Flooded_street_near_13th_Ave_and_J_Street_Cedar_Rapids_Iowa.jpg'),
    ('July_2023_flood', 'United States Military Academy', 'Public domain in the US', 'https://commons.wikimedia.org/wiki/File:July_2023_flood_damage_in_West_Point.jpg'),
    ('Sunken_cars_on', 'Don Becker/USGS', 'Public domain in the US', 'https://commons.wikimedia.org/wiki/File:Sunken_cars_on_flooded_street_in_Cedar_Rapids_Iowa.jpg'),
    ('Two_cars_stuck', 'Bidgee', 'CC BY-SA 2.5 Australia', 'https://commons.wikimedia.org/wiki/File:Two_cars_stuck_on_the_flooded_Berry_Street.jpg'),
    ('UAE_Flood', 'CherryPie94', 'CC0', 'https://commons.wikimedia.org/wiki/File:UAE_Flood_-_16_April_2024.jpg'),
]


if not CHECKPOINT_PATH.is_file():
    raise FileNotFoundError(
        f'Task 4 checkpoint not found: {CHECKPOINT_PATH}\n'
        'Check its Drive location before continuing.'
    )
if not SCENE_DIR.is_dir():
    raise FileNotFoundError(
        f'Scene folder not found: {SCENE_DIR}\n'
        'Create the folder in Drive and upload full-scene JPG/PNG images.'
    )

scene_paths = sorted(
    path for path in SCENE_DIR.rglob('*')
    if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS
)
if not scene_paths:
    raise RuntimeError(f'No JPG/PNG scene images found in {SCENE_DIR}')

bad_images = []
for image_path in scene_paths:
    try:
        with Image.open(image_path) as probe:
            probe.verify()
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        bad_images.append(f'{image_path}: {exc}')
if bad_images:
    raise RuntimeError('Unreadable scene file(s):\n' + '\n'.join(bad_images))

print(f'Ready: {len(scene_paths)} full-scene image(s)')
for image_path in scene_paths:
    print(' -', image_path.relative_to(SCENE_DIR))


image_credits = []
for path in scene_paths:
    match = next((row for row in credit_records if path.name.startswith(row[0])), None)
    if match:
        _, author, licence, url = match
        SOURCE_URLS[path.name] = url
    else:
        author, licence, url = 'Not supplied', 'Check before distributing', SOURCE_URLS.get(path.name, '')
    image_credits.append({'source_image': str(path.relative_to(SCENE_DIR)), 'author': author,
                          'license': licence, 'source_url': url,
                          'changes': 'Prediction boxes/text added to annotated copies; previews resized.'})
print(pd.DataFrame(image_credits).to_string(index=False))


# ## 3. Load the selected classifier and pretrained detector


# This checkpoint was created by our Task 4 notebook, so its metadata is trusted.
checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
required_keys = {
    'model_state_dict', 'class_names', 'image_size',
    'imagenet_mean', 'imagenet_std',
}
missing_keys = sorted(required_keys - set(checkpoint))
if missing_keys:
    raise KeyError(f'Checkpoint metadata missing: {missing_keys}')

class_names = list(checkpoint['class_names'])
if class_names != [f'Level{i}' for i in range(5)]:
    raise ValueError(f'Unexpected class order in checkpoint: {class_names}')

image_size = int(checkpoint['image_size'])
if image_size != 224:
    raise ValueError(f'Unexpected Task 4 image size: {image_size}')

classifier_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=list(checkpoint['imagenet_mean']),
        std=list(checkpoint['imagenet_std']),
    ),
])

classifier = models.resnet50(weights=None)
classifier.fc = nn.Linear(classifier.fc.in_features, len(class_names))
classifier.load_state_dict(checkpoint['model_state_dict'], strict=True)
classifier = classifier.to(device).eval()

detector = YOLO('yolov8n.pt')  # Fixed COCO-pretrained weights; no detector training.
print('ResNet-50 loaded. Selected Task 4 epoch:', checkpoint.get('best_epoch'))
print('Task 4 validation macro F1:', checkpoint.get('best_val_macro_f1'))
print('Class order:', class_names)
print('YOLOv8n loaded for COCO car class ID 2')


run_id = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f_UTC')
run_dir = OUTPUT_ROOT / f'run_{run_id}'
run_dir.mkdir(parents=True, exist_ok=False)
evaluation_dir = run_dir / 'classifier_evaluation'
evaluation_dir.mkdir()
checkpoint_record = {key: checkpoint.get(key) for key in [
    'class_names','image_size','imagenet_mean','imagenet_std','seed','selection_metric',
    'best_epoch','best_val_macro_f1','epochs_run','batch_size','learning_rate',
    'weight_decay','freeze_backbone']}
(evaluation_dir/'checkpoint_metadata.json').write_text(json.dumps(checkpoint_record,indent=2),encoding='utf-8')
print('Final run folder:', run_dir)


# ## 4. Verify the official classifier dataset
# 
# The official lists assign 2,692 training, 335 validation and 340 test images. Only those listed paths are used. The additional images in the downloaded package remain excluded. We keep the approved split and record its related-crop limitation rather than presenting it as an entirely unseen-scene test.


from collections import Counter
from itertools import combinations

expected_counts = {'train': 2692, 'val': 335, 'test': 340}
expected_classes = {
    'train': [640, 564, 892, 356, 240],
    'val': [80, 70, 111, 44, 30],
    'test': [80, 72, 113, 45, 30],
}
split_frames = {}
for split, expected in expected_counts.items():
    split_path = DATASET_DIR / f'{split}.txt'
    if not split_path.is_file():
        raise FileNotFoundError(f'Missing official split: {split_path}')
    records = []
    for raw in split_path.read_text(encoding='utf-8-sig').splitlines():
        relative = raw.strip().replace(chr(92), '/')
        if not relative:
            continue
        relative_path = Path(relative)
        if relative_path.is_absolute() or '..' in relative_path.parts:
            raise ValueError(f'Unexpected split path: {relative}')
        class_name = relative_path.parts[0]
        if class_name not in class_names:
            raise ValueError(f'Unknown relative class: {class_name}')
        path = IMAGE_ROOT / relative_path
        if not path.is_file():
            raise FileNotFoundError(f'Missing dataset image: {path}')
        records.append({'relative_path': relative, 'path': path,
                        'class_name': class_name, 'label': class_names.index(class_name)})
    frame = pd.DataFrame(records)
    assert len(frame) == expected, f'Unexpected {split} size'
    assert not frame['relative_path'].duplicated().any(), f'Duplicate {split} paths'
    counts = frame['class_name'].value_counts().reindex(class_names, fill_value=0).tolist()
    assert counts == expected_classes[split], f'Unexpected {split} class counts'
    split_frames[split] = frame
    print(f'{split}: {len(frame)} verified paths; class counts: {counts}')

split_sets = {k: set(v['relative_path']) for k, v in split_frames.items()}
for first, second in combinations(split_sets, 2):
    assert not split_sets[first] & split_sets[second], f'Identical paths across {first}/{second}'

def crop_group(relative):
    path = Path(relative)
    stem, separator, suffix = path.stem.rpartition('_')
    return (path.parent / stem).as_posix() if separator and suffix.isdigit() else relative

group_sets = {k: set(v['relative_path'].map(crop_group)) for k, v in split_frames.items()}
overlap_groups = {f'{a}/{b}': len(group_sets[a] & group_sets[b]) for a,b in combinations(group_sets,2)}
print('Shared filename base groups:', overlap_groups)
print('Shared groups are a related-crop limitation; they are not identical split paths.')
all_images = {p.relative_to(IMAGE_ROOT).as_posix() for p in IMAGE_ROOT.rglob('*')
              if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png'}}
listed = set().union(*split_sets.values())
dataset_record = {'split_counts': expected_counts, 'listed_total': len(listed),
                  'unlisted_excluded': len(all_images-listed), 'shared_filename_groups': overlap_groups,
                  'note': 'Official image-level split; related source crops may occur across splits.'}
(run_dir / 'dataset_verification.json').write_text(json.dumps(dataset_record,indent=2),encoding='utf-8')
print('Unlisted images excluded:', dataset_record['unlisted_excluded'])


# ## 5. Evaluate the saved classifier
# 
# The test transform matches the saved checkpoint: resize to 224 x 224, convert to a tensor and apply ImageNet normalization. Training augmentation was applied only during the earlier training run. This cell calculates accuracy, macro precision, macro recall, macro F1 and per-class metrics for the 340 labelled test crops.


import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class FinalTestDataset(Dataset):
    def __init__(self, frame, transform):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
    def __len__(self): return len(self.frame)
    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(row['path']) as im:
            tensor = self.transform(im.convert('RGB'))
        return tensor, int(row['label'])

test_frame = split_frames['test']
test_loader = DataLoader(FinalTestDataset(test_frame, classifier_transform),
                         batch_size=32, shuffle=False, num_workers=0,
                         pin_memory=(device.type == 'cuda'))
y_true, y_pred, all_probabilities = [], [], []
classifier.eval()
with torch.inference_mode():
    for tensors, labels in test_loader:
        probabilities = torch.softmax(classifier(tensors.to(device)),dim=1).cpu()
        y_true.extend(labels.tolist())
        y_pred.extend(probabilities.argmax(dim=1).tolist())
        all_probabilities.extend(probabilities.tolist())
assert len(y_true) == 340
report = classification_report(y_true,y_pred,labels=list(range(5)),target_names=class_names,
                               output_dict=True,zero_division=0)
metrics = {'test_images': len(y_true), 'accuracy': float(accuracy_score(y_true,y_pred)),
           'macro_precision': report['macro avg']['precision'],
           'macro_recall': report['macro avg']['recall'], 'macro_f1': report['macro avg']['f1-score'],
           'checkpoint_best_epoch': checkpoint.get('best_epoch'),
           'checkpoint_best_val_macro_f1': checkpoint.get('best_val_macro_f1'),
           'evaluation_scope': 'Labelled STURM official test crops; not full-scene pipeline accuracy.'}
(evaluation_dir/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
(evaluation_dir/'classification_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
pd.DataFrame(report).transpose().to_csv(evaluation_dir/'classification_report.csv')
test_predictions = test_frame[['relative_path','class_name','label']].copy()
test_predictions['predicted_class'] = [class_names[i] for i in y_pred]
test_predictions['classifier_confidence'] = [max(p) for p in all_probabilities]
test_predictions['correct'] = np.asarray(y_true) == np.asarray(y_pred)
for j,name in enumerate(class_names):
    test_predictions['probability_'+name] = [p[j] for p in all_probabilities]
test_predictions.to_csv(evaluation_dir/'test_predictions.csv',index=False)
cm = confusion_matrix(y_true,y_pred,labels=list(range(5)))
pd.DataFrame(cm,index=class_names,columns=class_names).to_csv(evaluation_dir/'confusion_matrix.csv')
print(json.dumps(metrics,indent=2))
print(classification_report(y_true,y_pred,labels=list(range(5)),target_names=class_names,digits=4,zero_division=0))
plt.figure(figsize=(7,6))
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=class_names,yticklabels=class_names)
plt.xlabel('Predicted relative level'); plt.ylabel('True relative level')
plt.title('Saved ResNet-50: official STURM test split'); plt.tight_layout()
plt.savefig(evaluation_dir/'confusion_matrix.png',dpi=180,bbox_inches='tight')
plt.show()


# ## 6. Record the completed training run
# 
# The following curves come from the saved Task 4 training history, not a new training run. The classifier reproduction notebook in the source package contains preprocessing, augmentation, training, macro-F1 checkpoint selection and test evaluation code.


if not TRAINING_HISTORY_PATH.is_file():
    raise FileNotFoundError(f'Training history missing: {TRAINING_HISTORY_PATH}')
history_df = pd.read_csv(TRAINING_HISTORY_PATH)
best_row = history_df.sort_values(['val_macro_f1','val_loss'],ascending=[False,True]).iloc[0]
assert int(best_row['epoch']) == checkpoint['best_epoch']
assert abs(float(best_row['val_macro_f1']) - checkpoint['best_val_macro_f1']) < 1e-8
history_df.to_csv(evaluation_dir/'training_history.csv',index=False)
fig,axes = plt.subplots(1,3,figsize=(15,4))
for axis,key,label in zip(axes,['loss','accuracy','macro_f1'],['Loss','Accuracy','Macro F1']):
    axis.plot(history_df['epoch'],history_df['train_'+key],marker='o',label='Train')
    axis.plot(history_df['epoch'],history_df['val_'+key],marker='o',label='Validation')
    axis.set_xlabel('Epoch'); axis.set_title(label); axis.legend()
fig.tight_layout(); fig.savefig(evaluation_dir/'training_curves.png',dpi=180,bbox_inches='tight'); plt.show()
print('Recorded selected training epoch:', int(best_row['epoch']))


# ## 7. Integrated car detection and flood-level classification


@torch.inference_mode()
def classify_crop(crop_bgr):
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    tensor = classifier_transform(Image.fromarray(crop_rgb)).unsqueeze(0).to(device)
    probabilities = torch.softmax(classifier(tensor), dim=1)[0]
    class_index = int(probabilities.argmax().item())
    return class_names[class_index], float(probabilities[class_index].item())


def expanded_crop_box(x1, y1, x2, y2, image_width, image_height):
    pad_x = int(round(CROP_PADDING_FRACTION * max(1, x2 - x1)))
    pad_y = int(round(CROP_PADDING_FRACTION * max(1, y2 - y1)))
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(image_width, x2 + pad_x),
        min(image_height, y2 + pad_y),
    )


def draw_detection(image_bgr, box, label, detector_confidence, classifier_confidence):
    x1, y1, x2, y2 = box
    color = (0, 170, 0) if classifier_confidence >= CLASSIFIER_CONFIDENCE else (0, 165, 255)
    cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, 2)
    text = f'{label} | det {detector_confidence:.2f} | cls {classifier_confidence:.2f}'
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.5, 1
    text_w, text_h = cv2.getTextSize(text, font, scale, thickness)[0]
    background_top = max(0, y1 - text_h - 9)
    background_bottom = min(image_bgr.shape[0] - 1, background_top + text_h + 8)
    background_right = min(image_bgr.shape[1] - 1, x1 + text_w + 8)
    cv2.rectangle(image_bgr, (x1, background_top), (background_right, background_bottom), color, -1)
    cv2.putText(image_bgr, text, (x1 + 4, background_bottom - 5),
                font, scale, (0, 0, 0), thickness, cv2.LINE_AA)


def process_scene(
    image_path, image_id,
    detector_confidence=DETECTOR_CONFIDENCE,
    detector_iou=DETECTOR_IOU,
    detector_imgsz=640,
):
    frame_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise RuntimeError(f'OpenCV could not decode {image_path}')
    annotated_bgr = frame_bgr.copy()
    image_height, image_width = frame_bgr.shape[:2]
    result = detector.predict(
        source=frame_bgr,
        classes=[CAR_CLASS_ID],
        conf=detector_confidence,
        iou=detector_iou,
        imgsz=detector_imgsz,
        device=0 if device.type == 'cuda' else 'cpu',
        verbose=False,
    )[0]
    records = []
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return annotated_bgr, records, (image_width, image_height)

    for detection_number, box in enumerate(boxes, start=1):
        x1, y1, x2, y2 = [int(round(v)) for v in box.xyxy[0].cpu().tolist()]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image_width, x2), min(image_height, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        crop_x1, crop_y1, crop_x2, crop_y2 = expanded_crop_box(
            x1, y1, x2, y2, image_width, image_height
        )
        crop_bgr = frame_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
        if crop_bgr.size == 0:
            continue

        detector_confidence = float(box.conf[0].cpu().item())
        level_name, classifier_confidence = classify_crop(crop_bgr)
        uncertain = classifier_confidence < CLASSIFIER_CONFIDENCE
        display_label = f'Uncertain ({level_name})' if uncertain else level_name
        draw_detection(
            annotated_bgr, (x1, y1, x2, y2), display_label,
            detector_confidence, classifier_confidence
        )
        records.append({
            'image_id': image_id,
            'source_image': str(image_path.relative_to(SCENE_DIR)),
            'detection_number': detection_number,
            'detector_class': 'car',
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            'crop_x1': crop_x1, 'crop_y1': crop_y1,
            'crop_x2': crop_x2, 'crop_y2': crop_y2,
            'detector_confidence': round(detector_confidence, 6),
            'relative_flood_level': level_name,
            'classifier_confidence': round(classifier_confidence, 6),
            'uncertain': uncertain,
        })
    return annotated_bgr, records, (image_width, image_height)


# Verify the inference defaults before running either profile.
import inspect
signature = inspect.signature(process_scene)
assert signature.parameters['detector_confidence'].default == DETECTOR_CONFIDENCE
assert signature.parameters['detector_iou'].default == DETECTOR_IOU
assert signature.parameters['detector_imgsz'].default == 640
assert expanded_crop_box(0,0,10,10,20,20) == (0,0,12,12)
assert expanded_crop_box(15,15,20,20,20,20) == (14,14,20,20)
assert expanded_crop_box(5,5,15,15,20,20) == (3,3,17,17)
print('Function defaults and crop-boundary checks passed.')


# ## 8. Run the baseline on all full-scene images


input_copy_dir = run_dir / 'input_images'
annotated_dir = run_dir / 'annotated_images'
input_copy_dir.mkdir(parents=True, exist_ok=False)
annotated_dir.mkdir(parents=True, exist_ok=False)

detection_columns = [
    'image_id', 'source_image', 'detection_number', 'detector_class',
    'x1', 'y1', 'x2', 'y2',
    'crop_x1', 'crop_y1', 'crop_x2', 'crop_y2',
    'detector_confidence', 'relative_flood_level',
    'classifier_confidence', 'uncertain',
]
summary_columns = [
    'image_id', 'source_image', 'source_url', 'image_width', 'image_height',
    'detected_cars', 'uncertain_predictions', 'mean_detector_confidence',
    'mean_classifier_confidence', 'input_copy', 'annotated_image',
]

all_detections = []
image_summaries = []
for image_index, image_path in enumerate(scene_paths, start=1):
    safe_stem = re.sub(r'[^A-Za-z0-9_-]+', '_', image_path.stem).strip('_')[:50]
    image_id = f'{image_index:03d}_{safe_stem or "image"}'
    input_copy = input_copy_dir / f'{image_id}{image_path.suffix.lower()}'
    annotated_path = annotated_dir / f'{image_id}_annotated.jpg'
    shutil.copy2(image_path, input_copy)

    annotated_bgr, rows, (image_width, image_height) = process_scene(image_path, image_id)
    if not cv2.imwrite(str(annotated_path), annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, 94]):
        raise OSError(f'Could not save annotated image: {annotated_path}')
    all_detections.extend(rows)
    source_relative = str(image_path.relative_to(SCENE_DIR))
    image_summaries.append({
        'image_id': image_id,
        'source_image': source_relative,
        'source_url': SOURCE_URLS.get(image_path.name, ''),
        'image_width': image_width,
        'image_height': image_height,
        'detected_cars': len(rows),
        'uncertain_predictions': sum(bool(row['uncertain']) for row in rows),
        'mean_detector_confidence': round(float(np.mean([row['detector_confidence'] for row in rows])), 4) if rows else None,
        'mean_classifier_confidence': round(float(np.mean([row['classifier_confidence'] for row in rows])), 4) if rows else None,
        'input_copy': str(input_copy.relative_to(run_dir)),
        'annotated_image': str(annotated_path.relative_to(run_dir)),
    })
    print(f'{image_index:>2}/{len(scene_paths)} {source_relative}: {len(rows)} car(s), '
          f'{image_summaries[-1]["uncertain_predictions"]} uncertain')

detections_df = pd.DataFrame(all_detections, columns=detection_columns)
summary_df = pd.DataFrame(image_summaries, columns=summary_columns)
detections_path = run_dir / 'detections.csv'
summary_path = run_dir / 'image_summary.csv'
detections_df.to_csv(detections_path, index=False)
summary_df.to_csv(summary_path, index=False)

metadata = {
    'project_title': 'Computer Vision-Based Flood Detection and Water-Level Monitoring System',
    'team_number': 20,
    'task': 7,
    'baseline_detector_imgsz': 640,
    'run_id': run_id,
    'timestamp_utc': datetime.now(timezone.utc).isoformat(),
    'scene_count': len(scene_paths),
    'total_car_detections': len(detections_df),
    'device': str(device),
    'torch_version': torch.__version__,
    'ultralytics_version': ultralytics.__version__,
    'detector_weights': 'yolov8n.pt (COCO pretrained)',
    'detector_class': 'car (COCO class 2)',
    'detector_confidence_threshold': DETECTOR_CONFIDENCE,
    'detector_iou_threshold': DETECTOR_IOU,
    'classifier_confidence_threshold': CLASSIFIER_CONFIDENCE,
    'crop_padding_fraction': CROP_PADDING_FRACTION,
    'checkpoint_path': str(CHECKPOINT_PATH),
    'checkpoint_best_epoch': checkpoint.get('best_epoch'),
    'checkpoint_best_val_macro_f1': checkpoint.get('best_val_macro_f1'),
    'class_names': class_names,
    'image_size': image_size,
    'note': 'Unlabelled full-scene examples; no ground-truth accuracy metrics are calculated.',
}
metadata_path = run_dir / 'run_metadata.json'
metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
print('Saved run folder:', run_dir)
print('Saved detection table:', detections_path)
print('Saved image summary:', summary_path)
print('Saved run metadata:', metadata_path)


pd.DataFrame(image_credits).to_csv(run_dir / 'image_credits.csv', index=False)
print('Saved image credits:', run_dir / 'image_credits.csv')


# Static text avoids Colab's interactive DataFrame reference errors.
print(summary_df.to_string(index=False))
print('\nTotal scenes:', len(summary_df))
print('Total car detections:', len(detections_df))
print('Scenes with no car detected:', int((summary_df['detected_cars'] == 0).sum()))
if not detections_df.empty:
    print('Uncertain predictions:', int(detections_df['uncertain'].sum()))
    print('Predicted level counts:')
    print(detections_df['relative_flood_level'].value_counts().reindex(class_names, fill_value=0).to_string())
else:
    print('No cars were detected at the current threshold; inspect the images before drawing conclusions.')


# ## 9. Compare the exploratory settings
# 
# The comparison reuses the Task 6 profiles: baseline confidence 0.30 / size 640 and exploratory confidence 0.15 / size 1280. Both profiles keep IoU 0.45, the same classifier and 20% padding. Counts are diagnostic. Two settings change together, so the comparison cannot isolate their individual effects or establish an accuracy improvement. The baseline remains the default.


candidate_dir = run_dir / 'exploratory_profile'
candidate_dir.mkdir(exist_ok=True)
candidate_settings = {'confidence': 0.15, 'iou': DETECTOR_IOU, 'imgsz': 1280}
candidate_records, comparison_rows = [], []
for image_path, baseline in zip(scene_paths, image_summaries):
    image_id = baseline['image_id']
    started = time.perf_counter()
    annotated, rows, _ = process_scene(image_path, image_id,
        detector_confidence=candidate_settings['confidence'],
        detector_iou=candidate_settings['iou'], detector_imgsz=candidate_settings['imgsz'])
    seconds = time.perf_counter() - started
    target = candidate_dir / f'{image_id}_annotated.jpg'
    if not cv2.imwrite(str(target), annotated, [cv2.IMWRITE_JPEG_QUALITY, 94]):
        raise OSError(f'Cannot save {target}')
    candidate_records.extend(rows)
    comparison_rows.append({
        'image_id': image_id, 'source_image': baseline['source_image'],
        'baseline_detections': baseline['detected_cars'], 'exploratory_detections': len(rows),
        'count_change': len(rows) - baseline['detected_cars'],
        'baseline_uncertain': baseline['uncertain_predictions'],
        'exploratory_uncertain': sum(row['uncertain'] for row in rows),
        'exploratory_seconds': round(seconds, 3),
        'baseline_output': baseline['annotated_image'],
        'exploratory_output': str(target.relative_to(run_dir)),
    })
    print(image_id, '| baseline:', baseline['detected_cars'], '| exploratory:', len(rows))
comparison_df = pd.DataFrame(comparison_rows)
candidate_df = pd.DataFrame(candidate_records, columns=detection_columns)
comparison_df.to_csv(run_dir / 'profile_comparison.csv', index=False)
candidate_df.to_csv(candidate_dir / 'detections.csv', index=False)
metadata['exploratory_settings'] = candidate_settings
metadata['exploratory_car_detections'] = len(candidate_df)
metadata['profile_comparison_note'] = 'Unlabelled diagnostic comparison; not evidence of an accuracy improvement.'
metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
print(comparison_df[['image_id', 'baseline_detections', 'exploratory_detections',
                     'count_change', 'baseline_uncertain', 'exploratory_uncertain']].to_string(index=False))
print('Timing is diagnostic only: it includes device transfers and may include first-run overhead.')


# ## 10. Scene outputs for the final report
# 
# The gallery shows the original scene, baseline output and exploratory output together. These saved images support a qualitative discussion of visible missed cars, partial boxes and questionable levels in the final report. There is no required review CSV to fill. The gallery itself does not supply ground-truth annotations.


from PIL import ImageDraw
gallery_dir = run_dir / 'review_gallery'
gallery_dir.mkdir()
for summary_row, comparison_row in zip(image_summaries,comparison_rows):
    sheet = Image.new('RGB',(1500,430),'white')
    draw = ImageDraw.Draw(sheet)
    views = [('Original',summary_row['input_copy']),
             ('Baseline',comparison_row['baseline_output']),
             ('Exploratory',comparison_row['exploratory_output'])]
    for column,(label,relative) in enumerate(views):
        with Image.open(run_dir/relative) as im:
            preview = ImageOps.contain(im.convert('RGB'),(490,390))
            sheet.paste(preview,(column*500+(500-preview.width)//2,35+(390-preview.height)//2))
        draw.text((column*500+10,10),label,fill='black')
    target = gallery_dir / f"{summary_row['image_id']}.jpg"
    sheet.save(target,quality=94)
    print(summary_row['image_id'], '| baseline:',comparison_row['baseline_detections'],
          '| exploratory:',comparison_row['exploratory_detections'])
    display(sheet)


# ## 11. Interactive image-testing panel
# 
# Choose an image and click **Run selected image** to test the prototype. Confidence and inference size can be adjusted. The classifier flag stays at 0.50. Each click saves its JPG, CSV and settings JSON. Batch outputs are kept separately. If you use this panel after packaging, rerun Sections 12 and 14.


import ipywidgets as widgets
from google.colab import output as colab_output
colab_output.enable_custom_widget_manager()

scene_selector = widgets.Dropdown(options=[(str(p.relative_to(SCENE_DIR)), str(p)) for p in scene_paths],
                                  description='Image:', layout=widgets.Layout(width='95%'))
confidence_slider = widgets.FloatSlider(value=DETECTOR_CONFIDENCE, min=0.05, max=0.80,
                                        step=0.05, description='Det. conf:', readout_format='.2f')
size_selector = widgets.Dropdown(options=[640, 960, 1280], value=640, description='Size:')
run_button = widgets.Button(description='Run selected image', button_style='')
panel_output = widgets.Output()
interactive_dir = run_dir / 'interactive_tests'
interactive_dir.mkdir(exist_ok=True)

def run_selected_image(_):
    run_button.disabled = True
    try:
        with panel_output:
            clear_output(wait=True)
            selected = Path(scene_selector.value)
            test_id = datetime.now(timezone.utc).strftime('test_%Y%m%d_%H%M%S_%f')
            annotated, records, _ = process_scene(selected, test_id,
                detector_confidence=float(confidence_slider.value),
                detector_imgsz=int(size_selector.value))
            image_file = interactive_dir / f'{test_id}.jpg'
            if not cv2.imwrite(str(image_file), annotated):
                raise OSError(f'Cannot save {image_file}')
            pd.DataFrame(records, columns=detection_columns).to_csv(interactive_dir / f'{test_id}.csv', index=False)
            settings = {'source_image': str(selected.relative_to(SCENE_DIR)),
                        'detector_confidence': float(confidence_slider.value),
                        'detector_iou': DETECTOR_IOU, 'detector_imgsz': int(size_selector.value),
                        'classifier_uncertainty_threshold': CLASSIFIER_CONFIDENCE,
                        'detections': len(records), 'timestamp_utc': datetime.now(timezone.utc).isoformat()}
            (interactive_dir / f'{test_id}.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
            print('Detections:', len(records), '| uncertain:', sum(r['uncertain'] for r in records))
            with Image.open(image_file) as im:
                display(ImageOps.contain(im.convert('RGB'), (950, 650)).copy())
            if records:
                print(pd.DataFrame(records)[['relative_flood_level', 'detector_confidence',
                                             'classifier_confidence', 'uncertain']].to_string(index=False))
            print('Saved:', image_file)
    except Exception as exc:
        with panel_output:
            print(type(exc).__name__ + ':', exc)
    finally:
        run_button.disabled = False

run_button.on_click(run_selected_image)
display(widgets.VBox([scene_selector, confidence_slider, size_selector, run_button, panel_output]))


# ## 12. Final integration checks and completion record
# 
# These checks verify that the saved results are internally consistent: one summary per scene, matching detection counts, valid boxes and probability ranges, correct uncertainty flags and readable output files. They do not establish the correctness of predicted cars or flood categories.


from collections import defaultdict
baseline_by_id, exploratory_by_id = defaultdict(list),defaultdict(list)
for records,group in [(all_detections,baseline_by_id),(candidate_records,exploratory_by_id)]:
    for row in records: group[row['image_id']].append(row)
assert len(image_summaries) == len(scene_paths) == len(comparison_rows)
assert len({s['image_id'] for s in image_summaries}) == len(scene_paths)
for summary_row, comparison_row in zip(image_summaries,comparison_rows):
    image_id = summary_row['image_id']
    assert comparison_row['image_id'] == image_id
    w,h = summary_row['image_width'],summary_row['image_height']
    for group,confidence,prefix in [(baseline_by_id,DETECTOR_CONFIDENCE,'baseline'),
                                    (exploratory_by_id,candidate_settings['confidence'],'exploratory')]:
        rr = group[image_id]
        assert len(rr) == comparison_row[prefix+'_detections']
        assert sum(r['uncertain'] for r in rr) == comparison_row[prefix+'_uncertain']
        for r in rr:
            assert r['relative_flood_level'] in class_names
            assert confidence-1e-6 <= r['detector_confidence'] <= 1
            assert 0 <= r['classifier_confidence'] <= 1
            # Flags use the original floating-point value; CSV confidence is rounded.
            if abs(r['classifier_confidence']-CLASSIFIER_CONFIDENCE) > 1e-6:
                assert r['uncertain'] == (r['classifier_confidence'] < CLASSIFIER_CONFIDENCE)
            assert 0 <= r['x1'] < r['x2'] <= w and 0 <= r['y1'] < r['y2'] <= h
            assert 0 <= r['crop_x1'] < r['crop_x2'] <= w
            assert 0 <= r['crop_y1'] < r['crop_y2'] <= h
    for path in [summary_row['input_copy'],comparison_row['baseline_output'],comparison_row['exploratory_output']]:
        with Image.open(run_dir/path) as im:
            im.load(); assert im.size == (w,h)
assert len(test_predictions)==340 and sum(sum(row) for row in cm)==340
assert len(image_credits)==len(scene_paths)
interactive_jsons = sorted((run_dir/'interactive_tests').glob('*.json'))
for path in interactive_jsons:
    entry = json.loads(path.read_text(encoding='utf-8'))
    assert len(pd.read_csv(path.with_suffix('.csv'))) == entry['detections']
    with Image.open(path.with_suffix('.jpg')) as im: im.load()

checks = {'status':'passed','test_crops':340,'scene_images':len(scene_paths),
          'baseline_detections':len(all_detections),'exploratory_detections':len(candidate_records),
          'interactive_tests_saved':len(interactive_jsons),
          'scope':'File consistency and integration checks; not prediction accuracy.'}
(run_dir/'integration_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
metadata['classifier_test_metrics'] = metrics
metadata['dataset_verification'] = dataset_record
metadata['interactive_tests_saved'] = len(interactive_jsons)
metadata['integration_checks'] = checks
metadata['final_scope'] = 'Still-image car detection and relative flood-level classification; video optional.'
metadata_path.write_text(json.dumps(metadata,indent=2),encoding='utf-8')
print(json.dumps(checks,indent=2))


# ## 13. Final scope and limitations
# 
# After the cells have run successfully, the implemented core methodology consists of official-split dataset preparation, deterministic test preprocessing, the previously trained classifier selected by validation macro F1, classifier evaluation, pretrained car detection, crop classification, confidence filtering, output visualization and integrated image testing.
# 
# The classifier training source and earlier training history are supplied with the project. The system can still miss severely submerged or distant cars, produce partial boxes, and misclassify detector crops. These are limitations to describe in the final report; completion of the implementation does not mean every prediction is correct. No full-scene accuracy metrics are claimed without labelled scene annotations. Related crops in the STURM split limit unseen-scene generalization claims.
# 
# The final report will include the classifier metrics, scene examples and qualitative observations. No exact water depth is measured. Video processing and detector retraining on a newly annotated flooded-scene dataset are possible future extensions beyond this completed still-image scope.


# ## 14. Package the final results


# Results package only: the source code is in the separately supplied source ZIP.
zip_base = PROJECT_DIR / f'task7_results_{run_id}'
zip_path = Path(shutil.make_archive(str(zip_base),'zip',root_dir=run_dir))
print('Final results ZIP:',zip_path)
print('ZIP size (MB):',round(zip_path.stat().st_size/(1024**2),2))
print('Download this ZIP and save/download the executed notebook with all outputs.')
print('If more interactive tests are performed, rerun the final checks and packaging cells.')
