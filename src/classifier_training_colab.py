# SPDX-License-Identifier: AGPL-3.0-only

"""Google Colab source export; use the corresponding notebook to run it."""


# # Team 20 - Classifier training reproduction
# This optional notebook reproduces the five-epoch STURM classifier procedure. The final Task 7 notebook already uses the saved checkpoint.
# Reproduction outputs are saved separately in `classifier_reproduction_outputs`.
# 2023BCS0235 Mahendra K; 2023BCS0232 Sanin K; 2023BCS0238 Bhukya Gandhi; 2022BCD0001 Ankit Subhash.


# Install the notebook dependencies before running this source.


import copy
import json
import random
import time
import warnings
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image, ImageFile

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

warnings.filterwarnings('ignore')
ImageFile.LOAD_TRUNCATED_IMAGES = False

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'PyTorch device: {device}')


# ## 1. Mount Google Drive and configure the dataset
# 
# The expected Drive layout is: `MyDrive/CSE411_Flood_Project/STURM-FloodDepth/`. If you use another location, change `DATASET_DIR` below.


from google.colab import drive
drive.mount('/content/drive')


# Change only this line if your Drive folder has a different location.
DATASET_DIR = Path('/content/drive/MyDrive/CSE411_Flood_Project/STURM-FloodDepth')
IMAGE_ROOT = DATASET_DIR / 'flooded_cars' / 'upscaled_images'
SPLIT_FILES = {name: DATASET_DIR / f'{name}.txt' for name in ['train', 'val', 'test']}
RESULTS_DIR = DATASET_DIR.parent / 'classifier_reproduction_outputs'
CLASS_NAMES = [f'Level{i}' for i in range(5)]
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)
IMAGE_SIZE = 224

print('Dataset directory:', DATASET_DIR)
print('Image root:', IMAGE_ROOT)
print('Dataset directory exists:', DATASET_DIR.exists())
print('Image root exists:', IMAGE_ROOT.exists())
for split_name, split_path in SPLIT_FILES.items():
    print(f'{split_name:>5} list exists:', split_path.exists(), split_path)


# ## 2. Read and verify the official train/validation/test lists
# 
# The text files contain relative image paths, not image pixels. The first folder name in each path supplies the class label.


def normalize_split_line(line):
    # The downloaded lists use Windows separators; Colab uses POSIX paths.
    return line.strip().replace(chr(92), '/').lstrip('/')


def load_split(split_name):
    split_path = SPLIT_FILES[split_name]
    rows = []
    with split_path.open('r', encoding='utf-8-sig') as handle:
        for raw_line in handle:
            relative_path = normalize_split_line(raw_line)
            if not relative_path:
                continue
            image_path = IMAGE_ROOT / Path(relative_path)
            class_name = Path(relative_path).parts[0]
            rows.append({
                'split': split_name,
                'relative_path': relative_path,
                'path': image_path,
                'class_name': class_name,
                'label': CLASS_TO_INDEX.get(class_name, -1),
            })
    return pd.DataFrame(rows)

split_frames = {name: load_split(name) for name in ['train', 'val', 'test']}
train_df = split_frames['train']
val_df = split_frames['val']
test_df = split_frames['test']
print(train_df[['split', 'relative_path', 'class_name', 'label']].head().to_string(index=False))


EXPECTED_SPLIT_COUNTS = {'train': 2692, 'val': 335, 'test': 340}
EXPECTED_CLASS_COUNTS = {
    'train': {'Level0': 640, 'Level1': 564, 'Level2': 892, 'Level3': 356, 'Level4': 240},
    'val': {'Level0': 80, 'Level1': 70, 'Level2': 111, 'Level3': 44, 'Level4': 30},
    'test': {'Level0': 80, 'Level1': 72, 'Level2': 113, 'Level3': 45, 'Level4': 30},
}

for split_name, frame in split_frames.items():
    missing = frame.loc[~frame['path'].map(Path.exists)]
    duplicate_count = int(frame['relative_path'].duplicated().sum())
    class_counts = frame['class_name'].value_counts().reindex(CLASS_NAMES, fill_value=0).to_dict()
    print(f'{split_name:>5}: {len(frame)} images | missing: {len(missing)} | duplicate rows: {duplicate_count}')
    print('      class counts:', class_counts)
    assert len(frame) == EXPECTED_SPLIT_COUNTS[split_name], f'Unexpected {split_name} count'
    assert len(missing) == 0, f'Missing files found in {split_name}'
    assert duplicate_count == 0, f'Duplicate paths found in {split_name}'
    assert class_counts == EXPECTED_CLASS_COUNTS[split_name], f'Unexpected class counts in {split_name}'

split_sets = {name: set(frame['relative_path']) for name, frame in split_frames.items()}
print('train/val overlap:', len(split_sets['train'] & split_sets['val']))
print('train/test overlap:', len(split_sets['train'] & split_sets['test']))
print('val/test overlap:', len(split_sets['val'] & split_sets['test']))
assert not (split_sets['train'] & split_sets['val'])
assert not (split_sets['train'] & split_sets['test'])
assert not (split_sets['val'] & split_sets['test'])

class_table = pd.DataFrame({name: frame['class_name'].value_counts().reindex(CLASS_NAMES, fill_value=0) for name, frame in split_frames.items()})
print(class_table.to_string())

all_image_paths = {path for path in IMAGE_ROOT.rglob('*') if path.is_file() and path.suffix.lower() in {'.jpg', '.jpeg', '.png'}}
listed_image_paths = set().union(*(set(frame['path']) for frame in split_frames.values()))
extra_image_paths = all_image_paths - listed_image_paths
print('All image files found in extracted folders:', len(all_image_paths))
print('Images assigned by official split files:', len(listed_image_paths))
print('Unlisted extra images (not used):', len(extra_image_paths))


# ### A note about the official splits
# 
# The checks above confirm that no exact file path is repeated between train, validation, and test. That does not guarantee that every crop comes from a different original photograph. Some files with the same base name, such as `car-1007-Level0_1.jpg` and `car-1007-Level0_4.jpg`, are placed in different splits. The check below counts these shared base names. We still use the official split, but describe its result as image-level evaluation.


def filename_group(relative_path):
    path = Path(relative_path)
    base, separator, suffix = path.stem.rpartition('_')
    if separator and suffix.isdigit():
        return (path.parent / base).as_posix()
    return relative_path


group_sets = {
    name: set(frame['relative_path'].map(filename_group))
    for name, frame in split_frames.items()
}
for first, second in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
    shared_groups = group_sets[first] & group_sets[second]
    second_crops = split_frames[second]['relative_path'].map(filename_group).isin(shared_groups).sum()
    print(f'{first}/{second}: {len(shared_groups)} shared base names; {second_crops} {second} crops in those groups')


# ## Inspect representative training crops
# Display sample images from each relative class folder.


def show_class_samples(frame, samples_per_class=3):
    fig, axes = plt.subplots(len(CLASS_NAMES), samples_per_class, figsize=(12, 16))
    axes = np.asarray(axes).reshape(len(CLASS_NAMES), samples_per_class)
    for row_index, class_name in enumerate(CLASS_NAMES):
        subset = frame[frame['class_name'] == class_name]
        sample_count = min(samples_per_class, len(subset))
        sampled = subset.sample(n=sample_count, random_state=SEED)
        for column_index in range(samples_per_class):
            axis = axes[row_index, column_index]
            axis.axis('off')
            if column_index < sample_count:
                item = sampled.iloc[column_index]
                image = Image.open(item['path']).convert('RGB')
                axis.imshow(image)
                axis.set_title(item['class_name'])
    plt.suptitle('STURM-FloodDepth samples by relative flood-level class', fontsize=16)
    plt.tight_layout()
    plt.show()

show_class_samples(train_df)


# ## 4. Preprocessing and augmentation
# 
# All images are resized to `224 x 224` and normalized with the ImageNet mean and standard deviation expected by the pretrained ResNet-50. Augmentation is applied only to the training split; validation and test images use deterministic preprocessing.


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=8),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class FloodLevelDataset(Dataset):
    def __init__(self, frame, transform=None):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        image = Image.open(row['path']).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        label = torch.tensor(int(row['label']), dtype=torch.long)
        return image, label


train_dataset = FloodLevelDataset(train_df, train_transform)
val_dataset = FloodLevelDataset(val_df, eval_transform)
test_dataset = FloodLevelDataset(test_df, eval_transform)

BATCH_SIZE = 32
NUM_WORKERS = 2
PIN_MEMORY = torch.cuda.is_available()

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

train_counts = train_df['class_name'].value_counts().reindex(CLASS_NAMES).to_numpy()
class_weights = train_counts.sum() / (NUM_CLASSES * train_counts)
class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)

sample_images, sample_labels = next(iter(train_loader))
print('One batch:', tuple(sample_images.shape), tuple(sample_labels.shape))
print('Class weights:', class_weights.detach().cpu().numpy())


# ## 5. ResNet-50 flood-level classifier
# 
# The backbone starts from ImageNet-pretrained weights, and the final fully connected layer is replaced with five outputs. The default setting fine-tunes the complete network. Set `FREEZE_BACKBONE = True` for a quick smoke test that trains only the new classification head.
# 
# The saved checkpoint is chosen by the highest validation macro F1. If two epochs tie, the one with lower validation loss is kept.


weights = ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

FREEZE_BACKBONE = False
if FREEZE_BACKBONE:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.fc.parameters():
        parameter.requires_grad = True

model = model.to(device)
trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
total_parameters = sum(parameter.numel() for parameter in model.parameters())
print(f'Trainable parameters: {trainable_parameters:,} / {total_parameters:,}')


criterion = nn.CrossEntropyLoss(weight=class_weights)


def run_epoch(model, loader, criterion, optimizer=None):
    is_training = optimizer is not None
    model.train(is_training)
    running_loss = 0.0
    all_labels = []
    all_predictions = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            logits = model(images)
            loss = criterion(logits, labels)
            if is_training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        running_loss += loss.item() * images.size(0)
        predictions = logits.argmax(dim=1)
        all_labels.extend(labels.detach().cpu().numpy().tolist())
        all_predictions.extend(predictions.detach().cpu().numpy().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_accuracy = accuracy_score(all_labels, all_predictions)
    epoch_macro_f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
    return epoch_loss, epoch_accuracy, epoch_macro_f1


def fit_model(model, train_loader, val_loader, criterion, epochs=5, learning_rate=1e-4, weight_decay=1e-4):
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.3, patience=1)
    history = []
    best_state = copy.deepcopy(model.state_dict())
    best_val_f1 = -float('inf')
    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        train_loss, train_accuracy, train_f1 = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_accuracy, val_f1 = run_epoch(model, val_loader, criterion)
        scheduler.step(val_f1)
        current_lr = optimizer.param_groups[0]['lr']

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'train_macro_f1': train_f1,
            'val_macro_f1': val_f1,
            'learning_rate': current_lr,
        })
        if val_f1 > best_val_f1 or (val_f1 == best_val_f1 and val_loss < best_val_loss):
            best_val_f1 = val_f1
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        elapsed = time.time() - start_time
        print(
            f'Epoch {epoch:02d}/{epochs} | '            f'train loss {train_loss:.4f}, acc {train_accuracy:.4f}, F1 {train_f1:.4f} | '            f'val loss {val_loss:.4f}, acc {val_accuracy:.4f}, F1 {val_f1:.4f} | '            f'lr {current_lr:.2e} | {elapsed:.1f}s'
        )

    return pd.DataFrame(history), best_state, best_epoch, best_val_f1


# ## Reproduce classifier training
# Fine-tune for five epochs and retain the epoch with highest validation macro F1.


EPOCHS = 5
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

history_df, best_state, best_epoch, best_val_f1 = fit_model(
    model,
    train_loader,
    val_loader,
    criterion,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)
model.load_state_dict(best_state)
model.eval()
print(f'Selected epoch {best_epoch} by validation macro F1 = {best_val_f1:.4f}')


fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(history_df['epoch'], history_df['train_loss'], marker='o', label='Train')
axes[0].plot(history_df['epoch'], history_df['val_loss'], marker='o', label='Validation')
axes[0].set_title('Cross-entropy loss')
axes[0].set_xlabel('Epoch')
axes[0].legend()

axes[1].plot(history_df['epoch'], history_df['train_accuracy'], marker='o', label='Train')
axes[1].plot(history_df['epoch'], history_df['val_accuracy'], marker='o', label='Validation')
axes[1].set_title('Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].legend()

axes[2].plot(history_df['epoch'], history_df['train_macro_f1'], marker='o', label='Train')
axes[2].plot(history_df['epoch'], history_df['val_macro_f1'], marker='o', label='Validation')
axes[2].set_title('Macro F1')
axes[2].set_xlabel('Epoch')
axes[2].legend()

for axis in axes:
    axis.grid(alpha=0.25)
plt.tight_layout()
plt.show()


# ## 7. Evaluate the classifier on held-out test images
# 
# The test set is used only after training and model selection. Report accuracy, macro F1, per-class precision/recall/F1, and a confusion matrix. These are classification metrics for relative flood-level categories.
# 
# These numbers describe the official image-level test split. Some crops appear to come from the same source photo across splits, so the score should not be described as performance on entirely new street scenes.


@torch.inference_mode()
def collect_predictions(model, loader):
    model.eval()
    labels = []
    predictions = []
    probabilities = []
    for images, batch_labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        labels.extend(batch_labels.numpy().tolist())
        predictions.extend(probs.argmax(dim=1).cpu().numpy().tolist())
        probabilities.append(probs.cpu())
    return np.asarray(labels), np.asarray(predictions), torch.cat(probabilities).numpy()

test_labels, test_predictions, test_probabilities = collect_predictions(model, test_loader)
print(f'Test accuracy: {accuracy_score(test_labels, test_predictions):.4f}')
test_macro_f1 = f1_score(test_labels, test_predictions, average='macro', zero_division=0)
print(f'Test macro F1: {test_macro_f1:.4f}')
print()
print(classification_report(test_labels, test_predictions, target_names=CLASS_NAMES, digits=4, zero_division=0))


cm = confusion_matrix(test_labels, test_predictions, labels=list(range(NUM_CLASSES)))
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel('Predicted relative level')
plt.ylabel('True relative level')
plt.title('ResNet-50 test confusion matrix')
plt.tight_layout()
plt.show()


RESULTS_DIR.mkdir(parents=True, exist_ok=True)
checkpoint_path = RESULTS_DIR / 'resnet50_sturm_best_macro_f1.pth'
checkpoint = {
    'model_state_dict': model.state_dict(),
    'class_names': CLASS_NAMES,
    'image_size': IMAGE_SIZE,
    'imagenet_mean': IMAGENET_MEAN,
    'imagenet_std': IMAGENET_STD,
    'seed': SEED,
    'selection_metric': 'validation_macro_f1',
    'best_epoch': best_epoch,
    'best_val_macro_f1': best_val_f1,
    'epochs_run': EPOCHS,
    'batch_size': BATCH_SIZE,
    'learning_rate': LEARNING_RATE,
    'weight_decay': WEIGHT_DECAY,
    'freeze_backbone': FREEZE_BACKBONE,
}
torch.save(checkpoint, checkpoint_path)
history_df.to_csv(RESULTS_DIR / 'resnet50_macro_f1_training_history.csv', index=False)
print('Saved classifier checkpoint:', checkpoint_path)
print('Saved training history:', RESULTS_DIR / 'resnet50_macro_f1_training_history.csv')
