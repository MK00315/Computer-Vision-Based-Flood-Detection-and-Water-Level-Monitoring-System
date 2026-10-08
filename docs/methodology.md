# Methodology

## Dataset and classifier development

STURM supplies vehicle-centred images with Level0–Level4 labels and official train, validation and test lists. The experiment uses 2,692 training, 335 validation and 340 test crops. The 412 additional unlisted images in the downloaded package are excluded.

ResNet-50 starts from ImageNet-pretrained Torchvision weights, with its final fully connected layer replaced by five outputs. All parameters are fine-tuned for five epochs, using batch size 32 and seed 42. The optimizer is AdamW with learning rate 0.0001 and weight decay 0.0001. The loss is inverse-frequency-weighted cross-entropy. Gradient norms are clipped at 5.0, and ReduceLROnPlateau monitors validation macro F1.

Training-only augmentation uses horizontal flipping, small affine changes and colour jitter. Validation and test processing use only resizing, tensor conversion and ImageNet normalization. Validation runs without gradient tracking. The highest validation macro F1 selects the checkpoint; lower validation loss is the tie-breaker. The evaluated checkpoint is epoch 5, with validation macro F1 of approximately 76.86%.

Related base filename groups occur across the official splits. This is a split-independence limitation, not proof that each overlap is an identical photograph. The official test metric should not be advertised as unseen-scene performance.

## Scene inference

1. Read the full-scene photograph.
2. Run COCO-pretrained YOLOv8n with class 2 (`car`), detector confidence filtering and IoU-based NMS. YOLO's weights are not updated.
3. Clip each valid detection to the original image. Add 20% of the box width on each horizontal side and 20% of the box height on each vertical side, then clip the padded rectangle.
4. Extract the crop from the **original unannotated pixels**. Convert BGR to RGB, resize to 224 × 224, convert to a tensor and normalize with ImageNet means `[0.485, 0.456, 0.406]` and standard deviations `[0.229, 0.224, 0.225]`.
5. Run the saved ResNet-50 and apply softmax. Select the maximum-scoring relative class.
6. Retain the prediction, but flag classifier scores below 0.50 as uncertain. Do not turn uncertainty into an extra flood class.
7. Draw boxes, car IDs, relative labels and scores; save prediction rows and settings/source-credit metadata.

Baseline detector settings are confidence 0.30, IoU 0.45 and image size 640. The integrated diagnostic comparison uses confidence 0.15 / size 1280. The Gradio page also allows other exploratory settings, including confidence 0.05. Changing two settings simultaneously does not isolate the effect of either setting.

## Evaluation and interface

The classifier test bypasses YOLO and evaluates labelled STURM crops. Full-scene demonstrations have no verified ground-truth boxes or class labels; their counts and visual comparisons are diagnostic, not end-to-end accuracy.

Gradio provides a password-protected upload interface around the same image pipeline. It displays the original, annotated result, prediction rows, padded crops and downloadable files. Model inference is serialized through one shared queue. Changing an input or detector setting clears the previous result to avoid stale predictions.

The completed scope is still-image processing and relative submergence categories. There is no exact centimetre/metre measurement, implemented temporal tracking, water segmentation or operational flood-warning service.

## Connection to the course

The implemented methods apply object detection, supervised classification, learned image features and neural networks. Cropping, colour conversion, resizing and normalization provide image-representation and transformation steps. The project does not claim to have implemented withdrawn or unused methods such as HOG/LBP/SVM, Hough transforms, SIFT or optical flow.
