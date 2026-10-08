# Third-party notices

## Project source

The integrated source in this repository is offered under GNU Affero General Public License version 3 only (AGPL-3.0-only). The licence text is in `LICENSE`. This source includes training, evaluation, preprocessing, inference and the Gradio interface; no Ultralytics modifications are made.

## Models and libraries

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics): AGPL-3.0. This project uses the COCO-pretrained `yolov8n.pt` model without retraining it. See [Ultralytics' publishing guidance](https://docs.ultralytics.com/help/contributing/).
- [PyTorch](https://github.com/pytorch/pytorch) and [Torchvision](https://github.com/pytorch/vision): retain their respective upstream BSD-style terms. ResNet-50 is initialized from Torchvision's ImageNet weights for classifier training.
- [Gradio](https://github.com/gradio-app/gradio): Apache-2.0.
- NumPy, pandas, scikit-learn, OpenCV, Pillow, Matplotlib, seaborn and ipywidgets retain their upstream licences. Installing a dependency does not transfer ownership of it to the project team.

Check upstream releases for the complete dependency notices. The repository does not bundle these libraries or pretrained weight files.

## Dataset and photographs

The [STURM-FloodDepth Flooded Cars record](https://zenodo.org/records/14833532) is the classifier dataset source. Download it from the publisher. Dataset images and original-source bounding-box material are not redistributed here, and the software licence does not grant rights over that material.

Photograph attribution is in `docs/image_credits.csv`. When photographs appear in the report or selected result assets, their original public-domain, CC0 or CC BY-SA terms continue to apply. Annotated copies add prediction boxes/text and may be resized. Attribution and applicable share-alike terms must be retained when redistributing those derivatives.

The original analysis results are not manually labelled ground truth. Their model predictions are not guarantees of safe passage or measured water depth.
