# Setup and execution

## 1. Choose the notebook

Use `notebooks/gradio_demo.ipynb` to present the image-upload interface. Use `notebooks/project_evaluation.ipynb` for the classifier test and integrated scene experiments. Use `notebooks/classifier_training.ipynb` only when reproducing classifier training or when no trained checkpoint is available.

Open the chosen notebook in Google Colab and select a GPU runtime if available. The source exports in `src/` use Colab-specific Drive mounting and display features; they are not standalone local applications.

## 2. Prepare Drive

The existing experiment expects:

```text
MyDrive/CSE411_Flood_Project/
  STURM-FloodDepth/
    train.txt
    val.txt
    test.txt
    flooded_cars/upscaled_images/
      Level0/
      Level1/
      Level2/
      Level3/
      Level4/
  task4_outputs/
    resnet50_sturm_best_macro_f1.pth
    resnet50_macro_f1_training_history.csv
  task5_scene_images/
    your_full_scene_photographs.jpg
```

Download **all four** files from [STURM on Zenodo](https://zenodo.org/records/14833532): `flooded_cars.zip`, `train.txt`, `val.txt` and `test.txt`. Extract the ZIP; do not flatten the class folders. The text files contain paths, not image pixels. Use the official 2,692 / 335 / 340 lists, rather than automatically training on every image in the package.

The Gradio demo needs only the saved classifier checkpoint; the dataset is not required for inference. The scene folder is optional because the interface accepts uploads. Do not use the STURM crop set as a full-scene YOLO training set.

## 3. Checkpoint availability

The evaluated checkpoint is not bundled or hosted in this repository. The project owner/evaluator must provide the saved `.pth` file and matching training history through an approved channel. Only load a checkpoint you trust; the original experiment's loader uses PyTorch serialization with `weights_only=False`.

If that checkpoint is unavailable, run `classifier_training.ipynb`. It writes a new checkpoint and history in `classifier_reproduction_outputs/`, preserving the earlier experiment. Point both `CHECKPOINT_PATH` and `TRAINING_HISTORY_PATH` in the evaluation notebook to that same new training run. For the demo, update `CHECKPOINT_PATH` only. Do not combine a new checkpoint with an older history file.

The new model is a reproduction, not the exact evaluated artifact. Five epochs, seed 42 and the same architecture do not guarantee the published scores across different hardware and software. YOLOv8n's pretrained weights download automatically on first use.

## 4. Run the demo

1. Run Sections 1–8 in order. Authorize your own Drive access and confirm the checkpoint path.
2. Keep Colab's compatible PyTorch/Torchvision pair. Do not install an old Pillow workaround or replace just one of the paired libraries.
3. Open the temporary public link printed in Section 8, then use its generated username/password.
4. Upload a full-scene image or load an example. Click **Analyse image**.
5. Compare the original and annotated scene. Inspect **Predictions**, **Car crops** and **Downloads**. Default detector settings are confidence 0.30 / size 640.
6. Download the PNG, CSV and JSON or enable Section 9's session backup after testing. The clean public source defaults to no backup until enabled.
7. Clear the launch-cell output before sharing an executed notebook publicly. Do not publish its login credentials.

The link ends when its serving runtime ends; it is not permanent hosting. GitHub Pages is not used for Python model inference.

## 5. Reconnecting or replacing an interface

If the same Colab runtime reconnects and still holds the loaded models and callbacks, rerun only the interface-building and launch cells. A reset/new runtime requires the setup and model-loading cells again, but does not require retraining a model that is already saved on Drive. Unbacked-up files in the old temporary runtime may be lost.

Finish any current analysis before closing/rebuilding the interface. Use the new link and login after relaunching; the old interface link is no longer the active page.

## 6. Repository checks

```bash
python -m unittest discover -s tests -v
```

These standard-library checks require no GPU, model files or dataset download. They check syntax, notebook/export parity, fixed settings, cropping, clean source, source-credit coverage and saved-result counts. They do not substitute for running the notebooks with the actual models.
