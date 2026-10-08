# Results and interpretation

## Classifier test

The selected ResNet-50 checkpoint correctly classifies **254 of 340** official STURM test crops:

| Metric | Value |
| --- | --- |
| Accuracy | 74.71% |
| Macro precision | 75.39% |
| Macro recall | 76.26% |
| Macro F1 | 75.73% |

The original metric JSON, class-wise report, training history and plots are in `results/classifier/`. The classifier is selected by validation macro F1, not test performance. Level1/Level2 confusion is a notable weakness. Related-crop filename groups across the official split limit unseen-scene interpretation.

## Integrated scene experiment

The evaluated notebook processes ten unlabelled scene photographs. Its baseline returns 24 boxes; confidence 0.15 / size 1280 returns 40. Each profile has three uncertain classifications. The results archive also records thirteen interactive tests.

These numbers count predictions, not verified correct cars. More boxes can mean recovered cars, duplicates, partial vehicles or incorrect objects. No detector mAP, scene precision/recall or complete-system accuracy is calculated without verified ground-truth annotations.

## Gradio demonstration

The submitted Drive ZIP contains two backups of the same session: an earlier four-run snapshot and a later ten-run snapshot. The four earlier records are repeated in the latter. There are **10 unique analyses of 4 photographs**, with **13 prediction rows** and **2 uncertain rows**. Two analyses also repeat the Trebic photograph with the same settings.

`results/gradio/run_summary.csv` lists every unique saved analysis. Each `run_XX/` folder contains the original small prediction CSV and settings/source-credit JSON, without model weights or passwords. Full-resolution images and complete backup ZIPs remain in the course-submission evidence, not the Git repository.

Visual inspection shows useful boxes in the flooded-street example, an uncertain Trebic car prediction with other visible cars missed, and strongly setting-dependent Atcham detections. The heavily submerged Cedar Rapids scene is a limitation example. None of these observations establishes an accuracy score.

## Scope of verification

Saved files were checked for readable images, valid coordinates, expected class order, approved preprocessing, matching CSV/metadata counts, uncertainty consistency and ZIP integrity. The latest notebook has no saved cell-error outputs or printed demo password. Its launch output is cleared, so the saved file itself is not a record of an active public URL.

These checks establish file and integration consistency. They do not prove that every box or class is correct, that the demo link remains online, or that the system is safe for navigation.

## Main limitations

1. COCO car detection can miss submerged, small or occluded vehicles.
2. A missed vehicle never reaches the classifier; a wrong crop can propagate an error.
3. STURM's curated/upscaled crops differ from detector-generated scene crops.
4. Adjacent flood categories can be visually ambiguous.
5. Detector and softmax scores are not calibrated safety probabilities.
6. Full-scene ground truth is unavailable for a quantitative complete-system benchmark.
7. Checkpoint weights must be supplied separately or reproduced; a new run may give different scores.

The report and presentation should keep these limits alongside the successful integration results.
