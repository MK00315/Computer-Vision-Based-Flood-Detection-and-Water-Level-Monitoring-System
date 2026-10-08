"""Standard-library checks; no model loading, training, downloads or GPU use."""
import ast
import csv
import json
from pathlib import Path
import random
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]


def notebook(name):
    return json.loads((ROOT / 'notebooks' / (name + '.ipynb')).read_text(encoding='utf-8'))


def code_text(name):
    return '\n'.join(''.join(cell['source']) for cell in notebook(name)['cells']
                     if cell['cell_type'] == 'code' and not ''.join(cell['source']).lstrip().startswith(('%pip', '!pip')))


def module_without_docstring(tree):
    body = list(tree.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        body = body[1:]
    return ast.Module(body=body, type_ignores=[])


class ProjectContracts(unittest.TestCase):
    def test_all_notebooks_are_clean(self):
        for name in ['classifier_training', 'project_evaluation', 'gradio_demo']:
            book = notebook(name)
            self.assertEqual(book['nbformat'], 4)
            self.assertNotIn('widgets', book['metadata'])
            for cell in book['cells']:
                if cell['cell_type'] == 'code':
                    self.assertIsNone(cell['execution_count'])
                    self.assertEqual(cell['outputs'], [])

    def test_python_exports_match_notebook_code(self):
        for name in ['classifier_training', 'project_evaluation', 'gradio_demo']:
            expected = module_without_docstring(ast.parse(code_text(name)))
            actual = module_without_docstring(ast.parse((ROOT / 'src' / (name + '_colab.py')).read_text(encoding='utf-8')))
            self.assertEqual(ast.dump(expected), ast.dump(actual), name)

    def test_car_detection_scope(self):
        for name in ['project_evaluation', 'gradio_demo']:
            tree = ast.parse(code_text(name))
            assignments = {}
            for node in tree.body:
                if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                    try:
                        assignments[node.targets[0].id] = ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        pass
            def car_only(value):
                if not isinstance(value, ast.List):
                    return False
                resolved = [assignments.get(item.id) if isinstance(item, ast.Name)
                            else ast.literal_eval(item) for item in value.elts]
                return resolved == [2]
            car_calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                         and any(keyword.arg == 'classes' and car_only(keyword.value)
                                 for keyword in node.keywords)]
            self.assertTrue(car_calls, name)

    def test_demo_defaults_and_no_training(self):
        tree = ast.parse(code_text('gradio_demo'))
        literal_assignments = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                try:
                    literal_assignments[node.targets[0].id] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass
        for key, value in [('DETECTOR_CONFIDENCE', .30), ('DETECTOR_IOU', .45),
                           ('DETECTOR_IMAGE_SIZE', 640), ('CLASSIFIER_CONFIDENCE', .50),
                           ('CROP_PADDING_FRACTION', .20), ('SAVE_SESSION_BACKUP', False)]:
            self.assertEqual(literal_assignments[key], value)
        text = code_text('gradio_demo')
        self.assertNotIn('.train(', text)
        self.assertIn('secrets.token_urlsafe(16)', text)
        self.assertIn("blocked_paths=['/content/drive', str(CHECKPOINT_PATH.resolve())]", text)
        self.assertIn("concurrency_id='models'", text)
        self.assertIn('theme=DEMO_THEME, css=DEMO_CSS', text)

    def test_padding_and_clipping(self):
        tree = ast.parse(code_text('gradio_demo'))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'expanded_crop_box')
        namespace = {'CROP_PADDING_FRACTION': .20}
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<crop-helper-only>', 'exec'), namespace)
        crop = namespace['expanded_crop_box']
        self.assertEqual(crop(10, 10, 40, 35, 80, 60), (4, 5, 46, 40))
        self.assertEqual(crop(0, 0, 20, 20, 80, 60), (0, 0, 24, 24))
        rng = random.Random(42)
        for _ in range(100):
            width, height = rng.randint(30, 800), rng.randint(30, 800)
            x1, y1 = rng.randrange(width - 1), rng.randrange(height - 1)
            x2, y2 = rng.randint(x1 + 1, width), rng.randint(y1 + 1, height)
            pad_x, pad_y = round(.2 * (x2 - x1)), round(.2 * (y2 - y1))
            self.assertEqual(crop(x1, y1, x2, y2, width, height),
                             (max(0, x1-pad_x), max(0, y1-pad_y), min(width, x2+pad_x), min(height, y2+pad_y)))

    def test_classifier_selection_rule(self):
        text = code_text('classifier_training')
        self.assertIn('val_macro_f1', text)
        self.assertIn('is_training = optimizer is not None', text)
        self.assertIn('torch.set_grad_enabled(is_training)', text)
        self.assertIn('@torch.inference_mode()', text)
        self.assertIn('val_f1 > best_val_f1 or (val_f1 == best_val_f1 and val_loss < best_val_loss)', text)
        self.assertIn('resnet50_sturm_best_macro_f1.pth', text)
        self.assertIn('EXPECTED_SPLIT_COUNTS', text)

    def test_gradio_records_and_scope(self):
        audit = json.loads((ROOT / 'results/gradio/audit.json').read_text())
        self.assertEqual((audit['unique_runs'], audit['unique_images'], audit['total_detection_rows'], audit['uncertain_rows']),
                         (10, 4, 13, 2))
        self.assertFalse(audit['scene_accuracy_measured'])
        self.assertTrue(audit['duplicate_backups_counted_once'])
        with (ROOT / 'results/gradio/run_summary.csv').open(encoding='utf-8', newline='') as handle:
            summary = list(csv.DictReader(handle))
        self.assertEqual(len(summary), 10)
        for run in summary:
            folder = ROOT / 'results/gradio' / f"run_{int(run['run']):02d}"
            metadata = json.loads((folder / 'settings_and_source.json').read_text(encoding='utf-8'))
            with (folder / 'predictions.csv').open(encoding='utf-8', newline='') as handle:
                predictions = list(csv.DictReader(handle))
            self.assertEqual(len(predictions), int(run['detections']))
            self.assertEqual(len(predictions), metadata['detected_cars'])
            self.assertEqual(metadata['class_names'], [f'Level{i}' for i in range(5)])
            self.assertEqual(metadata['settings']['crop_padding_per_side'], .2)
            self.assertEqual(metadata['settings']['classifier_image_size'], 224)
            self.assertEqual(metadata['settings']['detector_iou'], .45)
            self.assertEqual(sum(row['uncertain'].lower() == 'true' for row in predictions), int(run['uncertain_predictions']))

    def test_no_weight_or_archive_files(self):
        prohibited = {'.pth', '.pt', '.onnx', '.zip', '.pem', '.key'}
        for path in ROOT.rglob('*'):
            if path.is_file() and '.git' not in path.parts:
                self.assertNotIn(path.suffix.lower(), prohibited, str(path.relative_to(ROOT)))

    def test_classifier_metrics_match_class_report(self):
        metrics = json.loads((ROOT / 'results/classifier/metrics.json').read_text())
        report = json.loads((ROOT / 'results/classifier/classification_report.json').read_text())
        self.assertEqual(metrics['test_images'], 340)
        self.assertAlmostEqual(metrics['accuracy'], 254 / 340)
        self.assertAlmostEqual(metrics['accuracy'], report['accuracy'])
        self.assertEqual(sum(report[f'Level{i}']['support'] for i in range(5)), 340)
        for key, field in [('macro_precision', 'precision'), ('macro_recall', 'recall'), ('macro_f1', 'f1-score')]:
            self.assertAlmostEqual(metrics[key], report['macro avg'][field])
        self.assertEqual(metrics['checkpoint_best_epoch'], 5)
        with (ROOT / 'results/classifier/training_history.csv').open(encoding='utf-8', newline='') as handle:
            history = list(csv.DictReader(handle))
        selected = max(history, key=lambda row: (float(row['val_macro_f1']), -float(row['val_loss'])))
        self.assertEqual(int(selected['epoch']), metrics['checkpoint_best_epoch'])
        self.assertAlmostEqual(float(selected['val_macro_f1']), metrics['checkpoint_best_val_macro_f1'])

    def test_gradio_credit_coverage(self):
        with (ROOT / 'docs/gradio_image_credits.csv').open(encoding='utf-8', newline='') as handle:
            credits = {row['source_image']: row for row in csv.DictReader(handle)}
        with (ROOT / 'results/gradio/run_summary.csv').open(encoding='utf-8', newline='') as handle:
            runs = list(csv.DictReader(handle))
        self.assertEqual(set(credits), {row['source_image'] for row in runs})
        for credit in credits.values():
            for field in ['author', 'license', 'source_url', 'license_url', 'modifications']:
                self.assertTrue(credit[field], field)
            self.assertTrue(credit['source_url'].startswith('https://commons.wikimedia.org/wiki/'))

    def test_no_published_secret_literals(self):
        token_pattern = re.compile(r'(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})')
        for parent in ['notebooks', 'src', 'results', 'docs']:
            for path in (ROOT / parent).rglob('*'):
                if path.suffix.lower() in {'.py', '.ipynb', '.json', '.csv', '.md'}:
                    text = path.read_text(encoding='utf-8')
                    self.assertIsNone(token_pattern.search(text), str(path))
                    self.assertIsNone(re.search(r'(?m)^Password:\s*\S{10,}', text), str(path))


if __name__ == '__main__':
    unittest.main()
