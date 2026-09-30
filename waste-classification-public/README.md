# Waste Classification Research Experiments

Four models: Custom CNN, MobileNetV2, ResNet50, EfficientNetB0.
This package produces measured outputs for your research paper. It does not
contain fabricated results, pretrained waste classifiers or the dataset.
Internet is needed for the dataset and first download of ImageNet weights.

## 1. Install and open the folder

Extract this ZIP first. Open `waste_research` in VS Code and open its terminal.
Use **64-bit Python 3.11** (recommended) or 3.12 for this pinned TensorFlow version.
Do not use your Python 3.14 interpreter for this package. This project pins
TensorFlow 2.16.2 for reproducibility rather than depending on a changing latest release.

Windows commands:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Using the explicit interpreter avoids PowerShell activation-policy problems.
In all commands below, replace `python` with `.venv\Scripts\python.exe` on Windows.
Choose this interpreter in VS Code too.

Linux / macOS:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Native Windows uses CPU with this TensorFlow version. GPU training can be run
in Colab or supported Linux/WSL2. Do not expect a normal Windows install to use
your NVIDIA GPU automatically. Colab instructions are in `COLAB.md`.

## 2. Download and prepare the data

```bash
python research.py download
python research.py prepare
```

The downloader uses the original TrashNet repository. If downloading fails,
download `data/dataset-resized.zip` from https://github.com/garythung/trashnet,
extract it into `data`, and run prepare. The folder must contain:

`data/dataset-resized/cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`.

The source has 2,527 images; exclusions may reduce that number. Inspect
`outputs/dataset_counts.csv`, `excluded_images.csv` and
`near_duplicates_to_review.csv` **before training**. The perceptual-hash
threshold is a conservative grouping heuristic, not a guarantee that all
same-object images have been found. False positives are possible; inspect the
listed pairs. Exact decoded duplicates are excluded; if labels conflict, all
copies of the conflicting decoded image are excluded and audited. Similar images are grouped
across labels and cannot cross partitions. 20 stratified group folds approximate
70/15/15, with realized counts saved. If labels or grouping need correction,
change data/preparation code and prepare in a fresh output folder.

## 3. Run a quick check first

```bash
python research.py prepare --out outputs_smoke
python research.py train --out outputs_smoke --models custom_cnn --seeds 42 --epochs 1 --head-epochs 1 --fine-epochs 0 --smoke
python research.py report --out outputs_smoke --bootstrap 20
```

Smoke uses only a few real images and is **not a paper experiment**. It tests
installation, training, saving and reporting. Do not copy smoke metrics into
your paper. Do not reuse `outputs_smoke` for the full run.

## 4. Full research experiment

After preparing and reviewing the audit files, one command runs training,
CPU benchmarking and reporting:

```bash
python run_experiments.py
```

Or run the three stages separately:

```bash
python research.py train
python research.py benchmark
python research.py report
```

This runs 4 models × 3 seeds (42, 123, 2026). Transfer heads train for up to
10 epochs, followed by up to 20 fine-tuning epochs; the custom CNN trains for
up to 30. Early stopping uses validation macro F1. Fine-tuning unfreezes the
last 30 backbone layers except batch normalization; record that detail in
your paper. Preprocessing is model-specific and included in the saved model.
Brightness, blur and noise tests are run after clean test evaluation.

Training can take hours or longer on CPU. Runtime depends on hardware; no
fixed duration is promised. Keep the device type consistent across models.
You can train one model at a time with the SAME settings:

```bash
python research.py train --models custom_cnn
python research.py train --models mobilenetv2
python research.py train --models resnet50
python research.py train --models efficientnetb0
```

Rerunning skips completed model/seed runs. An interrupted run restarts from
the beginning; this is not mid-epoch resume. Completed results are retained.
Do not change experiment settings in the same output folder. The script checks
configuration and manifest hashes. If memory is insufficient, use batch size 8
for **every model**, in a new prepared output folder:

```bash
python research.py prepare --out outputs_batch8
python research.py train --out outputs_batch8 --batch-size 8
python research.py benchmark --out outputs_batch8
python research.py report --out outputs_batch8
```

## 5. What goes directly into your paper

Open `outputs/paper_ready/results.html` in a browser to view tables and figures.
Open `outputs/paper_ready/paper_results.docx` for editable Word tables,
results text and figures. Copy reviewed sections into your research paper;
this is a results supplement, not a completed manuscript or an automatic
replacement of your proposal. Seed 42 figures are used consistently.
Open `results_to_paste.md` in VS Code for a measured results paragraph and
limitations. CSV tables can be opened in Excel and inserted into Word.

| Output | Where to use it |
|---|---|
| comparison_mean_std.csv | Main model comparison table (mean ± SD over seeds) |
| paper_results.docx | Editable results supplement for Word |
| pareto_frontier.csv | Non-dominated models using macro F1, CPU latency and size |
| comparison_macro_f1.png | Model-performance figure |
| performance_vs_latency.png | Performance/CPU-speed comparison, after benchmark |
| class_metrics_all.csv | Per-class precision, recall, F1 and support |
| bootstrap_differences.csv | Exploratory paired confidence intervals |
| robustness_summary.csv | Brightness, blur and noise analysis |
| runs/.../accuracy_curve.png and loss_curve.png | Training plots |
| runs/.../confusion_matrix.png | Error-analysis figure |
| dataset_counts.csv | Actual class counts by split |
| experiment_config.json and *_environment.json | Experimental setup details |

Accuracy/F1/precision/recall values are fractions; multiply by 100 if presenting
percentages. SD is sample SD; one run has undefined SD. The report fills missing
SD with zero only for plotting, not in the CSV. Use a predeclared seed (such as
42) for illustrative confusion matrices, not the seed with highest test accuracy.
CPU timing includes preprocessing in the model; separate end-to-end timings add
file decoding/resizing. They use warm-up, batch size 1, float32, forced output
materialization, 500 calls and one CPU thread by default. File I/O may be cached.
Model size is a `.keras` archive without optimizer state. Training seconds include
validation F1 prediction and checkpoint I/O, and exclude post-training tests.

**Review the outputs; do not submit them blindly.** Update the proposal with your
actual settings and measured results. Report missing runs, negative findings and
limitations. This code does not automatically establish novelty or prove that a
smart bin works. Transfer models differ in architecture AND initialization from
the scratch CNN; their comparison cannot isolate the effect of pretraining.
Paired bootstrap intervals condition on one test split and average seed metrics;
they are exploratory, not independent evidence across datasets. Six pairwise
intervals are not corrected for multiple comparisons.

## 6. Prediction on your own image

```bash
python predict.py --model outputs/runs/mobilenetv2_seed42/model.keras --image my_waste.jpg
```

Softmax scores are not calibrated confidence, and the classifier has no
unknown-object rejection. It will always choose one of six categories.

## References and sources

- TrashNet: https://github.com/garythung/trashnet (Thung and Yang).
- Keras applications: https://keras.io/api/applications/
- TensorFlow installation: https://www.tensorflow.org/install/pip
- MobileNetV2: https://arxiv.org/abs/1801.04381
- ResNet: https://arxiv.org/abs/1512.03385
- EfficientNet: https://proceedings.mlr.press/v97/tan19a.html

Keep dataset attribution and follow its applicable terms. Raw data and ImageNet
weights are downloaded separately. The code and statistical protocol must be
described accurately in the final manuscript.
