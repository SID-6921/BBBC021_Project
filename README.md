# BBBC021 Project

Computer vision baseline pipeline for BBBC021 using classical image processing.

## Project Structure

```
BBBC021_Project/
|
|-- data/
|   |-- raw/
|   |   |-- images/
|   |   `-- metadata/
|   |       `-- BBBC021_v1_image.csv
|   |
|   `-- processed/
|       |-- resized/
|       |-- normalized/
|       `-- cleaned/
|
|-- outputs/
|   |-- detections/
|   |-- overlays/
|   |-- metrics/
|   `-- plots/
|
|-- models/
|   |-- checkpoints/
|   `-- configs/
|
|-- notebooks/
|   |-- 01_data_exploration.ipynb
|   |-- 02_preprocessing.ipynb
|   |-- 03_detection.ipynb
|   `-- 04_analysis.ipynb
|
|-- src/
|   |-- data_loader.py
|   |-- preprocess.py
|   |-- detect.py
|   |-- features.py
|   `-- visualize.py
|
|-- requirements.txt
`-- README.md
```

## What This Baseline Does

1. Download BBBC021 images + metadata.
2. Build a simple Group A vs Group B split from metadata.
3. Run spot detection (OpenCV thresholding + contour filtering).
4. Compute per-image metrics:
   - spot count
   - average brightness
   - total intensity
   - area covered
5. Save overlays, metric tables, and comparison plots.

## Cloud-Only Workflow (GitHub + Colab/Kaggle)

### 1) Push this project to GitHub

Run these commands in a terminal from the project folder:

```powershell
git init
git add .
git commit -m "Initial BBBC021 baseline pipeline"
# Create an empty repo on GitHub named BBBC021_Project, then:
git remote add origin https://github.com/<YOUR_USERNAME>/BBBC021_Project.git
git branch -M main
git push -u origin main
```

### 2) Open in Colab (recommended)

- In Colab, use GitHub tab and open this repository notebook.
- Run notebooks in order:
  1. notebooks/01_data_exploration.ipynb
  2. notebooks/02_preprocessing.ipynb
  3. notebooks/03_detection.ipynb
  4. notebooks/04_analysis.ipynb

### 3) Or run in Kaggle

- Create a new notebook.
- Clone repo in a cell:

```bash
!git clone https://github.com/<YOUR_USERNAME>/BBBC021_Project.git
%cd BBBC021_Project
!pip install -r requirements.txt
```

- Open and run the notebook sequence above.

## First Deliverable Paths

After running `03_detection.ipynb` and `04_analysis.ipynb`, check:

- Overlays: `outputs/overlays/` (5-10 sample images)
- Spot table: `outputs/metrics/spot_count_sample.csv`
- Full metrics: `outputs/metrics/image_metrics.csv`
- Group plots: `outputs/plots/*.png`

For sharing and verification in GitHub, packaged artifacts are copied to:

- `outputs/deliverable/overlays/`
- `outputs/deliverable/metrics/`
- `outputs/deliverable/plots/`

## Notes

- BBBC021 images are distributed as multiple plate ZIPs; the exploration notebook starts with one ZIP for quick testing and can be extended to more ZIP URLs.
- Raw images are excluded from git via `.gitignore` to keep repo size manageable.
- You can adjust detection parameters in `src/detect.py`.

## Phase Pipeline (Refined Detection + Modeling)

Run the end-to-end refined pipeline:

```powershell
python src/phase_pipeline.py
```

This creates:

- `final_figures/`
  - Figure 1: pipeline workflow
  - Figure 2: detection overlays
  - Figure 3: feature comparison boxplots
  - Figure 4: ROC + feature importance
  - Figure 5: PCA + clustering
- `final_tables/`
  - `image_feature_table.csv`
- `results_summary/`
  - `classification_metrics.json`
  - `feature_importance.csv`
  - `pca_clusters.csv`
  - `summary.md`
