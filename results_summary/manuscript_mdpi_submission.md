# Manuscript Draft (MDPI-Oriented Submission Version)

## Title
A Robust and Reproducible Pipeline for Automated Spot Detection, Quantification, and Group Classification in BBBC021 Fluorescence Microscopy Images

## Running Title
Automated BBBC021 Spot Analysis with Classical and Deep Models

## Abstract
High-content microscopy analysis requires scalable and reproducible computational methods that can replace or complement manual workflows. We present a complete image-analysis system for BBBC021 integrating robust spot detection, quantitative feature extraction, supervised and unsupervised learning, robustness validation across batches, and deep-learning comparison. The advanced run processed 1600 images and added confusion matrices, per-class ROC AUC, DeLong significance testing, calibration analysis, feature ablation, batch-effect PCA before/after normalization, biological feature validation, computational cost profiling, and nested cross-validation. Feature-based models achieved strong discrimination (Logistic Regression: accuracy 0.8938, ROC AUC 0.9231; Random Forest: accuracy 0.9156, ROC AUC 0.9399). CNN and ResNet-18 baselines reached accuracy 0.8906 and ROC AUC 0.7422. DeLong testing confirmed a significant AUC difference between feature and deep baselines (p = 5.28e-08). The workflow produces publication-ready figures and tables and is fully version-controlled for reproducibility.

## Keywords
BBBC021; fluorescence microscopy; image analysis; adaptive thresholding; feature engineering; machine learning; random forest; convolutional neural network; transfer learning; robustness analysis; reproducibility

## 1. Introduction
Automated image analysis has become essential in modern high-content screening, where manual inspection is limited by subjectivity and throughput constraints. Public benchmark datasets provide a foundation for evaluating algorithmic reliability and cross-study comparability. BBBC021 is a widely used benchmark containing multi-channel fluorescence images acquired under diverse treatment conditions.

Many pipelines focus exclusively on either classical image processing or end-to-end deep learning. Classical approaches offer transparency and often perform strongly when engineered carefully, while deep-learning approaches may capture richer nonlinear morphology but require additional model tuning and computational resources. In practice, a complete and reproducible workflow should include robust detection, interpretable quantification, predictive modeling, stability testing, and clear figure generation for publication.

The objective of this work was to build such a workflow for BBBC021. Specifically, we aimed to improve spot detection robustness, expand image-level feature extraction, evaluate supervised classification, assess unsupervised structure, validate robustness across batches and thresholds, compare feature models against deep baselines, and provide statistical and calibration analyses suitable for publication.

## 2. Materials and Methods
### 2.1. Dataset
BBBC021 image and metadata files were used from the Broad Bioimage Benchmark Collection. The advanced evaluation processed 1600 images from multiple weekly archives, exceeding the 1500-image target.

### 2.2. Preprocessing
For each metadata row with available channels, image channels were loaded and fused to grayscale for baseline analysis. Processing steps were: (i) spatial resizing, (ii) min-max normalization, and (iii) denoising/contrast enhancement.

### 2.3. Spot Detection
Detection used adaptive Gaussian thresholding, followed by morphological opening/closing. Candidate contours were filtered by area and contour-level mean intensity, reducing false positives under variable local brightness.

### 2.4. Feature Extraction
Per-image features included: spot count, mean intensity, total intensity, intensity variance, area covered (pixels and ratio), spot density, spot-size distribution statistics (mean/std/median/q25/q75), and size-fraction descriptors (small/medium/large).

### 2.5. Supervised Feature-Based Modeling
Logistic Regression and Random Forest were trained on extracted features. Model performance was measured by accuracy and ROC AUC. Per-class ROC AUC was reported as one-vs-rest for Group A and Group B.

### 2.6. Deep-Learning Baselines
Two deep baselines were used:
1. compact CNN,
2. ResNet-18 transfer baseline.

Both were trained on the same split protocol and compared against feature models.

### 2.7. Statistical and Reliability Analysis
The following analyses were added:
1. DeLong test for AUC differences (CNN vs Random Forest; ResNet-18 vs Random Forest),
2. confusion matrices,
3. calibration curves,
4. expected calibration error (ECE).

### 2.8. Batch Effect and Ablation Analysis
PCA was used to visualize batch effects before and after batch-wise normalization. Feature ablation progressively removed top-ranked Random Forest features to quantify performance degradation.

### 2.9. Biological Validation and Overfitting Checks
Top-ranked features were evaluated against compound metadata using nonparametric group tests and concentration correlations. Nested cross-validation on the training partition was performed to assess overfitting risk.

### 2.10. Computational Cost
Training time and peak memory estimates were recorded for Logistic Regression, Random Forest, CNN, and ResNet-18 baselines.

## 3. Results
### 3.1. Full-Dataset Feature-Based Classification (n = 1600)
- Logistic Regression: accuracy 0.8938, ROC AUC 0.9231.
- Random Forest: accuracy 0.9156, ROC AUC 0.9399.

### 3.2. Deep-Learning Comparison
- CNN: accuracy 0.8906, ROC AUC 0.7422.
- ResNet-18 transfer: accuracy 0.8906, ROC AUC 0.7422.

### 3.3. Statistical AUC Validation
DeLong tests against Random Forest showed significant AUC differences:
- CNN vs Random Forest: p = 5.28e-08.
- ResNet-18 vs Random Forest: p = 5.28e-08.

### 3.4. Calibration and Confusion Analysis
Calibration curves and ECE were computed for all models, and confusion matrices were generated for Random Forest, CNN, and ResNet-18.

### 3.5. Batch Effects, Ablation, and Biological Validation
PCA showed clear batch-structure changes after normalization. Feature ablation demonstrated expected AUC drops when top-ranked features were removed. Biological validation outputs quantified feature associations with compound identity and concentration metadata.

### 3.6. Nested Cross-Validation and Computational Cost
Nested CV summaries were generated for feature models to assess generalization stability. Training-time and peak-memory estimates were compiled into a computational cost comparison table.

## 4. Discussion
The full-scale 1600-image analysis confirms that an interpretable feature-engineered approach can outperform the tested deep baselines in AUC for this task configuration. Importantly, this result is supported by statistical testing rather than point estimates alone. The additional calibration, ablation, and nested CV diagnostics provide stronger methodological reliability for publication.

The deep baselines remain important references and establish a clear path for future improvements (stronger augmentation, longer schedules, and larger architecture tuning). Nonetheless, the present evidence supports Random Forest on engineered morphology/intensity descriptors as a robust and computationally efficient baseline.

## 5. Conclusions
We developed and validated a complete BBBC021 analysis system that scales to 1500+ images and includes robust classical detection, rich feature modeling, deep-learning baselines, statistical significance testing, calibration diagnostics, batch-effect analysis, feature ablation, biological validation, and overfitting checks. The workflow is reproducible and publication-ready.

## 6. Data Availability
All generated analysis artifacts are available in the repository under `final_figures/`, `final_tables/`, and `results_summary/`.

## 7. Conflicts of Interest
The authors declare no conflict of interest.

## 8. Author Contributions (Template)
Conceptualization, X.X.; methodology, X.X.; software, X.X.; validation, X.X.; formal analysis, X.X.; investigation, X.X.; writing-original draft preparation, X.X.; writing-review and editing, X.X.; supervision, X.X.

## 9. Funding (Template)
This research received no external funding. Replace with grant details if applicable.

## 10. Acknowledgments (Template)
The authors acknowledge the Broad Bioimage Benchmark Collection for data availability.
