# Manuscript Draft (MDPI-Oriented Submission Version)

## Title
A Robust and Reproducible Pipeline for Automated Spot Detection, Quantification, and Group Classification in BBBC021 Fluorescence Microscopy Images

## Running Title
Automated BBBC021 Spot Analysis with Classical and Deep Models

## Abstract
High-content microscopy analysis requires scalable and reproducible computational methods that can replace or complement manual workflows. We present a complete image-analysis system for the BBBC021 benchmark that integrates robust spot detection, quantitative feature extraction, supervised and unsupervised learning, robustness validation across batches, and deep-learning comparison. Detection uses adaptive thresholding, morphological refinement, and contour filtering by size and intensity, followed by extraction of interpretable image-level descriptors including spot count, intensity statistics, area coverage, density, and spot-size distribution metrics. Feature-based classification for Group A versus Group B achieved strong predictive performance (Logistic Regression: accuracy 0.8030, ROC AUC 0.9172; Random Forest: accuracy 0.8485, ROC AUC 0.9148). Robustness testing across three BBBC021 batches and three detection parameter regimes showed stable Random Forest performance, with best performance under a sensitive configuration (accuracy 0.8889, ROC AUC 0.9547). A compact convolutional neural network classifier was trained for comparison and reached accuracy 0.8194 and ROC AUC 0.8769. The workflow produces publication-ready figures and tables and is fully version-controlled for reproducibility. These results support the use of interpretable feature-based pipelines as strong baselines while enabling practical deep-learning extensions for future work.

## Keywords
BBBC021; fluorescence microscopy; image analysis; adaptive thresholding; feature engineering; machine learning; random forest; convolutional neural network; robustness analysis; reproducibility

## 1. Introduction
Automated image analysis has become essential in modern high-content screening, where manual inspection is limited by subjectivity and throughput constraints. Public benchmark datasets provide a foundation for evaluating algorithmic reliability and cross-study comparability. BBBC021 is a widely used benchmark containing multi-channel fluorescence images acquired under diverse treatment conditions.

Many pipelines focus exclusively on either classical image processing or end-to-end deep learning. Classical approaches offer transparency and often perform strongly when engineered carefully, while deep-learning approaches may capture richer nonlinear morphology but require additional model tuning and computational resources. In practice, a complete and reproducible workflow should include robust detection, interpretable quantification, predictive modeling, stability testing, and clear figure generation for publication.

The objective of this work was to build such a workflow for BBBC021. Specifically, we aimed to (i) improve spot detection robustness, (ii) expand image-level feature extraction, (iii) evaluate supervised classification of Group A versus Group B, (iv) assess unsupervised structure, (v) validate robustness across batches and thresholds, and (vi) compare feature-based models against a compact CNN baseline.

## 2. Materials and Methods
### 2.1. Dataset
BBBC021 image and metadata files were used from the Broad Bioimage Benchmark Collection. For core modeling and robustness analyses, images from multiple weekly batches were processed. Metadata were used to align channel files and construct analysis groups.

### 2.2. Preprocessing
For each metadata row with available channels, image channels were loaded and fused to grayscale for baseline analysis. The following preprocessing steps were applied:
1. Spatial resizing to a fixed input resolution.
2. Intensity normalization to a common scale.
3. Denoising and local contrast enhancement.

### 2.3. Spot Detection
The detection module uses adaptive Gaussian thresholding, followed by morphological opening and closing to suppress isolated noise and stabilize region boundaries. Contours are extracted and filtered by:
1. minimum and maximum area constraints,
2. contour-level mean intensity threshold.

This strategy reduces false positives while preserving true bright regions under varying local contrast conditions.

### 2.4. Feature Extraction
For each image, the following feature set is computed:
1. spot count,
2. mean intensity,
3. total intensity,
4. intensity variance,
5. covered area (pixels and ratio),
6. spot density (per pixel and per 10,000 pixels),
7. spot-size distribution statistics (mean, standard deviation, median, 25th and 75th percentiles),
8. small, medium, and large spot fractions.

These descriptors form an interpretable phenotype profile for downstream learning.

### 2.5. Supervised Classification
Feature-based models include Logistic Regression and Random Forest. Data are split into train and test partitions with stratification. Model quality is evaluated using accuracy and ROC AUC, and feature importance is exported for interpretation.

### 2.6. Unsupervised Analysis
Principal component analysis (PCA) and K-means clustering are applied to evaluate latent structure and natural grouping in feature space.

### 2.7. Deep-Learning Comparison
A compact CNN classifier was implemented in PyTorch and trained on preprocessed grayscale inputs. Dataset size for the CNN run was 240 samples with split counts: train 134, validation 34, and test 72. A decision threshold was selected from validation performance before final test reporting.

### 2.8. Robustness Validation
Robustness analysis was performed across three BBBC021 batches (Week1_22123, Week1_22141, Week2_24121) and three detection configurations:
1. sensitive,
2. default,
3. conservative.

For each configuration, detection outputs and downstream classification metrics were summarized.

### 2.9. Reproducibility and Outputs
All outputs are versioned in the repository:
1. figures in `final_figures/`,
2. tables in `final_tables/`,
3. summaries in `results_summary/`.

## 3. Results
### 3.1. Feature-Based Classification Performance
From the refined feature-modeling run (220 images):
1. Logistic Regression: accuracy 0.8030, ROC AUC 0.9172.
2. Random Forest: accuracy 0.8485, ROC AUC 0.9148.

Both models achieved strong discrimination; Random Forest provided the highest accuracy.

### 3.2. Unsupervised Structure
PCA and clustering outputs showed structured separation trends between analysis groups, supporting non-random organization in the engineered feature space.

### 3.3. Robustness Across Batches and Detection Settings
Across three batches and multiple settings (450 processed evaluations total):
1. Sensitive: LR accuracy 0.7556, LR AUC 0.8354, RF accuracy 0.8889, RF AUC 0.9547.
2. Default: LR accuracy 0.6222, LR AUC 0.7531, RF accuracy 0.8667, RF AUC 0.9352.
3. Conservative: LR accuracy 0.6444, LR AUC 0.6955, RF accuracy 0.8667, RF AUC 0.9270.

Random Forest remained consistently strong across configurations, indicating robust predictive behavior.

### 3.4. Deep-Learning Comparison
CNN comparison results were:
1. accuracy 0.8194,
2. ROC AUC 0.8769,
3. validation-derived decision threshold 0.25.

The CNN produced competitive results, while the feature-based Random Forest remained slightly stronger in this configuration.

## 4. Discussion
This study demonstrates that a carefully engineered classical vision pipeline can deliver high performance and strong interpretability on BBBC021 while remaining robust across batches and threshold regimes. The feature set captures both intensity and morphology-related characteristics and supports accurate group discrimination.

The deep-learning comparison confirms that compact CNN models are viable within this workflow and can serve as a baseline for future expansion. However, in the current setup, the feature-based Random Forest showed superior or comparable practical performance with more direct interpretability.

The robustness analysis strengthens the translational value of the pipeline by showing that results are not limited to a single threshold choice or single batch source.

## 5. Conclusions
A complete, reproducible, and publication-oriented BBBC021 analysis system was developed and validated. The pipeline includes robust detection, comprehensive feature extraction, supervised and unsupervised modeling, robustness benchmarking, and deep-learning comparison, with all outputs prepared for manuscript reporting.

## 6. Study Limitations and Future Work
1. Current analysis does not yet include all 55 BBBC021 image archives.
2. The deep-learning comparison uses a compact CNN baseline rather than a full detector architecture such as YOLO.
3. Future work should include all-plate scaling, detector-level supervision, and formal statistical significance testing between model families.

## 7. Data Availability
All generated analysis artifacts are available in the project repository under `final_figures/`, `final_tables/`, and `results_summary/`.

## 8. Conflicts of Interest
The authors declare no conflict of interest.

## 9. Author Contributions (Template)
Conceptualization, X.X.; methodology, X.X.; software, X.X.; validation, X.X.; formal analysis, X.X.; investigation, X.X.; writing-original draft preparation, X.X.; writing-review and editing, X.X.; supervision, X.X. All authors have read and agreed to the published version of the manuscript.

## 10. Funding (Template)
This research received no external funding. Replace with grant details if applicable.

## 11. Acknowledgments (Template)
The authors acknowledge the Broad Bioimage Benchmark Collection for providing BBBC021 data resources.
