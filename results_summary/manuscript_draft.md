# Draft Manuscript (Working Version)

## Title
Automated Spot Detection and Phenotypic Group Classification in BBBC021 Using Classical Computer Vision, Feature Modeling, and Deep Learning Comparison

## Introduction
Manual microscopy image analysis is time-consuming, subjective, and difficult to scale for high-throughput experiments. The BBBC021 benchmark provides a realistic setting for automated phenotypic profiling from fluorescence images. The goal of this work is to build a robust image analysis system that detects bright cellular regions, extracts quantitative features, and classifies two analysis groups (Group A vs Group B) without requiring domain-specific biology assumptions.

## Methods
### Dataset
We used BBBC021 images and metadata from the Broad Bioimage Benchmark Collection. For development and robustness validation, we processed multiple image batches (Week1_22123, Week1_22141, Week2_24121).

### Image Processing and Detection
Each image was fused across available channels and preprocessed (resize, min-max normalization, denoising/contrast cleaning). Spot detection used adaptive thresholding, morphological opening/closing, and contour filtering by both area and contour-level mean intensity.

### Feature Extraction
For each image, we computed:
- spot count
- mean intensity
- total intensity
- intensity variance
- area covered (px and ratio)
- density (spots per pixel and per 10,000 pixels)
- spot-size distribution descriptors (mean, std, median, q25, q75)
- small/medium/large spot fractions

### Supervised Modeling
Using the extracted feature set, we trained Logistic Regression and Random Forest models for Group A vs Group B classification. We report accuracy and ROC AUC.

### Unsupervised Analysis
PCA projection and K-means clustering were applied to evaluate natural separability of image groups in feature space.

### Deep Learning Comparison
A compact CNN classifier (PyTorch) was trained directly on preprocessed images (240 samples total; train/val/test = 134/34/72) to compare deep learning performance against feature-based models. Decision threshold was calibrated on the validation set.

### Robustness Testing
We evaluated three detection configurations (sensitive, default, conservative) across three batches to quantify parameter sensitivity and consistency.

## Results
### Core Feature-Based Models
From the refined modeling run (220 images):
- Logistic Regression: Accuracy = 0.8030, ROC AUC = 0.9172
- Random Forest: Accuracy = 0.8485, ROC AUC = 0.9148

### Deep Learning Comparison
From the CNN run (240 images):
- CNN: Accuracy = 0.8194, ROC AUC = 0.8769

The Random Forest achieved the highest accuracy among tested models, while Logistic Regression and Random Forest both achieved strong ROC AUC values above 0.91 in feature-based evaluation.

### Robustness Validation
Across three batches (450 processed image evaluations total across settings):
- Sensitive: LR acc = 0.7556, LR AUC = 0.8354, RF acc = 0.8889, RF AUC = 0.9547
- Default: LR acc = 0.6222, LR AUC = 0.7531, RF acc = 0.8667, RF AUC = 0.9352
- Conservative: LR acc = 0.6444, LR AUC = 0.6955, RF acc = 0.8667, RF AUC = 0.9270

These results indicate robust Random Forest performance across parameter regimes, with best performance under the sensitive configuration.

## Discussion
This work establishes a complete image analysis pipeline spanning robust detection, feature engineering, supervised and unsupervised learning, interpretability figures, and robustness validation. The feature-based approach provides strong predictive performance with interpretable importance rankings. The CNN comparison demonstrates that deep learning is feasible and competitive, though in this setup feature-based models still provide stronger or similar practical performance.

The current system is suitable for manuscript development and can be extended in future work by scaling to all BBBC021 image archives and adding a fully supervised detection network with annotation-quality labels for direct detector benchmarking.

## Conclusion
We developed a reproducible and robust computer vision pipeline for BBBC021 that produces publication-ready outputs, achieves strong group classification performance, and includes both classical and deep learning comparisons with robustness analysis across multiple batches.

## Artifacts
- Figures: `final_figures/`
- Final tables: `final_tables/`
- Summary metrics: `results_summary/`
- Model comparison table: `results_summary/model_comparison_final.csv`
