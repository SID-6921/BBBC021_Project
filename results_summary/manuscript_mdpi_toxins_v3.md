# Manuscript Draft (MDPI Toxins-Level Revision)

## Title
A Reproducible and Statistically Validated Pipeline for Automated Phenotypic Profiling in BBBC021 Fluorescence Microscopy Images

## Abstract
High-content microscopy studies increasingly depend on automated analysis, yet many reported pipelines remain difficult to reproduce and often under-report statistical validation, calibration behavior, and robustness across technical batches. We present a full-stack analysis framework for BBBC021 that integrates robust bright-region detection, interpretable feature engineering, supervised and unsupervised modeling, and publication-oriented validation. The advanced study processed 1600 images and incorporated confusion matrices, per-class ROC AUC, DeLong tests, calibration curves with Expected Calibration Error (ECE), batch-effect PCA before and after normalization, feature ablation, biological validation against compound metadata, computational profiling, and nested cross-validation. Feature-based models outperformed deep baselines in ROC AUC (Logistic Regression: accuracy 0.8938, ROC AUC 0.9231; Random Forest: accuracy 0.9156, ROC AUC 0.9399; CNN: accuracy 0.8906, ROC AUC 0.7422; ResNet-18 transfer: accuracy 0.8906, ROC AUC 0.7422). DeLong testing confirmed significant AUC differences between deep baselines and Random Forest (p = 5.28e-08). These findings indicate that carefully engineered and statistically audited feature pipelines remain strong, interpretable baselines for high-content phenotypic analysis.

## Keywords
BBBC021; fluorescence microscopy; phenotypic profiling; adaptive thresholding; feature engineering; random forest; deep learning; transfer learning; calibration; DeLong test; reproducibility

## 1. Introduction
Automated microscopy analysis is no longer optional in high-content workflows; it is a methodological necessity. As image volumes increase, manual curation becomes a source of delay, inconsistency, and irreproducible interpretation. In this context, robust computational pipelines must satisfy dual requirements: technical performance and scientific defensibility.

BBBC021 offers a useful benchmark for this challenge because it combines multi-channel fluorescence imaging with rich compound-treatment metadata. However, many analysis reports in this area focus narrowly on single performance metrics and omit key reliability components such as calibration behavior, batch effects, statistical significance testing between models, and explicit overfitting checks.

To address this gap, we developed an end-to-end BBBC021 framework designed not only for prediction, but for publication-level evidence quality. Our objectives were fivefold:
1. build a robust and interpretable spot-detection module;
2. engineer a quantitative feature set suitable for biological interpretation;
3. compare feature-based and deep-learning models under matched evaluation settings;
4. validate results through calibration, ablation, statistical testing, and nested CV;
5. document outputs in manuscript-ready format with complete figure and table traceability.

## 2. Materials and Methods
### 2.1. Dataset and Study Scope
The advanced evaluation used 1600 BBBC021 images distributed across multiple weekly archives, exceeding the requested 1500-image scope. Metadata were used for image-channel resolution and downstream group assignment.

### 2.2. Preprocessing and Detection
For each sample, available channels were fused into a grayscale representation. Images were resized, intensity-normalized, and denoised/contrast-enhanced prior to detection. Bright-region detection employed adaptive Gaussian thresholding, followed by morphological opening and closing. Contours were filtered by area and contour-level mean intensity to reduce spurious detections.

### 2.3. Feature Engineering
Per-image descriptors included:
1. abundance: spot count and density;
2. intensity: mean, total, and variance;
3. geometry: area coverage and spot-size distribution statistics (mean, standard deviation, median, q25, q75);
4. compositional morphology: small/medium/large spot fractions.

This feature design was intended to preserve interpretability while capturing heterogeneous phenotypic signals.

### 2.4. Supervised Modeling
Feature-based baselines were Logistic Regression and Random Forest. Deep-learning baselines included a compact CNN and a ResNet-18 transfer-learning classifier. Data were partitioned into train, validation, and test splits for consistent model comparison.

### 2.5. Reliability and Statistical Validation
The following analyses were performed:
1. confusion matrices for key models;
2. per-class ROC AUC reporting;
3. DeLong tests for AUC difference (CNN vs RF, ResNet-18 vs RF);
4. calibration curves and ECE;
5. nested CV on training data;
6. feature ablation;
7. batch-effect PCA before and after normalization;
8. biological validation of top features against compound metadata;
9. computational cost comparison (training time and peak memory).

## 3. Results
### 3.1. Feature-Based Performance
On the advanced 1600-image split, feature models achieved strong discrimination:
1. Logistic Regression: accuracy 0.8938, ROC AUC 0.9231.
2. Random Forest: accuracy 0.9156, ROC AUC 0.9399.

These values indicate robust separability under an interpretable feature representation.

### 3.2. Deep-Learning Baselines
Deep baselines achieved:
1. CNN: accuracy 0.8906, ROC AUC 0.7422.
2. ResNet-18 transfer: accuracy 0.8906, ROC AUC 0.7422.

Although test accuracy remained competitive, ROC AUC was substantially lower than Random Forest, suggesting weaker ranking quality across thresholds.

### 3.3. Statistical Comparison
DeLong tests against Random Forest showed statistically significant differences:
1. CNN vs RF: p = 5.28e-08.
2. ResNet-18 vs RF: p = 5.28e-08.

Thus, the AUC gap is not attributable to random split variation alone.

### 3.4. Calibration and Overfitting Diagnostics
ECE values were low across models, with calibration curves included for visual reliability assessment. Nested CV showed stable AUC for feature-based models (approximately 0.91 mean), supporting generalization robustness.

### 3.5. Ablation, Batch Effects, and Biological Validation
Feature ablation produced expected AUC degradation as top-ranked features were removed, supporting importance validity. PCA visualizations showed substantial batch structure before normalization and reduced batch-driven separation after normalization. Biological validation linked top descriptors to compound metadata and concentration trends through nonparametric and correlation analyses.

### 3.6. Computational Considerations
A cost table summarizing training time and peak memory showed the expected tradeoff: Logistic Regression and Random Forest remained computationally efficient, while deep models required higher training overhead.

## 4. Discussion
This study reinforces an often-overlooked point in computational imaging: high interpretability and high performance are not mutually exclusive. In this benchmark setting, feature-based Random Forest outperformed both deep baselines in ROC AUC and retained direct interpretability through explicit descriptors and ablation evidence.

The deep-learning results are still valuable: they provide a practical baseline and a trajectory for future optimization. However, the current evidence indicates that deep models did not surpass the strongest engineered-feature configuration under matched evaluation constraints.

The broader contribution is methodological rigor. By combining DeLong testing, calibration analysis, nested CV, batch-effect auditing, and biological validation, the study moves beyond point-metric reporting toward reproducible scientific inference. This is particularly important for toxicity-related phenotypic studies, where model reliability and interpretability are central to translational credibility.

## 5. Conclusions
We developed a reproducible BBBC021 analysis framework that integrates robust detection, interpretable feature modeling, deep baseline comparison, and publication-level statistical validation. On a 1600-image evaluation, feature-based Random Forest delivered the strongest discriminatory performance and significantly outperformed deep baselines in AUC under DeLong testing. The pipeline, outputs, and manuscript artifacts are organized for direct journal submission workflows.

## 6. Limitations and Future Directions
1. The present deep-learning comparison used compact training schedules and can be further improved with extended optimization and augmentation.
2. Future work should scale to additional BBBC021 archives and evaluate cross-domain transfer stability.
3. Explicit uncertainty quantification and external dataset validation would further strengthen deployment-readiness.

## 7. Data Availability
All generated figures, tables, and summaries are available in the repository under final_figures, final_tables, and results_summary.

## 8. Conflicts of Interest
The authors declare no conflict of interest.

## 9. Author Contributions
Conceptualization, methodology, software, validation, formal analysis, and writing: to be finalized with coauthor review.

## 10. Funding
To be completed as applicable.

## 11. Acknowledgments
The authors acknowledge the Broad Bioimage Benchmark Collection for data resources.
