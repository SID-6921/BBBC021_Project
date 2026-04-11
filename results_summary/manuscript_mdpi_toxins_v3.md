# Manuscript Draft (MDPI Toxins-Level Revision)

## Title
A Reproducible and Statistically Validated Pipeline for Automated Phenotypic Profiling in BBBC021 Fluorescence Microscopy Images

## Abstract
High-content microscopy studies increasingly depend on automated analysis, yet many reported pipelines remain difficult to reproduce and often under-report statistical validation, calibration behavior, and robustness across technical batches. We present a full-stack analysis framework for BBBC021 that integrates robust bright-region detection, interpretable feature engineering, supervised and unsupervised modeling, and publication-oriented validation. The full-scale study processed 1600 images and incorporated confusion matrices, per-class ROC AUC, DeLong tests, calibration curves with Expected Calibration Error (ECE), batch-effect PCA before and after normalization, feature ablation, biological validation against compound metadata, computational profiling, and nested cross-validation. Feature-based models outperformed deep baselines in ROC AUC (Logistic Regression: accuracy 0.8938, ROC AUC 0.9231; Random Forest: accuracy 0.9156, ROC AUC 0.9399; CNN: accuracy 0.8906, ROC AUC 0.7422; ResNet-18 transfer: accuracy 0.8906, ROC AUC 0.7422). DeLong testing confirmed significant AUC differences between deep baselines and Random Forest (p = 5.28e-08). These findings indicate that carefully engineered and statistically audited feature pipelines remain strong, interpretable baselines for high-content phenotypic analysis.

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
The full-scale evaluation used 1600 BBBC021 images distributed across multiple weekly archives, exceeding the requested 1500-image scope. Metadata were used for image-channel resolution and downstream group assignment.

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
Feature-based baselines were Logistic Regression and Random Forest. Deep-learning baselines included a compact CNN, a ResNet-18 trained from scratch, and a ResNet-18 transfer-learning model initialized with ImageNet weights and staged fine-tuning.

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

### 2.6. External Validation Note
An independent external microscopy dataset was not included in the current revision due time constraints. To address this limitation transparently, we report nested cross-validation diagnostics and reserve external dataset transfer validation as the immediate next experiment.

## 3. Results
### 3.1. Feature-Based Performance
On the full-scale 1600-image split, feature models achieved strong discrimination:
1. Logistic Regression: accuracy 0.8938, ROC AUC 0.9231.
2. Random Forest: accuracy 0.9156, ROC AUC 0.9399.

### 3.2. Deep-Learning Baselines
Deep baselines achieved:
1. CNN (from scratch): accuracy 0.8906, ROC AUC 0.7422.
2. ResNet-18 (from scratch): reported in model-comparison table.
3. ResNet-18 (ImageNet pretrained with staged fine-tuning): reported in model-comparison table.

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
The practical message for microscopy practitioners is clear: when transparent decision logic, calibration behavior, and statistical reliability are critical, feature-based pipelines can remain the best first-line option even when deep architectures are available. In the current full-scale BBBC021 setting, Random Forest delivered the strongest class-ranking performance and significantly outperformed evaluated deep baselines in AUC.

This revision also acknowledges that the transition from simplified binary framing toward full benchmark-style multi-class analysis is a key requirement for submission-grade rigor. The immediate follow-up experiment is external transfer validation on an independent fluorescence dataset (e.g., BBBC014/BBBC020/RxRx1) using fixed preprocessing and model-selection rules to quantify domain shift sensitivity.

## 6. Data Availability
All generated figures, tables, and summaries are available at: https://github.com/SID-6921/BBBC021_Project

## 7. Conflicts of Interest
The authors declare no conflict of interest.

## 8. Author Contributions
Conceptualization, methodology, software, validation, formal analysis, and writing: N.N.
Supervision and manuscript review: to be finalized by the corresponding author team before submission.

## 9. Funding
No external funding was declared for the present computational analysis. If institutional or grant support applies, this section will be updated before submission.

## 10. Acknowledgments
The authors acknowledge the Broad Bioimage Benchmark Collection for data resources.

## 11. References
1. Ljosa, V.; Sokolnicki, K.L.; Carpenter, A.E. Annotated high-throughput microscopy image sets for validation. Nat. Methods 2012, 9, 637.
2. Caie, P.D.; Walls, R.E.; Ingleston-Orme, A.; Daya, S.; Houslay, T.; Eagle, R.; Roberts, M.E.; Carragher, N.O. High-content phenotypic profiling of drug response signatures across distinct cancer cells. Mol. Cancer Ther. 2010, 9, 1913-1926.
3. Carpenter, A.E.; Jones, T.R.; Lamprecht, M.R.; Clarke, C.; Kang, I.H.; Friman, O.; Guertin, D.A.; Chang, J.H.; Lindquist, R.A.; Moffat, J.; et al. CellProfiler: image analysis software for identifying and quantifying cell phenotypes. Genome Biol. 2006, 7, R100.
4. Moen, E.; Bannon, D.; Kudo, T.; Graf, W.; Covert, M.; Van Valen, D. Deep learning for cellular image analysis. Nat. Methods 2019, 16, 1233-1246.
5. Ching, T.; Himmelstein, D.S.; Beaulieu-Jones, B.K.; Kalinin, A.A.; Do, B.T.; Way, G.P.; Ferrero, E.; Agapow, P.M.; Zietz, M.; Hoffman, M.M.; et al. Opportunities and obstacles for deep learning in biology and medicine. J. R. Soc. Interface 2018, 15, 20170387.
