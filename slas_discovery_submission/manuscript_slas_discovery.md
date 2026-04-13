# A Reproducible and Statistically Validated Pipeline for Automated Phenotypic Profiling in BBBC021 Fluorescence Microscopy Images

## Abstract

**Background.** High-content microscopy studies depend on automated analysis pipelines that must satisfy dual requirements: technical performance and scientific defensibility. Many published pipelines report single performance metrics and omit calibration behavior, batch-effect auditing, and formal statistical comparison between models.

**Objectives.** To develop and validate a full-stack analysis framework for the BBBC021 benchmark that integrates robust detection, interpretable feature engineering, five supervised classifiers, and publication-level statistical validation with complete traceability of results.

**Methods.** A total of 824 images spanning 10 mechanism-of-action (MOA) classes were processed through adaptive Gaussian thresholding, morphological feature extraction, and five classifiers: Logistic Regression (LR), Random Forest (RF), a compact CNN, ResNet-18 trained from scratch, and ImageNet pretrained ResNet-18 with staged fine-tuning. Evaluation included 95% bootstrap confidence intervals, DeLong nonparametric AUC tests, Expected Calibration Error (ECE), nested cross-validation, feature ablation, and batch-effect PCA before and after normalization.

**Results.** ImageNet pretrained ResNet-18 achieved accuracy 0.939 (95% CI: 0.903-0.970), macro-OvR ROC AUC 0.996 (95% CI: 0.991-0.999), and ECE 0.042, significantly outperforming Random Forest (DeLong z = 7.47, p = 7.8e-14). Feature-based classifiers achieved AUC 0.899-0.924 with well-calibrated outputs. CNN trained from scratch (AUC 0.786) and ResNet-18 from scratch (AUC 0.668) both underperformed feature-based models, establishing that architecture alone does not confer advantage without pretrained representations.

**Conclusions.** Pretraining strategy is the most consequential methodological decision in small-sample phenotypic imaging studies. Feature-based pipelines remain competitive and interpretable alternatives when labelled data are limited. The framework is fully reproducible and publicly available.

## Keywords

BBBC021; high-content screening; mechanism-of-action profiling; transfer learning; calibration; DeLong test; reproducibility; feature engineering

## 1. Introduction

Automated microscopy analysis is a methodological necessity in high-content screening workflows. As image volumes increase, manual curation introduces delay, inconsistency, and irreproducible interpretation. Robust computational pipelines must therefore satisfy dual requirements: technical performance and scientific defensibility.

BBBC021 [1] is an established benchmark combining multi-channel fluorescence imaging with compound-treatment metadata [2]. However, many analysis reports focus narrowly on single performance metrics and omit reliability components such as calibration behavior, batch-effect characterization, statistical significance testing between models, and explicit overfitting checks.

To address this gap, we developed an end-to-end BBBC021 framework designed for prediction and for publication-level evidence quality. The objectives were fivefold:
1. Build a robust and interpretable spot-detection module.
2. Engineer a quantitative feature set suitable for biological interpretation.
3. Compare feature-based and deep-learning models under matched evaluation settings.
4. Validate results through calibration, ablation, statistical testing, and nested cross-validation.
5. Document outputs in manuscript-ready format with complete figure and table traceability.

## 2. Materials and Methods

### 2.1. Dataset and Study Scope

BBBC021 archives approximately 15,000 images across 113 compound treatments [1]. This study applied a two-stage selection: (1) retrieval from 16 publicly accessible zip archives yielded approximately 1,600 candidate images; (2) filtering to the 10 most-represented MOA classes, retaining only classes with at least 80 images in the combined train/validation/test pool, produced the final 824-image, 10-class dataset (560 train / 99 validation / 165 test). Metadata were used for image-channel resolution and downstream MOA label assignment. The complete pipeline and all outputs are available at https://github.com/SID-6921/BBBC021_Project.

### 2.2. Preprocessing and Detection

Available channels were fused into a grayscale representation per sample. Images were resized, intensity-normalized, and contrast-enhanced. Bright-region detection used adaptive Gaussian thresholding followed by morphological opening and closing. Contours were filtered by area and contour-level mean intensity to reduce spurious detections. This approach mirrors preprocessing common in CellProfiler pipelines [3].

### 2.3. Feature Engineering

Per-image descriptors included:

1. Abundance: spot count and density.
2. Intensity: mean, total, and variance.
3. Geometry: area coverage and spot-size distribution statistics (mean, standard deviation, median, q25, q75).
4. Compositional morphology: small, medium, and large spot fractions.

This 14-descriptor design preserves interpretability while capturing heterogeneous phenotypic signals. Group-wise distributions of key features across MOA classes are shown in Supplementary Figures S1-S3.

### 2.4. Supervised Modeling

Feature-based baselines were Logistic Regression and Random Forest. Deep-learning baselines included a compact CNN, a ResNet-18 trained from scratch, and a ResNet-18 transfer-learning model initialized with ImageNet 1K weights [6,7] and staged fine-tuning. Stage 1: all layers frozen except the final fully-connected layer, 10 epochs. Stage 2: last two residual blocks unfrozen, learning rate 1e-4, remaining epochs. All deep models used 224x224 inputs, the standard input resolution for ResNet architectures.

### 2.5. Statistical Validation

The following analyses were applied:

1. Confusion matrices for all models (Supplementary Figures S9-S11).
2. Per-class ROC AUC with 95% bootstrap confidence intervals.
3. DeLong tests [8] for AUC difference: CNN versus RF, and ResNet-18 pretrained versus RF.
4. Calibration curves and ECE [9] as overfitting diagnostics (Figure 6).
5. Nested cross-validation [10] on training data to assess generalization.
6. Feature ablation to validate descriptor importance (Figure 8, Table 4).
7. Batch-effect PCA before and after within-batch z-score normalization (Figure 7, Supplementary Figure S12).
8. Biological validation of top features against compound metadata (Table 5).
9. Computational cost comparison: training time and peak memory (Table 6).

### 2.6. Limitations

An independent external microscopy dataset was not included due to scope constraints. Nested cross-validation diagnostics are reported to address this limitation transparently. External transfer validation on BBBC014, BBBC020, or RxRx1 [11,12] is the immediate follow-up experiment.

## 3. Results

### 3.1. Feature-Based Performance

On the 165-image held-out test set (Table 1):

1. Logistic Regression: accuracy 0.600 (95% CI: 0.521-0.679), macro-OvR ROC AUC 0.899 (95% CI: 0.874-0.923), ECE 0.066.
2. Random Forest: accuracy 0.588 (95% CI: 0.521-0.667), macro-OvR ROC AUC 0.924 (95% CI: 0.901-0.943), ECE 0.092.

Nested cross-validation on training data confirmed stability: LR AUC 0.898 +/- 0.018; RF AUC 0.907 +/- 0.015. The gap between nested CV and held-out test AUC was less than 0.03 for both models, indicating no material overfitting. ROC curves are shown in Figure 3. Confusion matrices are provided in Supplementary Figures S9 (LR) and S10 (RF). Feature importance rankings from the RF model are shown in Figure 4.

### 3.2. Deep-Learning Performance

Deep-learning results (Table 2, Figure 5):

1. CNN (from scratch): accuracy 0.412 (95% CI: 0.345-0.473), AUC 0.786 (95% CI: 0.758-0.819), ECE 0.080.
2. ResNet-18 (from scratch): accuracy 0.182 (95% CI: 0.133-0.236), AUC 0.668 (95% CI: 0.644-0.699), ECE 0.651. The elevated ECE reflects severe overconfidence when an 11-million-parameter architecture is trained from random initialization on fewer than 600 examples.
3. ResNet-18 (ImageNet pretrained, staged fine-tuning): accuracy 0.939 (95% CI: 0.903-0.970), AUC 0.996 (95% CI: 0.991-0.999), ECE 0.042. This improvement is attributable to pretrained feature reuse; the architecture and training procedure were otherwise identical to the scratch variant.

The confusion matrix for pretrained ResNet-18 is provided in Supplementary Figure S11. Training and validation loss curves for the compact CNN are shown in Supplementary Figure S8.

### 3.3. Statistical Comparison

DeLong tests confirmed statistically significant AUC differences:

1. CNN (scratch) versus Random Forest: z = -7.77, p = 7.6e-15 (RF superior).
2. ResNet-18 (pretrained) versus Random Forest: z = 7.47, p = 7.8e-14 (ResNet-18 pretrained superior).

Both results survive any standard multiple-testing correction. Full DeLong statistics are provided in Table 9.

### 3.4. Calibration

ECE values were low for LR (0.066), RF (0.092), CNN (0.080), and pretrained ResNet-18 (0.042), indicating well-calibrated probability outputs. ResNet-18 trained from scratch showed markedly elevated ECE (0.651), consistent with overconfident predictions in under-trained deep networks. Calibration curves are shown in Figure 6.

### 3.5. Ablation, Batch Effects, and Biological Validation

Feature ablation on the RF model demonstrated that descriptor importance is genuine and non-redundant. Removing the highest-ranked feature (mean_intensity) had negligible impact (AUC: 0.924 to 0.921, delta-AUC = 0.003). Removing the top five features reduced AUC to 0.796 (delta-AUC = 0.128), and removing the top eight produced a decline to 0.727 (delta-AUC = 0.197), confirming that the full 14-descriptor set is collectively informative (Table 4, Figure 8).

PCA of feature vectors revealed pronounced batch clustering before normalization, with weekly acquisition groups forming distinct spatial clusters in the first two principal components (Supplementary Figure S12). After within-batch z-score normalization, batch-driven separation was substantially reduced, confirming that normalization effectively removes technical variation (Figure 7).

Biological validation confirmed that the top five morphological features (mean_intensity, total_intensity, area_covered_ratio, spot_count, density_spots_per_10k_px) yield statistically significant MOA-class separation (Kruskal-Wallis H > 400, p < 1e-95; Table 5). Spearman correlation with compound concentration was consistent across discriminative descriptors (|rho| approximately 0.20, p < 1e-8). Note that mean_intensity and total_intensity, and spot_count and density_spots_per_10k_px, form linearly dependent pairs within fixed-area images; removing one member of each pair is recommended in future feature-selection steps.

### 3.6. Computational Cost

Logistic Regression (0.08 s, 379 MB peak RAM) and Random Forest (1.2 s, 317 MB) were computationally negligible. CNN training required 42 s and 858 MB. ResNet-18 from scratch required 313 s and 2,001 MB. Pretrained ResNet-18 completed in 226 s (1,976 MB), faster than scratch training because frozen layers require no gradient computation during Stage 1 (Table 6).

## 4. Discussion

This study demonstrates the decisive impact of pretraining strategy in small-sample fluorescence microscopy classification. When a ResNet-18 is initialized with ImageNet weights and fine-tuned with staged learning rates, it achieves accuracy 0.939 and AUC 0.996 on a 10-class MOA benchmark, significantly outperforming all other evaluated models (DeLong p = 7.8e-14 versus RF). Training the same architecture from random initialization yields 18.2% accuracy and AUC 0.668, an outcome below the simplest linear baseline. This contrast provides direct, controlled evidence that pretraining strategy matters more than architectural choice when labelled data are scarce.

Feature-based classifiers (LR and RF) remain valuable as interpretable, auditable baselines: they achieve AUC 0.899 and 0.924, respectively, with well-characterized feature importance, calibration behavior, and negligible training cost. Their lower accuracy relative to pretrained ResNet-18 reflects the inherent limits of hand-crafted spot descriptors under 10-class imbalance, not a failure of the feature-engineering paradigm.

The broader contribution is methodological rigor. Combining DeLong testing, calibration analysis, nested cross-validation, batch-effect auditing, and biological validation moves the study beyond point-metric reporting toward reproducible scientific inference. This is particularly relevant for compound mechanism-of-action profiling, where model reliability and interpretability support translational credibility.

The primary limitation is the absence of external dataset validation. Nested cross-validation diagnostics confirm internal generalization, but cross-dataset transfer experiments on BBBC014, BBBC020, or RxRx1 are necessary to characterize domain shift sensitivity. These experiments are designated as the immediate follow-up study.

## 5. Conclusions

When labelled data are limited (fewer than 1,000 images per experiment), the most important methodological decision is whether to use pretrained representations, not which classifier or architecture to select. ImageNet pretrained ResNet-18 with staged fine-tuning delivered the strongest performance across all three metrics: accuracy 0.939, AUC 0.996, and ECE 0.042.

Feature-based pipelines are the best first choice when interpretability, auditability, or computational efficiency are primary constraints. They provide competitive AUC (0.899-0.924) with minimal training cost and full decision transparency.

The pipeline and all outputs are publicly available at https://github.com/SID-6921/BBBC021_Project to support reproducibility.

## Data Availability

All generated figures, tables, source code, and result summaries are available at: https://github.com/SID-6921/BBBC021_Project

## Conflicts of Interest

The authors declare no conflict of interest.

## Author Contributions

Conceptualization, methodology, software, validation, formal analysis, visualization, and writing (original draft): N.N.

## Funding

This research received no external funding.

## Acknowledgments

The authors acknowledge the Broad Bioimage Benchmark Collection for providing the BBBC021 dataset.

## References

1. Ljosa, V.; Sokolnicki, K.L.; Carpenter, A.E. Annotated high-throughput microscopy image sets for validation. Nat. Methods 2012, 9, 637.
2. Caie, P.D.; Walls, R.E.; Ingleston-Orme, A.; Daya, S.; Houslay, T.; Eagle, R.; Roberts, M.E.; Carragher, N.O. High-content phenotypic profiling of drug response signatures across distinct cancer cells. Mol. Cancer Ther. 2010, 9, 1913-1926.
3. Carpenter, A.E.; Jones, T.R.; Lamprecht, M.R.; Clarke, C.; Kang, I.H.; Friman, O.; Guertin, D.A.; Chang, J.H.; Lindquist, R.A.; Moffat, J.; et al. CellProfiler: image analysis software for identifying and quantifying cell phenotypes. Genome Biol. 2006, 7, R100.
4. Moen, E.; Bannon, D.; Kudo, T.; Graf, W.; Covert, M.; Van Valen, D. Deep learning for cellular image analysis. Nat. Methods 2019, 16, 1233-1246.
5. Ching, T.; Himmelstein, D.S.; Beaulieu-Jones, B.K.; Kalinin, A.A.; Do, B.T.; Way, G.P.; Ferrero, E.; Agapow, P.M.; Zietz, M.; Hoffman, M.M.; et al. Opportunities and obstacles for deep learning in biology and medicine. J. R. Soc. Interface 2018, 15, 20170387.
6. He, K.; Zhang, X.; Ren, S.; Sun, J. Deep residual learning for image recognition. Proc. IEEE Conf. Comput. Vis. Pattern Recognit. 2016, 770-778.
7. Deng, J.; Dong, W.; Socher, R.; Li, L.-J.; Li, K.; Fei-Fei, L. ImageNet: A large-scale hierarchical image database. Proc. IEEE Conf. Comput. Vis. Pattern Recognit. 2009, 248-255.
8. DeLong, E.R.; DeLong, D.M.; Clarke-Pearson, D.L. Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach. Biometrics 1988, 44, 837-845.
9. Guo, C.; Pleiss, G.; Sun, Y.; Weinberger, K.Q. On calibration of modern neural networks. Proc. Int. Conf. Mach. Learn. 2017, 70, 1321-1330.
10. Varma, S.; Simon, R. Bias in error estimation when using cross-validation for model selection. BMC Bioinform. 2006, 7, 91.
11. Bray, M.A.; Singh, S.; Yost, H.J.; Carpenter, A.E. Cell painting, a high-content image-based assay for morphological profiling using multiplexed fluorescent dyes. Nat. Protoc. 2016, 11, 1757-1774.
12. Sypetkowski, M.; Rezanejad, M.; Saberian, S.; Kraus, O.; Urbanik, J.; Taylor, J.; Mabey, B.; Victors, M.; Yosinski, J.; Rezanejad, A.; et al. RxRx1: A dataset for evaluating experimental batch correction methods. Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. 2023, 4284-4293.
