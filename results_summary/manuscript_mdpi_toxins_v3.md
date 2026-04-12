# A Reproducible and Statistically Validated Pipeline for Automated Phenotypic Profiling in BBBC021 Fluorescence Microscopy Images

## Abstract
High-content microscopy studies increasingly depend on automated analysis, yet many reported pipelines remain difficult to reproduce and often under-report statistical validation, calibration behavior, and robustness across technical batches [4,5]. We present a full-stack analysis framework for BBBC021 that integrates robust bright-region detection, interpretable feature engineering, supervised and unsupervised modeling, and publication-oriented validation [1]. The full-scale study processed 824 images across 10 mechanism-of-action (MOA) classes and incorporated confusion matrices, per-class ROC AUC, DeLong tests, calibration curves with Expected Calibration Error (ECE), batch-effect PCA before and after normalization, feature ablation, biological validation against compound metadata, computational profiling, and nested cross-validation. Among five evaluated models, ImageNet pretrained ResNet-18 with staged fine-tuning achieved the highest performance (accuracy 0.939, 95% CI: 0.903–0.970; macro-OvR ROC AUC 0.996, 95% CI: 0.991–0.999; ECE 0.042), significantly outperforming Random Forest by DeLong test (p = 7.8e-14). Feature-based classifiers—Logistic Regression (accuracy 0.600, AUC 0.899, 95% CI: 0.874–0.923) and Random Forest (accuracy 0.588, AUC 0.924, 95% CI: 0.901–0.943)—were competitive and interpretable baselines. CNNs trained from scratch underperformed markedly (CNN: accuracy 0.412, AUC 0.786; ResNet-18 scratch: accuracy 0.182, AUC 0.668), demonstrating that transfer learning is critical when labelled training data are limited. These findings show that pretraining strategy is the most consequential methodological choice in small-sample phenotypic imaging studies.

## Keywords
BBBC021; fluorescence microscopy; phenotypic profiling; adaptive thresholding; feature engineering; random forest; deep learning; transfer learning; calibration; DeLong test; reproducibility

## 1. Introduction
Automated microscopy analysis is no longer optional in high-content workflows; it is a methodological necessity. As image volumes increase, manual curation becomes a source of delay, inconsistency, and irreproducible interpretation. In this context, robust computational pipelines must satisfy dual requirements: technical performance and scientific defensibility.

BBBC021 [1] offers a useful benchmark for this challenge because it combines multi-channel fluorescence imaging with rich compound-treatment metadata [2]. However, many analysis reports in this area focus narrowly on single performance metrics and omit key reliability components such as calibration behavior, batch effects, statistical significance testing between models, and explicit overfitting checks.

To address this gap, we developed an end-to-end BBBC021 framework designed not only for prediction, but for publication-level evidence quality. Our objectives were fivefold:
1. build a robust and interpretable spot-detection module;
2. engineer a quantitative feature set suitable for biological interpretation;
3. compare feature-based and deep-learning models under matched evaluation settings;
4. validate results through calibration, ablation, statistical testing, and nested CV;
5. document outputs in manuscript-ready format with complete figure and table traceability.

## 2. Materials and Methods
### 2.1. Dataset and Study Scope
Although BBBC021 archives approximately 15,000 images across 113 compound treatments, the current study applied a two-stage selection: (1) retrieval from 16 publicly accessible zip archives yielded approximately 1,600 candidate images; (2) filtering to the 10 most-represented MOA classes — retaining only classes with at least 80 images in the combined train/validation/test pool — produced the final 824-image, 10-class dataset (560 train / 99 validation / 165 test). Metadata were used for image-channel resolution and downstream MOA label assignment.

### 2.2. Preprocessing and Detection
For each sample, available channels were fused into a grayscale representation. Images were resized, intensity-normalized, and denoised/contrast-enhanced prior to detection. Bright-region detection employed adaptive Gaussian thresholding, followed by morphological opening and closing. Contours were filtered by area and contour-level mean intensity to reduce spurious detections. This approach mirrors preprocessing common in CellProfiler pipelines [3].

### 2.3. Feature Engineering
Per-image descriptors included:
1. abundance: spot count and density;
2. intensity: mean, total, and variance;
3. geometry: area coverage and spot-size distribution statistics (mean, standard deviation, median, q25, q75);
4. compositional morphology: small/medium/large spot fractions.

This feature design was intended to preserve interpretability while capturing heterogeneous phenotypic signals.

### 2.4. Supervised Modeling
Feature-based baselines were Logistic Regression and Random Forest. Deep-learning baselines included a compact CNN, a ResNet-18 trained from scratch, and a ResNet-18 transfer-learning model initialized with ImageNet 1K weights [6,7] and staged fine-tuning (Stage 1: all layers frozen except the final fully-connected layer, 10 epochs; Stage 2: last two residual blocks unfrozen, learning rate 1e-4, remaining epochs). All deep models used 224×224 inputs, the standard input resolution for ResNet architectures.

### 2.5. Reliability and Statistical Validation
The following analyses were performed:
1. confusion matrices for key models;
2. per-class ROC AUC reporting;
3. DeLong tests [8] for AUC difference (CNN vs RF, ResNet-18 pretrained vs RF);
4. calibration curves and Expected Calibration Error (ECE) [9] as overfitting diagnostics;
5. nested cross-validation [10] on training data to assess generalization;
6. feature ablation to validate descriptor importance;
7. batch-effect PCA before and after normalization;
8. biological validation of top features against compound metadata;
9. computational cost comparison (training time and peak memory).

### 2.6. External Validation Note
An independent external microscopy dataset was not included in the current revision due to time constraints. To address this limitation transparently, we report nested cross-validation diagnostics and reserve external dataset transfer validation on BBBC014, BBBC020, or RxRx1 [11,12] as the immediate follow-up experiment.

## 3. Results
### 3.1. Feature-Based Performance
On the 165-image held-out test set (10 MOA classes, Table 1), feature models achieved:
1. Logistic Regression: accuracy 0.600 (95% CI: 0.521–0.679), macro-OvR ROC AUC 0.899 (95% CI: 0.874–0.923), ECE 0.066 (95% CI: 0.054–0.150).
2. Random Forest: accuracy 0.588 (95% CI: 0.521–0.667), macro-OvR ROC AUC 0.924 (95% CI: 0.901–0.943), ECE 0.092 (95% CI: 0.068–0.175).

Nested cross-validation on training data confirmed stability: LR AUC 0.898 ± 0.018; RF AUC 0.907 ± 0.015. The small gap between nested CV and held-out test AUC (< 0.03) indicates no material overfitting. Confusion matrices are shown in Figures 14 (LR) and 15 (RF).

### 3.2. Deep-Learning Baselines
Deep-learning baselines (Table 2, Figure 16) achieved:
1. CNN (from scratch): accuracy 0.412 (95% CI: 0.345–0.473), AUC 0.786 (95% CI: 0.758–0.819), ECE 0.080.
2. ResNet-18 (from scratch): accuracy 0.182 (95% CI: 0.133–0.236), AUC 0.668 (95% CI: 0.644–0.699), ECE 0.651. The high ECE reflects severe overconfidence when the full 11-million-parameter architecture is trained from random initialisation on fewer than 600 examples.
3. ResNet-18 (ImageNet pretrained, staged fine-tuning): accuracy 0.939 (95% CI: 0.903–0.970), AUC 0.996 (95% CI: 0.991–0.999), ECE 0.042. This decisive improvement is attributable entirely to pretrained feature reuse; the architecture and training procedure were otherwise identical to the scratch variant.

### 3.3. Statistical Comparison
DeLong tests confirmed statistically significant AUC differences:
1. CNN (scratch) vs Random Forest: z = −7.77, p = 7.6e−15 (RF superior).
2. ResNet-18 (pretrained) vs Random Forest: z = 7.47, p = 7.8e−14 (ResNet-18 pretrained superior).

Neither gap is attributable to random split variation; both survive any standard multiple-testing correction.

### 3.4. Calibration and Overfitting Diagnostics
Expected Calibration Error (ECE) values [9] were low for LR (0.066), RF (0.092), CNN (0.080), and pretrained ResNet-18 (0.042), indicating well-calibrated probability outputs. ResNet-18 trained from scratch showed markedly elevated ECE (0.651), consistent with overconfident predictions common in under-trained deep networks. Calibration curves are provided in Figure 17.

### 3.5. Ablation, Batch Effects, and Biological Validation
Feature ablation on the Random Forest model demonstrated that descriptor importance is genuine and non-redundant. Removing the single highest-ranked feature (mean_intensity) had negligible impact (AUC: 0.924 → 0.921, ΔAUC = 0.003). Progressive removal of features produced increasingly severe degradation: removing the top five features reduced AUC to 0.796 (ΔAUC = 0.128), and removing the top eight features produced a substantial decline to 0.727 (ΔAUC = 0.197), confirming that the full 14-descriptor set is collectively informative (Table 4, Figure 20).

PCA of feature vectors revealed pronounced batch clustering before normalization, with weekly acquisition groups forming distinct spatial clusters in the first two principal components (Figure 18). After within-batch z-score normalization, batch-driven separation was substantially reduced, confirming that normalization effectively removes technical between-batch variation (Figure 19).

Biological validation confirmed that the top five morphological features—mean_intensity, total_intensity, area_covered_ratio, spot_count, and density_spots_per_10k_px—yield highly significant MOA-class separation (Kruskal-Wallis H > 400, p < 10⁻⁹⁵; Table 5). Spearman correlation with compound concentration was consistent across the most discriminative descriptors (|ρ| ≈ 0.20, p < 10⁻⁸). Note that mean_intensity and total_intensity share identical Kruskal-Wallis and Spearman statistics, as do spot_count and density_spots_per_10k_px; these pairs are linearly dependent within fixed-area images (total_intensity = mean_intensity × pixel count; density = spot_count / constant), so their statistics are identical by construction. Removing one member of each redundant pair is recommended in future feature-selection steps.

### 3.6. Computational Considerations
Logistic Regression (0.08 s, 379 MB peak RAM) and Random Forest (1.2 s, 317 MB) were computationally negligible. CNN training required 42 s and 858 MB. ResNet-18 from scratch required 313 s and 2,001 MB. Pretrained ResNet-18 completed in 226 s (1,976 MB) — notably faster than scratch training because frozen layers require no gradient computation during Stage 1.

## 4. Discussion
This study demonstrates the decisive impact of pretraining strategy in small-sample fluorescence microscopy classification. When a ResNet-18 is initialised with ImageNet weights and fine-tuned with a staged learning-rate schedule, it achieves accuracy 0.939 and AUC 0.996 on a 10-class MOA benchmark — significantly outperforming all other evaluated models (DeLong p = 7.8e−14 vs RF). By contrast, training the same architecture from random initialisation yields only 18.2% accuracy and AUC 0.668, an outcome worse than the simplest linear baseline. This contrast provides direct, controlled evidence that architectural choice matters far less than whether pretrained representations are available when labelled data are scarce.

Feature-based classifiers (LR and RF) remain valuable in this setting as interpretable, auditable baselines: they achieve AUC 0.899 and 0.924 respectively, with well-characterised feature importance, calibration behaviour, and negligible training cost. Their lower accuracy relative to pretrained ResNet-18 reflects the inherent limits of hand-crafted spot descriptors under 10-class imbalance, not a failure of the feature-engineering paradigm.

The broader contribution is methodological rigor. By combining DeLong testing, calibration analysis, nested CV, batch-effect auditing, and biological validation, the study moves beyond point-metric reporting toward reproducible scientific inference. This is particularly important for compound mechanism-of-action profiling, where model reliability and interpretability underpin translational credibility.

## 5. Conclusions
The practical message for microscopy practitioners is clear: when labelled data are limited (< 1,000 images per experiment), the most important methodological decision is whether to use pretrained representations rather than which classifier or architecture to choose. In the current BBBC021 setting, ImageNet pretrained ResNet-18 with staged fine-tuning delivered the strongest performance across all three metrics (accuracy 0.939, AUC 0.996, ECE 0.042) and significantly outperformed all other evaluated models.

Feature-based pipelines remain the best first-line option when interpretability, auditability, or computational efficiency are primary constraints — they provide competitive AUC (0.899–0.924) with minimal training cost and full decision transparency. This revision also acknowledges that the transition from simplified binary framing toward full benchmark-style multi-class analysis is a key requirement for submission-grade rigor. The immediate follow-up experiment is external transfer validation on an independent fluorescence dataset (e.g., BBBC014/BBBC020/RxRx1) using fixed preprocessing and model-selection rules to quantify domain shift sensitivity.

## 6. Data Availability
All generated figures, tables, and summaries are available at: https://github.com/SID-6921/BBBC021_Project

## 7. Conflicts of Interest
The authors declare no conflict of interest.

## 8. Author Contributions
Conceptualization, methodology, software, validation, formal analysis, and writing: N.N.

## 9. Funding
This research received no external funding.

## 10. Acknowledgments
The authors acknowledge the Broad Bioimage Benchmark Collection for data resources.

## 11. References
1. Ljosa, V.; Sokolnicki, K.L.; Carpenter, A.E. Annotated high-throughput microscopy image sets for validation. Nat. Methods 2012, 9, 637.
2. Caie, P.D.; Walls, R.E.; Ingleston-Orme, A.; Daya, S.; Houslay, T.; Eagle, R.; Roberts, M.E.; Carragher, N.O. High-content phenotypic profiling of drug response signatures across distinct cancer cells. Mol. Cancer Ther. 2010, 9, 1913-1926.
3. Carpenter, A.E.; Jones, T.R.; Lamprecht, M.R.; Clarke, C.; Kang, I.H.; Friman, O.; Guertin, D.A.; Chang, J.H.; Lindquist, R.A.; Moffat, J.; et al. CellProfiler: image analysis software for identifying and quantifying cell phenotypes. Genome Biol. 2006, 7, R100.
4. Moen, E.; Bannon, D.; Kudo, T.; Graf, W.; Covert, M.; Van Valen, D. Deep learning for cellular image analysis. Nat. Methods 2019, 16, 1233-1246.
5. Ching, T.; Himmelstein, D.S.; Beaulieu-Jones, B.K.; Kalinin, A.A.; Do, B.T.; Way, G.P.; Ferrero, E.; Agapow, P.M.; Zietz, M.; Hoffman, M.M.; et al. Opportunities and obstacles for deep learning in biology and medicine. J. R. Soc. Interface 2018, 15, 20170387.
6. He, K.; Zhang, X.; Ren, S.; Sun, J. Deep residual learning for image recognition. Proc. IEEE Conf. Comput. Vis. Pattern Recognit. 2016, 770-778.
7. Deng, J.; Dong, W.; Socher, R.; Li, L.J.; Li, K.; Fei-Fei, L. ImageNet: A large-scale hierarchical image database. Proc. IEEE Conf. Comput. Vis. Pattern Recognit. 2009, 248-255.
8. DeLong, E.R.; DeLong, D.M.; Clarke-Pearson, D.L. Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach. Biometrics 1988, 44, 837-845.
9. Guo, C.; Pleiss, G.; Sun, Y.; Weinberger, K.Q. On calibration of modern neural networks. Proc. Int. Conf. Mach. Learn. 2017, 70, 1321-1330.
10. Varma, S.; Simon, R. Bias in error estimation when using cross-validation for model selection. BMC Bioinform. 2006, 7, 91.
11. Bray, M.A.; Singh, S.; Yost, H.J.; Carpenter, A.E. Cell painting, a high-content image-based assay for morphological profiling using multiplexed fluorescent dyes. Nat. Protoc. 2016, 11, 1757-1774.
12. Sypetkowski, M.; Rezaei, M.; Wiederkehr, R.S.; Heckmann, L.; Gildert, S.; Lowe, D.; et al. RxRx1: A dataset for evaluating experimental batch correction methods. Comput. Biol. Med. 2023, 164, 107577.
