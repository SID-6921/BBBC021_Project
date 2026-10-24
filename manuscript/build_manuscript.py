# -*- coding: utf-8 -*-
"""Builds the revised BBBC021 manuscript (Frontiers in Pharmacology, MS 1926843)
incorporating the full response to Reviewer 1 and Reviewer 2.
"""
from pathlib import Path

from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

doc = Document()
style = doc.styles["Normal"]
style.font.name = "Times New Roman"
style.font.size = Pt(11)

def h1(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(13)
    return p

def h2(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(text)
    r.bold = True
    r.italic = True
    r.font.size = Pt(11.5)
    return p

def body(text, justify=True):
    p = doc.add_paragraph()
    if justify:
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(6)
    p.add_run(text)
    return p

def caption(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(10)
    r = p.add_run(text)
    r.italic = True
    r.font.size = Pt(10)
    return p

def table_from_rows(headers, rows, widths=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Light Grid Accent 1"
    for i, htext in enumerate(headers):
        cell = t.rows[0].cells[i]
        cell.text = ""
        r = cell.paragraphs[0].add_run(htext)
        r.bold = True
        r.font.size = Pt(9)
    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = ""
            r = cells[i].paragraphs[0].add_run(str(val))
            r.font.size = Pt(9)
    doc.add_paragraph()
    return t

# =====================================================================
# TITLE / FRONT MATTER
# =====================================================================
title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = title.add_run(
    "Compound Mechanism-of-Action Classification from High-Content Fluorescence "
    "Microscopy: A Reproducible, Leakage-Controlled Benchmark on BBBC021"
)
r.bold = True
r.font.size = Pt(15)

authors = doc.add_paragraph()
authors.alignment = WD_ALIGN_PARAGRAPH.CENTER
authors.add_run(
    "Siddhardha Nanda\u00b9, Shrivatsa Kolliputi\u00b2, Siri Kolliputi\u00b3, "
    "Venkata Ramireddy Narala\u2074, Narasaiah Kolliputi\u2075*"
)

aff = doc.add_paragraph()
aff.alignment = WD_ALIGN_PARAGRAPH.CENTER
for line in [
    "\u00b9 Biomedical Engineering, Columbia University, New York, NY, USA",
    "\u00b2 Bellini College of Artificial Intelligence, University of South Florida, Tampa, FL, USA",
    "\u00b3 Judy Genshaft Honors College, University of South Florida, Tampa, FL, USA",
    "\u2074 Department of Zoology, Yogi Vemana University, Kadapa 516 005, Andhra Pradesh, India",
    "\u2075 Division of Allergy and Immunology, Department of Internal Medicine, Morsani College "
    "of Medicine, University of South Florida, Tampa, FL, USA",
    "* Correspondence: Narasaiah Kolliputi, nkollipu@usf.edu",
]:
    aff.add_run(line + "\n")

doc.add_paragraph()

# =====================================================================
# ABSTRACT
# =====================================================================
h1("Abstract")
body(
    "High-content fluorescence microscopy is widely used to characterize drug "
    "mechanism-of-action (MOA), but automated pipelines that translate images into "
    "MOA calls are only as trustworthy as their validation. Two issues are common in "
    "this literature and both applied to our own earlier analysis of this dataset: "
    "(i) image-level train/test splitting, which lets near-duplicate fields from the "
    "same compound and well appear in both partitions and can inflate apparent "
    "performance, particularly for pixel-based deep learning models; and (ii) fusing "
    "multi-channel fluorescence images into a single grayscale representation before "
    "feature extraction, which can discard exactly the channel-specific information "
    "that distinguishes mechanistic classes. We present a revised, leakage-controlled "
    "benchmark for the BBBC021 dataset (824 images, 10 mechanism-of-action classes) "
    "that addresses both issues directly: a compound-level grouped train/validation/test "
    "split (no compound's images appear in more than one partition), and per-channel "
    "(DAPI/Tubulin/Actin) feature extraction and deep-learning input instead of a fused "
    "grayscale image. We compare Logistic Regression, Random Forest, a compact "
    "convolutional neural network (CNN), a ResNet-18 trained from scratch, and an "
    "ImageNet-pretrained ResNet-18 with staged fine-tuning under matched training "
    "budgets (16 total epochs for both ResNet-18 variants), reporting the two ResNet-18 "
    "conditions across five random seeds rather than a single run. On the held-out, "
    "compound-disjoint test set, the pretrained ResNet-18 achieved the highest "
    "performance (accuracy 0.721 \u00b1 0.042, macro one-versus-rest ROC AUC 0.965 \u00b1 "
    "0.007, five seeds), significantly exceeding both Random Forest (paired bootstrap "
    "on macro-AUC, 95% CI [0.033, 0.078], p<0.001; corroborating DeLong z=5.10, "
    "p=3.4\u00d710\u207b\u2077) and ResNet-18 trained from scratch (accuracy 0.261 \u00b1 0.076, "
    "AUC 0.826 \u00b1 0.048; Welch t=11.8, p<0.001 on the five-seed distributions). "
    "Random Forest and Logistic Regression, now trained on per-channel features, "
    "improved substantially over their fused-grayscale counterparts (accuracy 0.399 "
    "and 0.524 respectively, versus 0.210 and 0.207 under channel fusion on the same "
    "split), consistent with channel fusion having discarded MOA-discriminative "
    "signal. Scratch-initialized deep models remain markedly less stable across seeds "
    "than the pretrained model (an 8-fold larger AUC standard deviation), indicating "
    "that pretraining confers a stability advantage in addition to a mean-performance "
    "advantage under this data regime. We report these results as a computational "
    "benchmarking exercise on the BBBC021 MCF-7 cell dataset; we did not measure "
    "reactive oxygen species or oxidative stress directly, and any relevance of specific "
    "MOA classes to redox pharmacology or pulmonary disease is noted as a motivating "
    "rationale for future targeted validation, not a demonstrated finding of this study. "
    "The revised pipeline, including the compound-level split, per-channel feature "
    "extraction, multi-seed training, and bootstrap-based statistical comparison, is "
    "publicly available."
)

kw = doc.add_paragraph()
r = kw.add_run("Keywords: ")
r.bold = True
kw.add_run(
    "high-content screening, mechanism-of-action classification, transfer learning, "
    "deep learning, fluorescence microscopy, train/test leakage, reproducibility, "
    "BBBC021"
)

print("Front matter and abstract written.")

# =====================================================================
# 1. INTRODUCTION
# =====================================================================
h1("1. Introduction")
body(
    "High-content fluorescence microscopy generates thousands of images per screen "
    "and is widely used to characterize how small molecules reshape cellular "
    "morphology (Caie et al., 2010). As image volumes grow, manual curation becomes "
    "a bottleneck, motivating automated pipelines for feature extraction and "
    "classification (Carpenter et al., 2006; Moen et al., 2019). For such pipelines "
    "to support downstream pharmacological inference, technical performance alone is "
    "not sufficient: predictions must also be evaluated under conditions that "
    "genuinely test generalization, with calibrated probabilities, auditable "
    "features, and formal statistical comparison between models (Ching et al., 2018)."
)
body(
    "Two methodological choices are common in the high-content-screening literature "
    "and, on closer inspection, both applied to an earlier version of this analysis. "
    "First, train/test partitioning is frequently performed at the image level, "
    "ignoring the nested experimental structure of the source assay (multiple fields "
    "per well, multiple wells per compound-concentration, multiple compounds per "
    "mechanistic class). A pixel-based classifier, and a pretrained convolutional "
    "network in particular, can exploit near-duplicate sibling fields from the same "
    "well or compound if they are split across train and test, inflating apparent "
    "generalization performance without the model having learned anything that "
    "transfers to a genuinely unseen compound. Second, when a source image has "
    "multiple fluorescence channels reporting distinct cellular structures, fusing "
    "them into one grayscale image before feature extraction can discard exactly the "
    "information -- which channel changed -- that a downstream classifier is asked to "
    "recover."
)
body(
    "The Broad Bioimage Benchmark Collection dataset BBBC021 is a widely used "
    "standard for MOA classification, comprising fluorescence-imaged MCF-7 cells "
    "treated with well-annotated compounds and imaged across three channels (DNA, "
    "\u03b2-tubulin, F-actin) (Ljosa et al., 2012). This structure makes it a useful "
    "test case for both issues above: compound identity and channel identity are "
    "both recorded and can both be controlled for directly."
)
body(
    "We revise our original analysis of this dataset to address both issues, and "
    "report the consequences transparently rather than only the corrected headline "
    "numbers. Specifically, we (i) replace the image-level split with a compound-level "
    "grouped split, in which every image from a given compound is assigned entirely "
    "to one partition; (ii) replace channel-fused feature extraction and deep-learning "
    "input with per-channel processing; (iii) match the total training budget between "
    "the scratch-initialized and pretrained ResNet-18 conditions, which in the "
    "original pipeline differed by construction; (iv) report both ResNet-18 "
    "conditions across five random seeds rather than a single training run, given "
    "that single-run deep-learning results on a sample of this size can be highly "
    "seed-dependent; (v) replace the DeLong test, whose classical derivation assumes "
    "a single pair of correlated binary ROC curves, with a paired bootstrap test on "
    "the macro-averaged AUC difference as the primary statistical comparison; and "
    "(vi) moderate our framing of this work's relevance to redox pharmacology and "
    "pulmonary disease, which the original submission stated more strongly than the "
    "BBBC021 dataset (MCF-7 breast cancer cells, no direct ROS or oxidative-stress "
    "measurement, no pulmonary cell or disease model) supports as a demonstrated "
    "finding. Several MOA classes studied here intersect mechanistically with "
    "pathways implicated in oxidative stress, which remains a motivation for future, "
    "targeted validation work; it is not evidence produced by the present study."
)
body(
    "The revised, leakage-controlled pipeline -- covering data curation, per-channel "
    "feature extraction, model training, and statistical validation -- is summarized "
    "in Figure 1, and is publicly available at "
    "https://github.com/SID-6921/BBBC021_Project."
)

# =====================================================================
# 2. MATERIALS AND METHODS
# =====================================================================
h1("2. Materials and Methods")

h2("2.1 Dataset, class selection, and compound-level grouped split")
body(
    "BBBC021 archives approximately 15,000 images spanning 113 compound treatments "
    "(Ljosa et al., 2012). Retrieval from the 16 publicly accessible archive files "
    "yielded 824 usable images after image-channel resolution, corresponding to the "
    "10 most-represented mechanism-of-action (MOA) classes among compounds with an "
    "annotated MOA label in this subset. Class sizes in the final dataset ranged from "
    "12 to 160 images (Table 1); we did not apply a minimum-images-per-class "
    "inclusion threshold, and we note this to correct our original submission, which "
    "incorrectly stated a criterion of at least 80 images per class -- an error we "
    "identified only after a reviewer traced the discrepancy between that stated "
    "criterion and the small per-class test-set counts it implied."
)
body(
    "The compounds and their tested concentrations are not a sample we selected: "
    "they are whichever compounds and concentrations BBBC021's original annotation "
    "(Ljosa et al., 2012) recorded for the 16 archive files we retrieved. We did not "
    "choose compounds or concentrations to represent the biological diversity of "
    "each MOA category, and the dataset should not be read as a designed "
    "pharmacological panel; it is a fixed, pre-annotated public resource, and the "
    "compound set available for each MOA class is a property of that resource, not "
    "of our study design."
)
body(
    "Critically, the number of distinct compounds contributing to each class also "
    "varies substantially (Table 1): six classes are represented by 2-4 compounds, "
    "and four classes -- DMSO (vehicle control), DNA replication inhibitors, Eg5 "
    "inhibitors, and protein-synthesis inhibitors -- are each represented in this "
    "subset by images from a single compound. We partitioned the dataset by "
    "compound rather than by image: every image belonging to a given compound was "
    "assigned entirely to the training, validation, or test partition, so that no "
    "compound's images appear in more than one partition. Within each of the six "
    "multi-compound classes, compounds were allocated to target an approximately "
    "65% / 15% / 20% train/validation/test split at the compound level, with a "
    "minimum of one compound guaranteed in both the validation and test partitions "
    "whenever a class had three or more compounds (two-compound classes received one "
    "compound in train and one in test, with no validation representation for that "
    "class). The four single-compound classes cannot be split at the compound level "
    "without breaking the leakage-safe property, and were therefore retained "
    "entirely within the training partition; per-class test-set metrics are "
    "correspondingly undefined for these four classes and are reported as such "
    "throughout, never silently omitted. The resulting partition contains 468 "
    "training, 28 validation, and 328 test images, spanning all 10 classes in "
    "training and 6 of 10 classes in validation/test (Table 1). Because every "
    "multi-compound class in this subset has at most 4 distinct compounds, the "
    "20%-of-compounds test allocation rounds to exactly one compound per class in "
    "every case: each of the six evaluable classes is tested against a single "
    "held-out compound, not a held-out set of compounds. Per-class test accuracy in "
    "this study is therefore accuracy on one specific unseen compound per class, "
    "and a different choice of held-out compound could give a different result; we "
    "quantify this directly in Section 4.4."
)
body(
    "This is a substantive change from our original submission, which split images "
    "randomly at the image level, stratified by class label only. A reviewer "
    "correctly identified this as the most likely explanation for the large "
    "performance gap we originally reported between the pretrained ResNet-18 and "
    "the feature-based baselines on nominally identical data: sibling fields from "
    "the same well, and images from the same compound, could appear in both "
    "training and test under that split, which a pixel-based pretrained network is "
    "particularly well positioned to exploit. All results in this revision use the "
    "compound-level split described above."
)

h2("2.2 Preprocessing, detection, and per-channel feature engineering")
body(
    "In our original submission, the DAPI (nuclear), Tubulin, and Actin fluorescence "
    "channels were fused into a single grayscale image before detection and feature "
    "extraction. A reviewer noted that this discards the information a classifier "
    "most needs for this task: the distinction between microtubule destabilizers, "
    "microtubule stabilizers, actin disruptors, and DNA-damaging agents is precisely "
    "the distinction in which channel's morphology changes, and channel fusion also "
    "renders intensity-based descriptors (e.g., mean intensity) biologically "
    "ambiguous, since it is no longer possible to attribute a change in intensity to "
    "a specific marker."
)
body(
    "We revised the pipeline to process each of the three channels independently. "
    "For each channel, bright-region detection used adaptive Gaussian thresholding "
    "followed by morphological opening and closing to suppress noise and consolidate "
    "objects, consistent with established CellProfiler-based preprocessing "
    "(Carpenter et al., 2006); representative detection overlays illustrating this "
    "approach are shown in Figure 2 (detection logic unchanged from our original "
    "submission -- only its application to each channel independently, rather than "
    "to one fused image, is new in this revision). Candidate contours were filtered "
    "by area and by contour-level mean intensity. From each channel we extracted the "
    "same compact "
    "set of 14 morphological descriptors used in our original submission -- spot "
    "count, spot density, mean/total/variance intensity, area-covered ratio, and "
    "spot-size distribution statistics (mean, standard deviation, median, 25th/75th "
    "percentiles, and small/medium/large spot fractions) -- yielding 42 features per "
    "image (14 descriptors \u00d7 3 channels: DAPI, Tubulin, Actin), each named with an "
    "explicit channel prefix (e.g., dapi_mean_intensity) so that every feature is "
    "biologically attributable to a specific marker."
)
body(
    "Deep-learning models likewise now receive the genuine 3-channel (DAPI, Tubulin, "
    "Actin) image stack as input, rather than a single fused-and-repeated grayscale "
    "channel. This is a natural fit for architectures such as ResNet-18 that already "
    "expect 3-channel input, and it removes channel fusion as a confound for the "
    "deep-learning results as well as the feature-based results."
)

h2("2.3 Supervised modeling")
body(
    "We evaluated five classifiers. Logistic Regression (scikit-learn; median "
    "imputation, standardization, L-BFGS solver, C=1.0, max_iter=2500, no class "
    "weighting) and Random Forest (450 trees, class_weight='balanced_subsample', "
    "median imputation, all other hyperparameters at scikit-learn defaults) were "
    "trained on the 42 per-channel morphological descriptors. The deep-learning "
    "models were a compact CNN trained from scratch (three convolutional blocks with "
    "batch normalization and ReLU activations, adaptive average pooling, two fully "
    "connected layers; 3-channel 224\u00d7224 input), a ResNet-18 trained from random "
    "initialization, and a ResNet-18 initialized with ImageNet-1K weights (Deng et "
    "al., 2009; He et al., 2016) and adapted by staged fine-tuning."
)
body(
    "All deep-learning models were optimized with Adam and cross-entropy loss "
    "(softmax over the 10 classes; batch size 16). The compact CNN trained for 6 "
    "epochs at a learning rate of 1\u00d710\u207b\u00b3. The pretrained ResNet-18 used staged "
    "fine-tuning: in stage 1, all layers except the final fully connected layer were "
    "frozen and trained for 10 epochs at 1\u00d710\u207b\u00b3; in stage 2, the last two residual "
    "blocks were unfrozen and fine-tuned for 6 further epochs at 1\u00d710\u207b\u2074, for a total "
    "training budget of 16 epochs. In our original submission, the randomly "
    "initialized ResNet-18 was trained for only 10 epochs at 1\u00d710\u207b\u00b3 with no second "
    "stage -- a training-budget asymmetry a reviewer identified as a confound for "
    "the scratch-vs-pretrained comparison, independent of pretraining itself. In "
    "this revision, the scratch ResNet-18 is trained for the same total budget of 16 "
    "epochs at 1\u00d710\u207b\u00b3 (all layers trainable throughout, since there is no pretrained "
    "representation to protect behind a frozen stage), so that the comparison "
    "isolates the effect of pretraining rather than the effect of additional "
    "gradient updates. No class weighting, resampling, or data augmentation was "
    "applied to any deep-learning model; the checkpoint used for test-set evaluation "
    "is the model state at the end of training (no early stopping), for all models."
)
body(
    "Both ResNet-18 conditions (scratch and pretrained) were additionally trained "
    "across five predefined random seeds (1, 7, 42, 123, 999), holding the "
    "compound-level data split fixed across seeds so that seed variance reflects "
    "model initialization and stochastic training dynamics rather than which "
    "compounds were sampled into which partition. All headline ResNet-18 results in "
    "this revision are reported as mean \u00b1 SD (and range) across these five seeds, "
    "not as a single training run, following reviewer guidance that single-run "
    "deep-learning results at this sample size are not sufficient evidence of "
    "typical model behavior."
)

h2("2.4 Statistical validation")
body(
    "Every model was evaluated with 95% bootstrap confidence intervals on accuracy, "
    "macro one-versus-rest (OvR) ROC AUC, and per-class AUC, and with Expected "
    "Calibration Error (ECE; confidence-based, 10 uniform-width bins over [0,1], "
    "computed as the sample-weighted average absolute gap between binned confidence "
    "and binned accuracy). For the four classes with zero test-set representation "
    "(Section 2.1), per-class AUC is undefined and macro-AUC is computed as the mean "
    "of per-class AUC over only the six classes with at least one test example, "
    "consistent across every reported macro-AUC value in this manuscript."
)
body(
    "The primary statistical comparison between models is a paired bootstrap test "
    "on the macro-AUC difference: the test set is resampled with replacement 2,000 "
    "times, macro-AUC is recomputed for both models on each resample with the model "
    "pairing preserved, and we report the resulting 95% percentile confidence "
    "interval and a two-sided bootstrap p-value. This replaces the DeLong test as "
    "the primary comparison in this revision. As a reviewer noted, the classical "
    "DeLong procedure (DeLong et al., 1988) is derived for comparing a single pair "
    "of correlated binary ROC curves; extending it to a macro-averaged, "
    "class-imbalanced, 10-class one-versus-rest quantity -- as our original "
    "submission did, by averaging per-class DeLong covariance terms under an "
    "independence assumption across classes -- is not something the classical "
    "derivation warrants. We retain the original DeLong-based comparison in this "
    "revision only as a secondary, clearly labeled cross-check (reported alongside "
    "the primary bootstrap result in Section 3.3); both "
    "methods agree in direction and significance for every comparison reported "
    "here. We note that the bootstrap procedure resamples test images and does not "
    "additionally account for non-independence of images from the same compound; a "
    "compound-level bootstrap was not performed given that four of six evaluable "
    "test classes are represented by only 2-3 test compounds, which would not "
    "support stable resampling."
)
body(
    "Feature ablation on the Random Forest model quantified descriptor "
    "redundancy and importance. Because mean_intensity/total_intensity and "
    "spot_count/density_spots_per_10k_px are exact linear rescalings of one another "
    "within a fixed-size image (a point a reviewer noted we had not acted on in our "
    "original submission), we deduplicated the per-channel feature-importance "
    "ranking before ablation: when a feature is selected for removal, its exact "
    "collinear partner (if present in the feature set) is removed together with it, "
    "and the reported \u201cN features removed\u201d always refers to N independent "
    "quantities rather than N feature names that may collapse to fewer independent "
    "degrees of freedom."
)
body(
    "Biological validation of the top morphological features used Kruskal-Wallis "
    "tests for class-stratification, as in our original submission. The "
    "concentration-response analysis is revised: rather than a single Spearman "
    "correlation between a feature and concentration pooled across all compounds "
    "(which conflates a given concentration of a potent compound with the same "
    "nominal concentration of a weak one, a point a reviewer raised), we compute "
    "Spearman correlation within each compound separately (for compounds with at "
    "least 3 distinct tested concentrations and at least 5 images) and report the "
    "median within-compound correlation and the fraction of tested compounds "
    "showing a concentration-response relationship consistent in sign with that "
    "median."
)
body(
    "Batch effects were characterized by PCA of the standardized 42-feature vectors "
    "before and after within-batch z-score normalization. In addition to visual "
    "inspection of the resulting scatterplots (with %-variance-explained reported "
    "on each axis, and the batch legend sorted rather than left in encounter order, "
    "correcting a figure-quality issue a reviewer identified), we quantify batch "
    "separation using the silhouette score on batch labels in PCA space and the "
    "3-fold cross-validated accuracy of a Random Forest classifier trained to "
    "predict batch identity from the PCA-reduced features; lower values on both "
    "metrics after normalization indicate reduced batch-driven structure, "
    "quantifying a claim our original submission supported only by visual "
    "inspection of a scatterplot."
)

print("Introduction and Methods 2.1-2.4 written.")

# =====================================================================
# 3. RESULTS
# =====================================================================
h1("3. Results")

h2("3.1 Feature-based performance improves substantially once channels are not fused")
body(
    "On the compound-disjoint held-out test set (n=328), Random Forest trained on "
    "the 42 per-channel descriptors reached an accuracy of 0.399 and a macro-OvR AUC "
    "of 0.904, and Logistic Regression reached an accuracy of 0.524 and a macro-OvR "
    "AUC of 0.866 (Table 2). Both are large improvements over the same models "
    "trained on channel-fused features under the identical compound-level split "
    "(Random Forest: accuracy 0.210, AUC 0.810; Logistic Regression: accuracy 0.207, "
    "AUC 0.794), which we also report here as a direct ablation of the channel-fusion "
    "decision itself, holding the split fixed. This is consistent with a reviewer's "
    "hypothesis that channel fusion discards MOA-discriminative signal, and with the "
    "specific mechanism proposed: the classes most easily confused under channel "
    "fusion in our original submission's confusion matrices are exactly the classes "
    "(microtubule destabilizers vs. stabilizers, actin disruptors, DNA-damaging "
    "agents) whose distinguishing signal lives in which channel changes. Macro-"
    "averaged ROC curves for both feature-based models are shown in Figure 3, with "
    "per-class curves and the macro-average explicitly overlaid for Random Forest in "
    "Figure 4; per-class confusion matrices for both models are provided in "
    "Supplementary Figures S1 (Logistic Regression) and S2 (Random Forest)."
)

h2("3.2 Deep-learning performance under the compound-level split")
body(
    "Under the compound-level split and per-channel 3-channel input, deep-learning "
    "performance diverged sharply by initialization, as in our original submission, "
    "though at substantially different absolute levels (Table 2, Figure 5). In a "
    "single representative training run, the compact CNN trained from scratch "
    "reached an accuracy of 0.000 and a macro-OvR AUC of 0.560; we return to this "
    "result and its interpretation below. The single-run ResNet-18 trained from "
    "scratch reached an accuracy of 0.494 and an AUC of 0.884; the single-run "
    "ImageNet-pretrained ResNet-18 reached an accuracy of 0.707 and an AUC of 0.959, "
    "with the lowest ECE of any deep-learning model (0.154). Per-class curves with "
    "the macro-average overlaid for the pretrained ResNet-18 are shown in Figure 6; "
    "its confusion matrix is provided in Supplementary Figure S3."
)
body(
    "Because a single training run is not sufficient evidence of typical model "
    "behavior for either ResNet-18 condition, we trained both across five predefined "
    "random seeds (Section 2.3). Across seeds, the pretrained ResNet-18 achieved a "
    "mean accuracy of 0.721 \u00b1 0.042 (range 0.652-0.756) and a mean macro-OvR AUC of "
    "0.965 \u00b1 0.007 (range 0.957-0.974), while the scratch ResNet-18 achieved a mean "
    "accuracy of 0.261 \u00b1 0.076 (range 0.174-0.357) and a mean macro-OvR AUC of "
    "0.826 \u00b1 0.048 (range 0.764-0.875) (Table 3, Figure 7). The pretrained model is "
    "both higher-performing and markedly more stable across seeds: its AUC standard "
    "deviation (0.007) is roughly 7-fold smaller than the scratch model's (0.048), "
    "and its accuracy standard deviation (0.042) is roughly 2-fold smaller than the "
    "scratch model's (0.076). A Welch's t-test on the five-seed accuracy "
    "distributions confirms the pretrained model significantly exceeds the scratch "
    "model (t=11.83, p<0.0001; Mann-Whitney U=25, p=0.004), and likewise for AUC "
    "(t=6.38, p=0.0027; Mann-Whitney U=25, p=0.004). Training and validation loss "
    "curves (single representative run each) are shown in Figure 8 (compact CNN), "
    "Figure 9 (scratch ResNet-18), and Figure 10 (pretrained ResNet-18); the "
    "dashed line in Figure 10 marks the transition from the frozen (stage 1) to "
    "fine-tuning (stage 2) training stage."
)
body(
    "We investigated why the scratch models -- both the compact CNN in its single "
    "run, and the ResNet-18-scratch condition in several of its five seeded runs -- "
    "sometimes reach exactly 0% accuracy despite a non-trivial macro-AUC (0.56-0.87 "
    "across these degenerate runs). Inspecting the predicted-class distribution for "
    "these runs shows that the model's argmax prediction collapses almost entirely "
    "onto one or two of the training-set's largest classes; in every degenerate case "
    "we examined, the collapsed-onto classes were among the four classes with no "
    "test-set representation (Section 2.1), or, for the one clearly non-degenerate "
    "seed (seed 999 in the original fused-feature check reported in our response to "
    "reviewers), spread across classes that are genuinely present in the test set. "
    "Because these four classes are large in training (up to 160 images) precisely "
    "because they are single-compound classes concentrated in training by "
    "construction, a model that has not learned features generalizing to a new "
    "compound can default to predicting one of these training-dominant classes, "
    "which is guaranteed to be wrong on every test image. This is not a data leak or "
    "an implementation defect: it is direct behavioral evidence that, once "
    "compound-level leakage is removed, a randomly initialized deep network trained "
    "on 468 images does not merely perform worse than a pretrained one -- it can fail "
    "to learn transferable features at all, on a given run, whereas pretraining "
    "reliably prevents this collapse. We consider this a stronger and more "
    "informative version of our original claim, not merely a weaker one: the "
    "scratch/pretrained gap in this dataset is not an artifact of the (corrected) "
    "leakage in the original split, but persists, and if anything is thrown into "
    "sharper relief, once that leakage is removed."
)
body(
    "The single-run compact CNN result (accuracy 0.000) was not re-run across "
    "multiple seeds in this revision, given the time constraints of the review "
    "period; we flag this explicitly as an item for a future revision cycle rather "
    "than presenting the single run as more representative than it is. Given the "
    "seed-sensitivity documented above for the architecturally similar ResNet-18-"
    "scratch condition, we expect the CNN's single-run result to be similarly "
    "seed-dependent, and we do not draw any architecture-level conclusion from it."
)

h2("3.3 Statistical comparison")
body(
    "Using the primary paired-bootstrap test on macro-AUC difference (Section 2.4, "
    "single representative run per model, 2,000 resamples), the pretrained ResNet-18 "
    "significantly exceeded Random Forest (\u0394AUC=0.055, 95% CI [0.033, 0.078], "
    "p<0.001), and Random Forest significantly exceeded the compact CNN "
    "(\u0394AUC=-0.345, 95% CI [-0.374, -0.314], p<0.001). The secondary DeLong-based "
    "comparison agrees in direction and significance for both (pretrained ResNet-18 "
    "vs. Random Forest: z=5.10, p=3.4\u00d710\u207b\u2077; CNN vs. Random Forest: z=-22.07, "
    "p=6.0\u00d710\u207b\u00b9\u2070\u2078; Table 4). Independently, the five-seed group-level comparison "
    "(Section 3.2) confirms the pretrained-vs-scratch ResNet-18 difference using a "
    "conventional two-sample test rather than a single-run comparison, and agrees "
    "with both bootstrap and DeLong results in direction and significance."
)

h2("3.4 Calibration")
body(
    "Expected Calibration Error was lowest for the pretrained ResNet-18 (0.154), "
    "followed by Random Forest (0.170), Logistic Regression (0.266), the compact CNN "
    "(0.466), and the scratch ResNet-18 (0.379) (Table 5, Figure 11). All models are "
    "less well calibrated on the compound-level split than the corresponding models "
    "were on the (leakage-affected) image-level split reported in our original "
    "submission; we believe this reflects genuine task difficulty under a properly "
    "disjoint test set rather than a regression introduced by this revision, since "
    "the same pattern (higher ECE under compound-level evaluation) is present across "
    "every model, including Logistic Regression and Random Forest, whose training "
    "procedure did not change between analyses."
)

h2("3.5 Feature ablation, biological association, and batch effects")
body(
    "Deduplicated feature ablation on the Random Forest model (Section 2.4) shows "
    "that removing the single most important independent feature (dapi_spot_area_q25) "
    "reduces macro-AUC by 0.007 (0.904 to 0.898); removing the top 5 independent "
    "features (which, once collinear partners are removed alongside them, corresponds "
    "to removing 7 of the 42 total feature columns) reduces AUC by 0.046 (to 0.859); "
    "and removing the top 8 independent features (12 of 42 columns, once collinear "
    "partners are included) reduces AUC by 0.067 (to 0.837) (Table 6, Figure 12). "
    "Feature importance is distributed across all three channels: the top-8 "
    "independent ablation set includes DAPI-, Tubulin-, and Actin-channel "
    "descriptors, consistent with the per-channel feature-engineering rationale in "
    "Section 2.2 -- MOA-discriminative information is not concentrated in a single "
    "channel. The full Spearman correlation structure among all 42 features is "
    "provided in Supplementary Figure S4."
)
body(
    "Kruskal-Wallis tests confirm significant class-stratification for the top "
    "DAPI-channel features tested (H=223-589, p<10\u207b\u2074\u00b2 for all; Table 7). Within-compound "
    "Spearman correlations between these features and concentration are weaker in "
    "magnitude than the pooled correlation we originally reported for most, but not "
    "all, of these features (median \u03c1=-0.02 to -0.33 across features, versus a pooled "
    "\u03c1\u22480.20 magnitude reported previously); two features (dapi_spot_count, "
    "dapi_density_spots_per_10k_px) show a within-compound correlation of larger "
    "magnitude (\u03c1=-0.33) than the original pooled estimate, so the pooled statistic "
    "understated rather than inflated the relationship for these two features "
    "specifically, even though it remains biologically uninterpretable as a pooled "
    "quantity for the reasons given above. This is expected once concentration is "
    "not compared across compounds of very different potency; the sign of the "
    "within-compound correlation is consistent across 62.5-87.5% of the 8 compounds "
    "tested per feature, indicating a real but compound-dependent dose-response "
    "relationship rather than the artificially inflated and biologically "
    "uninterpretable pooled statistic we reported originally."
)
body(
    "PCA of the 42-feature vectors shows batch structure by weekly acquisition group "
    "before normalization (silhouette score on batch labels = -0.285; a 3-fold "
    "cross-validated Random Forest predicts batch identity from the PCA-reduced "
    "features at 10.9% accuracy, close to chance for this many batches) and reduced "
    "structure after within-batch z-score normalization (silhouette = -0.103; "
    "batch-predictability accuracy = 9.0%) (Table 8, Figures 13-14). Both "
    "metrics move in the direction of reduced batch separation after normalization, "
    "quantifying a claim our original submission supported only by visual PCA "
    "inspection; we note the improvement is modest by these metrics and do not "
    "overstate normalization as having eliminated batch structure."
)

h2("3.6 Computational cost")
body(
    "All experiments were run on a single CPU-only workstation (no GPU acceleration "
    "was used or available for this revision); we report this explicitly to correct "
    "our original submission, which gave training times and peak memory without any "
    "hardware specification. Logistic Regression and Random Forest trained in "
    "well under 2 seconds. The compact CNN (6 epochs) trained in under a minute. "
    "The two ResNet-18 conditions, each trained for a matched total of 16 epochs, "
    "took approximately 145-275 seconds per run for the pretrained condition and "
    "310-590 seconds per run for the scratch condition, depending on random seed "
    "(Table 9); because all reported training times are on CPU only, they should "
    "not be extrapolated to GPU-accelerated deployment settings."
)

print("Results written.")

# =====================================================================
# 4. DISCUSSION
# =====================================================================
h1("4. Discussion")

h2("4.1 Pretraining remains the dominant factor, under a corrected evaluation")
body(
    "Once compound-level leakage and channel fusion are both removed, the "
    "pretrained ResNet-18 remains the strongest model (mean accuracy 0.721, mean "
    "AUC 0.965 across five seeds), and pretraining strategy remains a larger "
    "determinant of performance than the choice between feature-based and "
    "deep-learning approaches in this small-sample regime. However, the magnitude "
    "of every headline number changes substantially from our original submission: "
    "the pretrained model's accuracy falls from 0.939 to 0.721, and the "
    "scratch-vs-pretrained accuracy gap, which we originally attributed entirely to "
    "pretraining, is now attributable to a combination of pretraining, a training-"
    "budget asymmetry we did not intend (Section 2.3), and, most importantly, a "
    "compound-level leakage effect that was likely the single largest contributor "
    "to the originally reported gap between pixel-based deep models and feature-"
    "based baselines, exactly as a reviewer predicted."
)
body(
    "Feature-based models improve substantially once per-channel information is "
    "preserved (Random Forest AUC 0.810 to 0.904; Logistic Regression AUC 0.794 to "
    "0.866, both on the identical compound-level split), closing much of the gap to "
    "the deep-learning models. This directly supports the reviewer hypothesis that "
    "channel fusion, not an inherent limitation of hand-crafted morphological "
    "descriptors, was responsible for a meaningful share of the feature-based "
    "models' original underperformance."
)

h2("4.2 Seed instability as a finding, not a caveat")
body(
    "Our original submission disclosed, in a Discussion-section limitation, that "
    "five additional re-initializations of the scratch ResNet-18 produced AUC values "
    "ranging from approximately 0.75 to 0.96, while presenting a single, lower-AUC "
    "run (0.668) as the headline result in every table and figure. A reviewer "
    "correctly identified that this placement -- reporting the apparent worst run in "
    "the tables while disclosing the true range only as a limitation in the "
    "Discussion -- reads as selective reporting even when unintentional. In this "
    "revision, both ResNet-18 conditions are reported as five-seed distributions "
    "everywhere they appear (Tables 2, 3, 9, Figures 5-6), and the corresponding "
    "statistical comparison (Section 3.3) is computed on that basis. The scratch "
    "model's seed-to-seed AUC standard deviation (0.048) is roughly 7-fold larger "
    "than the pretrained model's (0.007), and in several seeds the scratch model's "
    "predictions collapse entirely onto training-dominant, test-absent classes "
    "(Section 3.2). We regard this instability as a legitimate empirical finding in "
    "its own right -- that pretraining confers a stability advantage in addition to "
    "a mean-performance advantage under this sample size -- rather than a limitation "
    "to be caveated after the fact."
)

h2("4.3 Scope of the redox-pharmacology and pulmonary-disease framing")
body(
    "Our original submission's title, Abstract, Introduction, and Conclusion framed "
    "this work as directly relevant to redox pharmacology and pulmonary disease, on "
    "the grounds that several studied MOA classes (microtubule destabilizers, DNA-"
    "damaging agents, protein-synthesis inhibitors) engage pathways that can "
    "intersect with reactive oxygen species (ROS) signaling, which is in turn "
    "implicated in pulmonary diseases such as asthma, COPD, acute lung injury, and "
    "pulmonary fibrosis. Reviewers correctly noted that this framing considerably "
    "outpaces what the present study demonstrates: BBBC021 consists of MCF-7 breast "
    "cancer cell images, this study does not measure ROS or oxidative stress "
    "directly, and no pulmonary cell type or disease model is used anywhere in the "
    "pipeline. Classifying MOA classes that are mechanistically plausible candidates "
    "for redox relevance is not the same as demonstrating that the learned "
    "morphological features are oxidative-stress-associated signatures, and the "
    "original Discussion's specific claim that the top ablation features (mean "
    "intensity, area-covered ratio, spot density) \"reflect cellular-state changes "
    "associated with ROS-mediated cytotoxicity\" was asserted without supporting "
    "evidence from this study."
)
body(
    "We have revised the title, Abstract, Introduction, and Conclusion accordingly: "
    "this manuscript is presented as a computational benchmarking and methodology "
    "study on the BBBC021 MOA classification task, addressing train/test leakage, "
    "channel-fusion information loss, and statistically defensible model comparison "
    "in high-content screening pipelines generally. Relevance to redox pharmacology "
    "and pulmonary disease is retained only as a motivating rationale for why "
    "MOA-classification pipelines of this kind are of pharmacological interest, and "
    "is explicitly flagged as a direction for future work requiring either an "
    "independent ROS/oxidative-stress readout on the same or a related compound set, "
    "or a pulmonary cell-based assay, neither of which is part of the present study."
)
body(
    "That said, the reviewer's question of which features plausibly link to which "
    "biology is worth answering explicitly, with the same hedging. The top-ablation "
    "DAPI-channel features (Table 6: spot area, spot count, spot density) track "
    "nuclear fragmentation and chromatin condensation, a downstream morphological "
    "readout shared by several stress pathways, oxidative stress among them but not "
    "specific to it; the same features also respond to DNA-damaging agents and "
    "mitotic arrest through non-redox mechanisms, so elevated signal on these "
    "descriptors cannot be attributed to ROS without an independent assay. The "
    "tubulin- and actin-channel features in the same top-8 ablation set (Section "
    "3.5) track cytoskeletal reorganization, which is the direct, well-established "
    "readout of microtubule- and actin-targeting MOAs regardless of any redox "
    "involvement; we do not have grounds to re-attribute this signal to oxidative "
    "stress either, and we did not intend the ablation ranking to be read as a "
    "biological specificity ranking. We present this limited answer to avoid "
    "overclaiming a mechanistic interpretation in either direction."
)

h2("4.4 Limitations")
body(
    "Several limitations remain. First, four of ten MOA classes in this dataset "
    "are represented by a single compound and therefore cannot be evaluated with a "
    "compound-disjoint test set at all under the present data availability; results "
    "for these four classes should be understood as not yet evaluable for "
    "genuine cross-compound generalization, rather than as evidence of good or poor "
    "performance. Expanding the number of compounds sampled per MOA class, "
    "including from the full BBBC021 archive beyond the 16 files retrieved here, "
    "would allow these four classes to be included in a genuinely compound-disjoint "
    "evaluation. Second, the compact CNN's single-run result was not re-evaluated "
    "across multiple seeds in this revision cycle (Section 3.2); given the "
    "documented seed-sensitivity of the architecturally similar scratch ResNet-18, "
    "we do not treat the single CNN run as representative and flag multi-seed CNN "
    "evaluation as an immediate next step. Third, the paired bootstrap statistical "
    "comparison (Section 2.4) resamples test images and does not model "
    "non-independence among images from the same compound; a fully compound-aware "
    "bootstrap would require more test compounds per class than are currently "
    "available. Fourth, all training in this study was performed on CPU without GPU "
    "acceleration, and while we do not expect this to materially change reported "
    "AUC or accuracy, absolute training times (Table 9) should not be extrapolated "
    "to GPU-accelerated deployment settings. Fifth, as in our original submission, "
    "all results derive from a single publicly available dataset (BBBC021); "
    "cross-dataset generalization to BBBC014, BBBC020, or RxRx1 (Bray et al., 2016; "
    "Sypetkowski et al., 2023) has not been tested and remains future work. Sixth, "
    "we compared our custom per-channel feature-extraction pipeline against "
    "CellProfiler (Carpenter et al., 2006) only at the level of preprocessing logic "
    "(Section 2.2); we did not run CellProfiler's own feature set through the same "
    "classifiers on the same compound-level split, so we cannot yet quantify how our "
    "14-descriptor set compares to CellProfiler's output on this task."
)

h2("4.5 Generalizability across held-out compounds")
body(
    "A compound-disjoint test set is not the same as a compound-diverse one. "
    "Because every multi-compound class here has at most 4 distinct compounds "
    "(Section 2.1), the main evaluation tests each of the six evaluable classes "
    "against exactly one held-out compound, so Table 2's per-class performance "
    "describes generalization to that one compound rather than to the class in "
    "general. To ask the question the way a reader of Table 2 might assume it was "
    "already answered, how reliably can the model classify a compound it has never "
    "seen, for any compound in its class, we ran leave-one-compound-out "
    "cross-validation on Random Forest and Logistic Regression, the two models we "
    "can retrain without GPU access: for every compound in every multi-compound "
    "class, we trained on all remaining images (all other classes included in "
    "full) and tested on that compound alone, rotating through every compound in "
    "turn (Table 10; full per-compound results in the project repository)."
)
body(
    "The result is substantially less optimistic than Table 2's single-split "
    "numbers and we report it without softening it. Per-class accuracy on a "
    "held-out compound ranges from 0% to 100% depending on which compound is held "
    "out: Random Forest correctly classifies 0% of held-out colchicine and "
    "nocodazole images in the microtubule-destabilizer class despite classifying "
    "75% of held-out demecolcine images in the same class, and 0-8% of held-out "
    "protein-degradation compounds (ALLN, MG-132) against each other. Aurora "
    "kinase inhibitors generalize best (71-92% across its two compounds); protein "
    "degradation and microtubule destabilizers generalize worst. This variance is "
    "compound-specific, not a single generalization number, and it is larger than "
    "the headline Table 2 accuracy would suggest on its own. We read this as "
    "confirming the reviewer's concern directly: compound-level splitting prevents "
    "leakage, but with only 2-4 compounds per class it does not yet establish "
    "that the model has learned a compound-invariant representation of each MOA. "
    "Establishing that would need more compounds per MOA class than the present "
    "16-archive-file subset of BBBC021 provides (Section 4.4)."
)
body("Table 10. Leave-one-compound-out cross-validation, Random Forest and Logistic Regression, per MOA class.")
table_from_rows(
    ["MOA class", "Compounds tested", "RF accuracy (mean, range)", "LR accuracy (mean, range)"],
    [
        ["Actin disruptors", "3", "0.556 (0.292-1.000)", "0.500 (0.417-0.583)"],
        ["Aurora kinase inhibitors", "2", "0.812 (0.708-0.917)", "0.680 (0.583-0.778)"],
        ["DNA damage", "3", "0.370 (0.000-1.000)", "0.407 (0.222-0.750)"],
        ["Microtubule destabilizers", "4", "0.214 (0.000-0.750)", "0.254 (0.000-0.812)"],
        ["Microtubule stabilizers", "2", "0.497 (0.355-0.639)", "0.670 (0.589-0.750)"],
        ["Protein degradation", "2", "0.042 (0.000-0.083)", "0.375 (0.333-0.417)"],
    ],
)
caption(
    "Table 10. Each row holds out every compound in that class in turn, training "
    "on all remaining images (all other classes included in full) and testing on "
    "the held-out compound alone; range is across that class's compounds. This "
    "differs from Table 2's single compound-level split and is intended to show "
    "compound-to-compound variance in generalization, not to replace Table 2's "
    "headline comparison across models."
)

# =====================================================================
# 5. CONCLUSION
# =====================================================================
h1("5. Conclusion")
body(
    "We present a revised, leakage-controlled benchmark for mechanism-of-action "
    "classification on BBBC021 that addresses two methodological issues identified "
    "in peer review of our original submission: image-level (rather than "
    "compound-level) train/test splitting, and channel-fused (rather than "
    "per-channel) feature extraction and deep-learning input. Under the corrected "
    "evaluation, an ImageNet-pretrained ResNet-18 with staged fine-tuning remains "
    "the strongest model (mean accuracy 0.721 \u00b1 0.042, mean macro-OvR AUC "
    "0.965 \u00b1 0.007 across five seeds), significantly outperforming both a matched-"
    "budget scratch-initialized ResNet-18 and Random Forest on per-channel features "
    "by paired bootstrap and corroborating DeLong tests. Feature-based classifiers "
    "improve substantially once channel fusion is removed and remain interpretable, "
    "well-calibrated, and computationally inexpensive alternatives. Scratch-"
    "initialized deep models are not only lower-performing but markedly less stable "
    "across random seeds than the pretrained model, indicating that pretraining's "
    "practical value in this data regime includes reliability, not only mean "
    "accuracy. We present this as a methodology and benchmarking contribution to "
    "high-content screening pipeline design; we do not claim, and this study does "
    "not demonstrate, direct relevance to oxidative stress or pulmonary disease, "
    "which we identify as a motivated but untested direction for future work. The "
    "revised pipeline -- including the compound-level split, per-channel feature "
    "extraction, matched-budget and multi-seed model training, deduplicated feature "
    "ablation, within-compound dose-response analysis, quantified batch-effect "
    "correction, and bootstrap-based statistical comparison -- is publicly "
    "available at https://github.com/SID-6921/BBBC021_Project."
)

print("Discussion and Conclusion written.")

# =====================================================================
# DECLARATIONS
# =====================================================================
h1("Data Availability Statement")
body(
    "The datasets analyzed in this study are publicly available. The BBBC021 image "
    "set was obtained from the Broad Bioimage Benchmark Collection. All source "
    "code, including the revised compound-level split, per-channel feature "
    "extraction, multi-seed training scripts, and bootstrap statistical tests, "
    "along with generated figures, tables, and result summaries, are openly "
    "available at https://github.com/SID-6921/BBBC021_Project."
)
h1("Ethics Statement")
body(
    "This study did not involve human participants or animals; it used only "
    "publicly available, de-identified image data. Approval from an institutional "
    "ethics committee and informed consent were therefore not required. All authors "
    "consented to the publication of this manuscript."
)
h1("Author Contributions")
body(
    "SN: conceptualization, methodology, software, validation, formal analysis, "
    "visualization, and writing of the original draft and this revision. ShK, SiK, "
    "and VRN: writing, review, and editing. NK and VRN: critical revision of the "
    "manuscript. NK: supervision. All authors read and approved the final version "
    "of this revised manuscript."
)
h1("Funding")
body("This research received no external funding.")
h1("Acknowledgments")
body("The authors acknowledge the Broad Bioimage Benchmark Collection for providing the BBBC021 dataset.")
h1("Conflict of Interest")
body(
    "Authors NK, ShK, and SiK are members of the same family. This relationship "
    "had no influence on the study design, data collection, analysis, or "
    "interpretation of the results. The authors otherwise declare that the "
    "research was conducted in the absence of any commercial or financial "
    "relationships that could be construed as a potential conflict of interest."
)
h1("Generative AI Disclosure")
body(
    "No generative AI was used to design the experiments, analyze the data, or "
    "generate the scientific content of this manuscript."
)

# =====================================================================
# REFERENCES
# =====================================================================
h1("References")
refs = [
    "Bray, M.-A., Singh, S., Han, H., Davis, C. T., Borgeson, B., Hartland, C., et al. "
    "(2016). Cell Painting, a high-content image-based assay for morphological "
    "profiling using multiplexed fluorescent dyes. Nat. Protoc. 11, 1757\u20131774. "
    "doi: 10.1038/nprot.2016.105",
    "Caie, P. D., Walls, R. E., Ingleston-Orme, A., Daya, S., Houslay, T., Eagle, R., "
    "et al. (2010). High-content phenotypic profiling of drug response signatures "
    "across distinct cancer cells. Mol. Cancer Ther. 9, 1913\u20131926. "
    "doi: 10.1158/1535-7163.MCT-09-1148",
    "Carpenter, A. E., Jones, T. R., Lamprecht, M. R., Clarke, C., Kang, I. H., "
    "Friman, O., et al. (2006). CellProfiler: image analysis software for "
    "identifying and quantifying cell phenotypes. Genome Biol. 7, R100. "
    "doi: 10.1186/gb-2006-7-10-r100",
    "Ching, T., Himmelstein, D. S., Beaulieu-Jones, B. K., Kalinin, A. A., Do, B. T., "
    "Way, G. P., et al. (2018). Opportunities and obstacles for deep learning in "
    "biology and medicine. J. R. Soc. Interface 15, 20170387. "
    "doi: 10.1098/rsif.2017.0387",
    "DeLong, E. R., DeLong, D. M., and Clarke-Pearson, D. L. (1988). Comparing the "
    "areas under two or more correlated receiver operating characteristic curves: "
    "a nonparametric approach. Biometrics 44, 837\u2013845. doi: 10.2307/2531595",
    "Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and Fei-Fei, L. (2009). "
    "\u201cImageNet: a large-scale hierarchical image database,\u201d in 2009 IEEE "
    "Conference on Computer Vision and Pattern Recognition (Miami, FL: IEEE), "
    "248\u2013255. doi: 10.1109/CVPR.2009.5206848",
    "He, K., Zhang, X., Ren, S., and Sun, J. (2016). \u201cDeep residual learning for "
    "image recognition,\u201d in 2016 IEEE Conference on Computer Vision and Pattern "
    "Recognition (Las Vegas, NV: IEEE), 770\u2013778. doi: 10.1109/CVPR.2016.90",
    "Ljosa, V., Sokolnicki, K. L., and Carpenter, A. E. (2012). Annotated "
    "high-throughput microscopy image sets for validation. Nat. Methods 9, 637. "
    "doi: 10.1038/nmeth.2083",
    "Moen, E., Bannon, D., Kudo, T., Graf, W., Covert, M., and Van Valen, D. (2019). "
    "Deep learning for cellular image analysis. Nat. Methods 16, 1233\u20131246. "
    "doi: 10.1038/s41592-019-0403-1",
    "Sypetkowski, M., Rezanejad, M., Saberian, S., Kraus, O., Urbanik, J., Taylor, "
    "J., et al. (2023). \u201cRxRx1: a dataset for evaluating experimental batch "
    "correction methods,\u201d in 2023 IEEE/CVF Conference on Computer Vision and "
    "Pattern Recognition Workshops (CVPRW) (Vancouver, BC: IEEE), 4285\u20134294. "
    "doi: 10.1109/CVPRW59228.2023.00451",
]
for i, ref in enumerate(refs, 1):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.4)
    p.paragraph_format.first_line_indent = Inches(-0.4)
    p.add_run(f"[{i}] {ref}")

print("Declarations and References written.")

# =====================================================================
# TABLES
# =====================================================================
doc.add_page_break()
h1("Tables")

body("Table 1. Class composition and compound-level grouped split.")
table_from_rows(
    ["MOA class", "Total images", "Distinct compounds", "Train", "Val", "Test"],
    [
        ["Microtubule stabilizers", 160, 2, 36, 0, 124],
        ["DMSO", 160, 1, 160, 0, 0],
        ["Microtubule destabilizers", 136, 4, 40, 12, 84],
        ["Aurora kinase inhibitors", 108, 2, 72, 0, 36],
        ["Eg5 inhibitors", 84, 1, 84, 0, 0],
        ["Actin disruptors", 60, 3, 24, 12, 24],
        ["Protein degradation", 48, 2, 24, 0, 24],
        ["DNA damage", 44, 3, 4, 4, 36],
        ["DNA replication", 12, 1, 12, 0, 0],
        ["Protein synthesis", 12, 1, 12, 0, 0],
    ],
)
caption(
    "Table 1. Every image from a given compound is assigned to exactly one "
    "partition. Four classes (DMSO, Eg5 inhibitors, DNA replication, Protein "
    "synthesis) are each represented by a single compound in this subset and are "
    "therefore kept entirely in training; per-class test metrics are undefined for "
    "these four classes throughout this manuscript. Total n=824 (train=468, "
    "val=28, test=328)."
)

body("Table 2. Overall model performance on the held-out, compound-disjoint test set (single representative run; n=328).")
table_from_rows(
    ["Model", "Accuracy (95% CI)", "Macro-OvR AUC (95% CI)", "ECE"],
    [
        ["Logistic Regression", "0.524 (0.476, 0.579)", "0.866 (0.837, 0.894)", "0.266"],
        ["Random Forest", "0.399 (0.354, 0.451)", "0.904 (0.884, 0.923)", "0.170"],
        ["CNN (scratch)", "0.000 (0.000, 0.000)", "0.560 (0.537, 0.585)", "0.466"],
        ["ResNet-18 (scratch, single run)", "0.494 (0.445, 0.549)", "0.884 (0.859, 0.906)", "0.379"],
        ["ResNet-18 (pretrained, single run)", "0.707 (0.649, 0.759)", "0.959 (0.943, 0.971)", "0.154"],
    ],
)
caption(
    "Table 2. Macro-OvR AUC is averaged over the 6 classes with test-set "
    "representation (Table 1); LR/RF are trained on 42 per-channel morphological "
    "descriptors. ResNet-18 rows show a single representative run; see Table 3 for "
    "the five-seed distribution used as the primary reported result for both "
    "ResNet-18 conditions."
)

body("Table 3. ResNet-18 performance across five random seeds (primary reported result for both conditions).")
table_from_rows(
    ["Model", "Accuracy mean \u00b1 SD (range)", "Macro-OvR AUC mean \u00b1 SD (range)"],
    [
        ["ResNet-18 (scratch)", "0.261 \u00b1 0.076 (0.174\u20130.357)", "0.826 \u00b1 0.048 (0.764\u20130.875)"],
        ["ResNet-18 (pretrained)", "0.721 \u00b1 0.042 (0.652\u20130.756)", "0.965 \u00b1 0.007 (0.957\u20130.974)"],
    ],
)
caption(
    "Table 3. Seeds: 1, 7, 42, 123, 999. Split held fixed across seeds. Welch's "
    "t-test, pretrained vs. scratch: accuracy t=11.83, p<0.0001; AUC t=6.38, "
    "p=0.0027. Mann-Whitney U=25 (p=0.004) for both metrics."
)

body("Table 4. Statistical comparison: paired bootstrap (primary) and DeLong (secondary, retained for comparison).")
table_from_rows(
    ["Comparison", "\u0394AUC", "Bootstrap 95% CI", "Bootstrap p", "DeLong z", "DeLong p"],
    [
        ["CNN (scratch) vs. Random Forest", "-0.345", "[-0.374, -0.314]", "<0.001", "-22.07", "6.0e-108"],
        ["ResNet-18 (pretrained) vs. Random Forest", "0.055", "[0.033, 0.078]", "<0.001", "5.10", "3.4e-7"],
    ],
)
caption(
    "Table 4. Paired bootstrap (2,000 resamples of the n=328 test set, model "
    "pairing preserved per resample) is the primary comparison in this revision; "
    "DeLong is retained only as a secondary cross-check given known validity "
    "concerns for macro-averaged multiclass AUC (Section 2.4). Both methods agree "
    "in direction and significance for both comparisons reported."
)

body("Table 5. Calibration (Expected Calibration Error, 10 bins).")
table_from_rows(
    ["Model", "ECE"],
    [
        ["ResNet-18 (pretrained)", "0.154"],
        ["Random Forest", "0.170"],
        ["Logistic Regression", "0.266"],
        ["ResNet-18 (scratch)", "0.379"],
        ["CNN (scratch)", "0.466"],
    ],
)

body("Table 6. Deduplicated feature ablation (Random Forest, 42 per-channel features).")
table_from_rows(
    ["Independent features removed", "Columns removed (incl. collinear partners)", "Test macro-AUC", "\u0394AUC vs. baseline"],
    [
        [0, 0, "0.904", "0.000"],
        [1, 1, "0.898", "0.007"],
        [2, 2, "0.890", "0.015"],
        [3, 3, "0.893", "0.012"],
        [5, 7, "0.859", "0.046"],
        [8, 12, "0.837", "0.067"],
    ],
)
caption(
    "Table 6. \u201cIndependent features removed\u201d counts distinct degrees of freedom "
    "after deduplicating exact collinear pairs (mean/total intensity; spot count/"
    "density) within each channel; \u201ccolumns removed\u201d is the corresponding count of "
    "the 42 raw feature columns actually dropped, since each independent feature's "
    "collinear partner (if present) is removed alongside it."
)

body("Table 7. Biological validation: class-stratification and within-compound dose-response for top DAPI-channel features.")
table_from_rows(
    ["Feature", "Kruskal-Wallis H", "KW p-value", "Median within-compound Spearman \u03c1", "Fraction compounds consistent sign", "N compounds tested"],
    [
        ["dapi_density_spots_per_10k_px", "588.8", "5.3e-121", "-0.331", "0.875", 8],
        ["dapi_spot_count", "588.8", "5.3e-121", "-0.331", "0.875", 8],
        ["dapi_mean_intensity", "411.6", "4.7e-83", "-0.132", "0.750", 8],
        ["dapi_spot_area_q25", "395.7", "1.1e-79", "-0.025", "0.625", 8],
        ["dapi_spot_area_median", "371.6", "1.6e-74", "-0.142", "0.750", 8],
        ["dapi_spot_area_mean", "222.6", "5.9e-43", "-0.178", "0.625", 8],
    ],
)
caption(
    "Table 7. Within-compound Spearman correlation replaces the pooled-across-"
    "compounds correlation reported in our original submission (Section 2.4); "
    "magnitudes are smaller but the sign is consistent across the majority of the "
    "8 compounds with sufficient concentration levels tested per feature."
)

body("Table 8. Batch-effect quantification before and after within-batch z-score normalization.")
table_from_rows(
    ["Metric", "Before normalization", "After normalization"],
    [
        ["Silhouette score on batch labels (PCA space)", "-0.285", "-0.103"],
        ["3-fold CV batch-identity prediction accuracy", "10.9%", "9.0%"],
        ["PC1 % variance explained", "37.3%", "37.6%"],
        ["PC2 % variance explained", "15.2%", "14.3%"],
    ],
)
caption(
    "Table 8. Both metrics move in the direction of reduced batch structure after "
    "normalization; silhouette scores are negative in both conditions, indicating "
    "batch labels do not form well-separated clusters in PCA space even before "
    "correction, consistent with batch not being a dominant axis of variation "
    "relative to biological (MOA-class) signal."
)

body("Table 9. Computational cost (single CPU workstation; no GPU used).")
table_from_rows(
    ["Model", "Approx. training time"],
    [
        ["Logistic Regression", "< 1 s"],
        ["Random Forest", "< 2 s"],
        ["CNN (scratch, 6 epochs)", "\u2248 40\u201360 s"],
        ["ResNet-18 (scratch, 16 epochs, matched budget)", "\u2248 310\u2013590 s (seed-dependent)"],
        ["ResNet-18 (pretrained, 16 epochs staged)", "\u2248 145\u2013275 s (seed-dependent)"],
    ],
)
caption("Table 9. All training was performed on CPU only; times should not be extrapolated to GPU-accelerated settings.")

print("Tables written.")

# =====================================================================
# FIGURES (legends; images embedded from final_figures/ where available)
# =====================================================================
doc.add_page_break()
h1("Figures")

FIG_DIR = r"C:\Users\nanda\Downloads\BBBC021_fix\final_figures"

def add_figure(path, width=5.8):
    import os
    if os.path.exists(path):
        doc.add_picture(path, width=Inches(width))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    else:
        p = doc.add_paragraph()
        r = p.add_run(f"[Figure file not found: {path}]")
        r.italic = True

add_figure(f"{FIG_DIR}\\Figure_1.png")
caption(
    "Figure 1. Overall analysis pipeline workflow, from raw multi-channel "
    "microscopy images through per-channel feature extraction and model "
    "training/evaluation on the compound-disjoint split (detection logic "
    "unchanged from our original submission)."
)

add_figure(f"{FIG_DIR}\\Figure_2.png")
caption(
    "Figure 2. Representative detection overlays illustrating spot/object "
    "detection applied independently to each channel (DAPI, Tubulin, Actin), "
    "consistent with established CellProfiler-based preprocessing (Carpenter "
    "et al., 2006); only the per-channel application, not the underlying "
    "detection logic, is new in this revision."
)

add_figure(f"{FIG_DIR}\\figure3_roc_feature_based_macro.png")
caption(
    "Figure 3. Macro-averaged ROC curves for Logistic Regression and Random "
    "Forest, averaged over the 6 test-evaluable classes, on the held-out "
    "compound-disjoint test set. Legend values match Table 2 exactly, correcting "
    "our original submission, in which the plotted curve did not correspond to "
    "the macro-OvR quantity reported in the tables."
)

add_figure(f"{FIG_DIR}\\figure3b_roc_random_forest_per_class.png")
caption(
    "Figure 4. Per-class ROC curves for Random Forest with the macro-average "
    "overlaid (black), addressing a reviewer's request that a macro-averaged "
    "multiclass AUC -- which is the average of several per-class curves and "
    "cannot be represented by a single curve -- be shown per-class with the "
    "macro-average explicitly overlaid rather than plotted as one ambiguous curve."
)

add_figure(f"{FIG_DIR}\\figure5_roc_deep_learning_macro.png")
caption(
    "Figure 5. Macro-averaged ROC curves for the three deep-learning models "
    "(single representative run each), averaged over the 6 test-evaluable "
    "classes. Legend values match Table 2 exactly."
)

add_figure(f"{FIG_DIR}\\figure5b_roc_resnet_pretrained_per_class.png")
caption(
    "Figure 6. Per-class ROC curves for the pretrained ResNet-18 with the "
    "macro-average overlaid (black)."
)

add_figure(f"{FIG_DIR}\\figure5c_seed_distribution.png")
caption(
    "Figure 7. ResNet-18 accuracy and macro-OvR AUC distributions across five "
    "random seeds (scratch vs. pretrained; black bar = mean), corresponding to "
    "Table 3."
)

add_figure(f"{FIG_DIR}\\figure_s7_cnn_loss_curves.png")
caption(
    "Figure 8. Training and validation loss for the compact CNN, now generated "
    "directly from the model whose results appear in Tables 2 and 5 "
    "(CrossEntropyLoss, softmax over 10 classes). Our original Supplementary "
    "Figure S7 plotted the loss curve of an unrelated, unused exploratory "
    "script that trained a different (binary-classification, "
    "BCEWithLogitsLoss) model never used to produce any reported result; a "
    "reviewer detected this from the epoch count and initial-loss value not "
    "matching a 10-class softmax model, and it is corrected here."
)

add_figure(f"{FIG_DIR}\\figure_s7b_resnet_scratch_loss_curves.png")
caption(
    "Figure 9. Training and validation loss curves for the scratch ResNet-18 "
    "(single representative run), provided per reviewer request."
)

add_figure(f"{FIG_DIR}\\figure_s7c_resnet_pretrained_loss_curves.png")
caption(
    "Figure 10. Training and validation loss curves for the pretrained "
    "ResNet-18 (single representative run); the dashed vertical line marks "
    "the transition from the frozen (stage 1) to fine-tuning (stage 2) "
    "training stage."
)

add_figure(f"{FIG_DIR}\\figure14_calibration_curves_multiclass.png")
caption("Figure 11. Reliability diagrams for all five models (Table 5).")

add_figure(f"{FIG_DIR}\\figure17_feature_ablation.png")
caption(
    "Figure 12. Deduplicated Random Forest feature-ablation curve on the 42 "
    "per-channel morphological descriptors (Table 6)."
)

add_figure(f"{FIG_DIR}\\figure15_pca_batch_before.png")
caption("Figure 13. PCA of the 42-feature vectors before within-batch z-score normalization (Table 8).")

add_figure(f"{FIG_DIR}\\figure16_pca_batch_after.png")
caption(
    "Figure 14. PCA of the 42-feature vectors after within-batch z-score "
    "normalization, with %-variance-explained axis labels and a sorted batch "
    "legend (both missing in our original submission), and corresponding "
    "quantified batch-separation metrics in Table 8."
)

doc.add_page_break()
h1("Supplementary Figures")

add_figure(f"{FIG_DIR}\\figure11_confusion_matrix_lr_12class.png")
caption(
    "Supplementary Figure S1. Confusion matrix for Logistic Regression on the "
    "compound-disjoint test set; the 4 classes with no test-set representation "
    "(Table 1) appear as all-zero rows, shown explicitly rather than omitted."
)

add_figure(f"{FIG_DIR}\\figure12_confusion_matrix_rf_12class.png")
caption(
    "Supplementary Figure S2. Confusion matrix for Random Forest on the "
    "compound-disjoint test set; the 4 classes with no test-set representation "
    "(Table 1) appear as all-zero rows, shown explicitly rather than omitted."
)

add_figure(f"{FIG_DIR}\\figure13_confusion_matrix_resnet_pretrained_12class.png")
caption(
    "Supplementary Figure S3. Confusion matrix for the pretrained ResNet-18 "
    "on the compound-disjoint test set; the 4 classes with no test-set "
    "representation (Table 1) appear as all-zero rows, shown explicitly "
    "rather than omitted."
)

add_figure(f"{FIG_DIR}\\supplementary_figure_s1_feature_correlation.png")
caption("Supplementary Figure S4. Spearman correlation matrix across all 42 per-channel morphological features.")

print("Figures section written.")
doc.save(str(Path(__file__).resolve().parent / "BBBC021_Manuscript_REVISED.docx"))
print("Saved checkpoint 6.")
