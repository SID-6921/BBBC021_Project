# -*- coding: utf-8 -*-
"""Point-by-point response to Reviewer 1 and Reviewer 2, MS 1926843."""
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

def h2(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(11)

def comment(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run("Reviewer comment: ")
    r.bold = True
    r.italic = True
    p.add_run(text).italic = True

def response(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(10)
    r = p.add_run("Response: ")
    r.bold = True
    p.add_run(text)

title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = title.add_run("Response to Reviewers \u2014 Manuscript ID 1926843")
r.bold = True
r.font.size = Pt(14)

body_intro = doc.add_paragraph()
body_intro.add_run(
    "We thank both reviewers for a careful, technically substantive review. Both "
    "reviewers converged on the same root cause for our original submission's "
    "central methodological weakness (image-level train/test splitting) and "
    "identified two further real issues we had not recognized: channel fusion "
    "before feature extraction, and a bug in Supplementary Figure S7 in which the "
    "plotted loss curve came from an unrelated, unused script. We have re-run the "
    "full analysis pipeline to address every point raised. Below we respond to each "
    "comment individually; italicized text reproduces the reviewer's comment, and "
    "our response follows in each case. All results below reflect the revised "
    "manuscript's actual output tables (Tables 1-9), not summaries."
)

# =====================================================================
# REVIEWER 1
# =====================================================================
h1("Reviewer 1")

h2("Comment 1 \u2014 Train/validation/test split construction")
comment(
    "The manuscript states that 824 images were divided into 560 training, 99 "
    "validation, and 165 test images, but it is not clear whether the split was "
    "performed randomly at the image level or grouped by compound... The authors "
    "should provide the exact splitting procedure and demonstrate that there is no "
    "overlap of compound/treatment identities between the relevant partitions. "
    "Ideally, the principal analysis should be repeated using a group-wise split..."
)
response(
    "The original split was indeed performed at the image level, stratified only "
    "by class label, with no control for compound identity. We agree this was the "
    "single most important issue with the original submission. We have replaced it "
    "with a compound-level grouped split (Methods 2.1): every image belonging to a "
    "given compound is assigned entirely to one partition (train, validation, or "
    "test), so no compound's images can appear in more than one partition. The "
    "entire principal analysis (all five models, all tables and figures) has been "
    "re-run on this corrected split. The new split contains 468 training, 28 "
    "validation, and 328 test images (Table 1); the change in partition sizes from "
    "the original 560/99/165 reflects the change in splitting unit (compound "
    "rather than image) and the uneven number of images per compound. Four of ten "
    "classes are represented in this dataset subset by a single compound each and "
    "cannot be split at the compound level without leakage; we kept these four "
    "classes entirely in training and report their absence from validation/test "
    "explicitly throughout, rather than silently dropping their metrics (Table 1, "
    "Methods 2.1, Limitations). All headline results changed substantially under "
    "the corrected split (Results 3.1-3.3); we consider the new results more "
    "defensible, not merely different."
)

h2("Comment 2 \u2014 Pulmonary/redox-pharmacology framing")
comment(
    "The pulmonary and redox-pharmacology framing appears considerably stronger "
    "than the experimental evidence supports... The authors should either provide "
    "an independent redox-related validation or substantially moderate the title, "
    "Abstract, Discussion, and Conclusion..."
)
response(
    "We agree and have taken the second option. The title, Abstract, Introduction, "
    "Discussion (Section 4.3), and Conclusion have all been rewritten to present "
    "this work as a computational benchmarking and methodology study on BBBC021 "
    "MOA classification. We no longer state or imply that the learned morphological "
    "features are demonstrated oxidative-stress-associated signatures, and we "
    "removed the unsupported claim (originally in the Discussion) that specific "
    "ablation-ranked features \u201creflect cellular-state changes associated with "
    "ROS-mediated cytotoxicity.\u201d Relevance to redox pharmacology and pulmonary "
    "disease is retained only as a stated motivation for why MOA classification "
    "pipelines of this kind matter, explicitly flagged in the Discussion "
    "(Section 4.3) and Limitations (Section 4.4) as an untested direction requiring "
    "either an independent ROS/oxidative-stress readout or a pulmonary cell-based "
    "assay, neither of which is part of the present study."
)

h2("Comment 3 \u2014 Seed replication for pretrained vs. scratch comparison")
comment(
    "The Discussion states that five additional random initializations of the "
    "scratch ResNet-18 produced AUC values ranging approximately from 0.75 to 0.96, "
    "whereas the principal reported scratch result is substantially lower... All "
    "deep-learning models, particularly both ResNet-18 conditions, should be "
    "trained using multiple predefined random seeds, and performance should be "
    "reported as distributions or mean \u00b1 SD/CI..."
)
response(
    "Both ResNet-18 conditions are now trained across five predefined seeds (1, 7, "
    "42, 123, 999) under the corrected compound-level split and per-channel "
    "features, with the split held fixed across seeds (Methods 2.3). This is now "
    "the primary reported result for both conditions everywhere they appear "
    "(Table 3, Figures 5c, Results 3.2-3.3), not a single run. Under the five-seed "
    "distribution: scratch ResNet-18 accuracy = 0.261 \u00b1 0.076 (range "
    "0.174-0.357), AUC = 0.826 \u00b1 0.048 (range 0.764-0.875); pretrained ResNet-18 "
    "accuracy = 0.721 \u00b1 0.042 (range 0.652-0.756), AUC = 0.965 \u00b1 0.007 (range "
    "0.957-0.974). The pretrained model is significantly higher-performing (Welch "
    "t=11.83, p<0.0001 for accuracy; t=6.38, p=0.0027 for AUC) and roughly 7-fold "
    "more stable across seeds by AUC standard deviation, which we now report as a "
    "finding in its own right (Discussion 4.2) rather than a Discussion-only "
    "caveat."
)

h2("Comment 4 \u2014 Training-protocol comparability between ResNet conditions")
comment(
    "Please provide the complete training protocol for each model, including "
    "optimizer, learning rates, batch size, total epochs, weight decay, "
    "augmentation, loss function, class weighting, early stopping/model-selection "
    "criteria, and random seeds, and ensure that the comparison is methodologically "
    "fair."
)
response(
    "On inspecting the code to answer this question precisely, we found a genuine "
    "asymmetry the original manuscript did not disclose because we had not "
    "recognized it ourselves: the scratch ResNet-18 was trained for only 10 epochs, "
    "while the pretrained ResNet-18 was trained for 10 (frozen stage) + 6 "
    "(fine-tuning stage) = 16 epochs. We have corrected this: the scratch model now "
    "trains for the same total budget of 16 epochs (Methods 2.3), so the "
    "comparison isolates the effect of pretraining rather than the effect of "
    "additional gradient updates. Full protocol, now stated explicitly (Methods "
    "2.3): Adam optimizer, softmax cross-entropy loss, batch size 16, no weight "
    "decay, no data augmentation, no class weighting or resampling for any "
    "deep-learning model, no early stopping (final-epoch checkpoint used for "
    "evaluation for all models), 5 predefined seeds per ResNet-18 condition. CNN: "
    "6 epochs, lr=1e-3. ResNet-18 pretrained: stage 1 (frozen backbone, FC only) 10 "
    "epochs at lr=1e-3; stage 2 (last two residual blocks + FC unfrozen) 6 epochs "
    "at lr=1e-4. ResNet-18 scratch: 16 epochs at lr=1e-3, all layers trainable "
    "throughout (no frozen stage, since there is no pretrained representation to "
    "protect)."
)

h2("Comment 5 \u2014 DeLong test validity for macro-averaged multiclass AUC")
comment(
    "The conventional DeLong procedure is defined for comparison of correlated ROC "
    "AUCs, and its application to an aggregated multiclass macro-OvR quantity is "
    "not self-evident... A paired bootstrap analysis of the difference in "
    "macro-AUC... would provide a more transparent complementary comparison."
)
response(
    "We agree and have made the paired bootstrap test the primary statistical "
    "comparison (Methods 2.4): the test set is resampled with replacement 2,000 "
    "times, macro-AUC is recomputed for both models on each resample with model "
    "pairing preserved, and we report the resulting 95% percentile CI and a "
    "two-sided bootstrap p-value. DeLong is retained only as a secondary, clearly "
    "labeled cross-check (Table 9); both methods agree in direction and "
    "significance for both comparisons reported (pretrained ResNet-18 vs. Random "
    "Forest: bootstrap CI [0.033, 0.078], p<0.001, DeLong z=5.10, p=3.4e-7; CNN vs. "
    "Random Forest: bootstrap CI [-0.374, -0.314], p<0.001, DeLong z=-22.07, "
    "p=6.0e-108). We describe explicitly how the multiclass predictions were "
    "transformed (macro-OvR, mean over per-class binarized AUCs restricted to "
    "test-evaluable classes) and note that the bootstrap resamples test images, "
    "not compounds, since 4 of 6 test-evaluable classes have only 2-3 test "
    "compounds -- too few to support a compound-level resampling scheme (Methods "
    "2.4, Limitations 4.4)."
)

h2("Comment 6 \u2014 Preprocessing, channel fusion, and normalization detail")
comment(
    "The fluorescence channels are fused into a grayscale image and subsequently "
    "intensity-normalized and contrast-enhanced, yet intensity-based variables such "
    "as mean intensity and total intensity are among the principal predictive "
    "features. The authors should specify whether normalization was performed per "
    "image, per batch, or globally... All normalization parameters must also be "
    "estimated without using test-set information."
)
response(
    "We have replaced channel fusion with per-channel feature extraction entirely "
    "(Methods 2.2), which directly resolves the interpretability concern: every "
    "intensity-based feature is now explicitly attributed to a channel (e.g., "
    "dapi_mean_intensity), so \u201cintensity of which marker\u201d is no longer ambiguous. "
    "Per-image intensity normalization/contrast-enhancement (resize, clip, "
    "normalize) is applied identically and independently to every image regardless "
    "of split membership, so it introduces no train/test leakage. Within-batch "
    "z-score normalization for the batch-effect analysis (Methods 2.4, Results "
    "3.5) is computed per acquisition batch using that batch's own images; we note "
    "this normalization is used only for the batch-effect diagnostic in Section "
    "3.5/Figure 8, not for the features fed to the classifiers evaluated in "
    "Sections 3.1-3.3, so it cannot leak test-set information into model training."
)

h2("Comment 7 \u2014 \u201cBiological validation\u201d terminology and dose-response analysis")
comment(
    "The reported Kruskal-Wallis differences among MOA classes essentially "
    "demonstrate that selected morphological features vary across the same class "
    "labels used for classification. This does not constitute independent "
    "biological validation... The authors should consider compound-stratified "
    "dose-response analyses..."
)
response(
    "We agree the Kruskal-Wallis result is a statistical association with the "
    "classification labels, not independent biological validation, and we no "
    "longer describe it as validation in the revised text (Results 3.5 now reads "
    "\u2018class-stratification\u2019 rather than \u2018biological validation\u2019 as the primary "
    "framing). The dose-response analysis is now computed within each compound "
    "separately rather than pooled across compounds of differing potency (Methods "
    "2.4, Results 3.5, Table 6): we report the median within-compound Spearman "
    "correlation and the fraction of tested compounds (of 8 with sufficient "
    "concentration levels) showing a sign-consistent dose-response relationship. "
    "Magnitudes are smaller than the pooled correlation we originally reported "
    "(median \u03c1 -0.02 to -0.33 vs. a pooled \u03c1\u224810.20), which we present as the "
    "expected and more honest consequence of not conflating potency across "
    "compounds."
)

h2("Comment 8 \u2014 Missing implementation details (LR/RF hyperparameters, ECE, proper scoring rules)")
comment(
    "For Logistic Regression and Random Forest, hyperparameters, scaling, "
    "regularization, class weighting, number of trees, and tuning procedures "
    "should be reported. For ECE, the number and definition of calibration bins "
    "and the multiclass calculation procedure should be specified... reporting "
    "additional proper scoring rules such as negative log-likelihood or Brier "
    "score would strengthen this analysis."
)
response(
    "Full LR/RF configuration is now stated in Methods 2.3: Logistic Regression "
    "(median imputation, standardization, L-BFGS solver, C=1.0, max_iter=2500, no "
    "class weighting, no hyperparameter search) and Random Forest (450 trees, "
    "class_weight='balanced_subsample', median imputation, remaining "
    "hyperparameters at scikit-learn defaults, no hyperparameter search). ECE "
    "definition is now stated explicitly (Methods 2.4): confidence-based "
    "(max-probability), 10 uniform-width bins over [0,1], sample-weighted average "
    "absolute gap between binned confidence and binned accuracy. We have not added "
    "NLL/Brier score in this revision cycle given the scope of changes already "
    "required; we note this as a straightforward addition for a future revision "
    "round if the editor and reviewers consider it necessary before acceptance, "
    "since ECE alone (n=328 across 6 evaluable classes) is admittedly a limited "
    "basis for a strong calibration claim, a point we now state directly (Results "
    "3.4)."
)

doc.add_page_break()

# =====================================================================
# REVIEWER 2
# =====================================================================
h1("Reviewer 2")

items = [
    ("1", "Split respects experimental structure",
     "No mention is made of how the partition respects the experimental structure "
     "of BBBC021... Please re-run all five models under well-level grouped "
     "splitting at minimum, and under NSC / NSCB leave-one-compound-out as the "
     "primary protocol.",
     "Implemented as requested: every image is now assigned to train/val/test by "
     "compound identity (Methods 2.1), which is a stricter guarantee than "
     "well-level grouping (no compound's images, across any of its wells, appear "
     "in more than one partition) and is effectively a leave-some-compounds-out "
     "protocol per class, given how few compounds most classes contain (Table 1). "
     "We did not implement a strict leave-one-compound-out cross-validation "
     "protocol (which would require retraining each model once per held-out "
     "compound); given the very small number of compounds in several classes "
     "(as few as 1-2), a single grouped train/val/test split was judged more "
     "stable and interpretable than per-compound cross-validation at this sample "
     "size, and we say so directly as a scoping decision (Methods 2.1). All five "
     "models were re-run under this split (Tables 2-3); as the reviewer predicted, "
     "the gap between the pretrained ResNet-18 and the feature-based baselines "
     "narrows substantially once leakage is removed."),
    ("2", "\u201c\u226580 images\u201d inclusion criterion contradicted by data",
     "This is contradicted by the data. Row sums of the confusion matrices... give "
     "test-set supports of 12, 22, 32, 9, 2, 17, 27, 32, 10 and 2 across the ten "
     "classes... Either the criterion is inaccurate as stated or the split was not "
     "stratified. Please clarify and correct.",
     "The reviewer is correct, and we thank them for identifying this precisely -- "
     "the stated criterion was simply wrong. The true class-selection rule (Methods "
     "2.1) is: the 10 most-represented MOA classes among compounds with an "
     "annotated MOA label in the ~1,600-image candidate pool retrieved from the 16 "
     "downloaded archive files, with no minimum-images-per-class threshold applied. "
     "Actual class sizes range from 12 to 160 images (Table 1); the two smallest "
     "classes (DNA replication, Protein synthesis) have only 12 images each. We "
     "have removed the incorrect \u201c\u226580 images\u201d statement and replaced it with the "
     "actual criterion."),
    ("3", "Channel fusion discards MOA-discriminative signal",
     "BBBC021 images are three-channel (DNA, \u03b2-tubulin, F-actin), and the "
     "distinction between microtubule destabilizers, stabilizers, actin disruptors "
     "and DNA-damaging agents is precisely the distinction between which channel "
     "changes... Please extract features per channel and report the result.",
     "Implemented exactly as requested (Methods 2.2): all 14 morphological "
     "descriptors are now computed independently per channel (DAPI, Tubulin, "
     "Actin), yielding 42 named, channel-attributable features, and deep-learning "
     "models now receive the genuine 3-channel image stack rather than a fused-"
     "and-repeated grayscale channel. Holding the compound-level split fixed, this "
     "change alone improved Random Forest macro-AUC from 0.810 to 0.904 and "
     "Logistic Regression from 0.794 to 0.866 (Results 3.1), directly supporting "
     "the reviewer's hypothesis about the mechanism of the original models' "
     "underperformance."),
    ("4", "Training configuration not reported",
     "Please state total epochs, optimiser, initial learning rate and schedule, "
     "batch size, weight decay, augmentation policy, any class weighting or "
     "resampling... the early-stopping criterion, and the checkpoint-selection "
     "rule. The loss function should also be stated explicitly. The stage-2 epoch "
     "count is described only as \u2018the remaining epochs\u2019, with no total given "
     "anywhere.",
     "Full protocol now stated (Methods 2.3), reproduced here: Adam optimizer, "
     "softmax cross-entropy loss (all deep models), batch size 16, no weight "
     "decay, no augmentation, no class weighting/resampling, no early stopping "
     "(final-epoch checkpoint evaluated for every model). CNN: 6 epochs, lr=1e-3. "
     "ResNet-18 pretrained: stage 1 = 10 epochs at lr=1e-3 (frozen backbone); stage "
     "2 = 6 epochs at lr=1e-4 (last two residual blocks + FC unfrozen); total = 16 "
     "epochs. ResNet-18 scratch: 16 epochs at lr=1e-3 (all layers trainable), "
     "matched to the pretrained model's total budget -- this is itself a "
     "correction, since the original scratch model trained for only 10 epochs, an "
     "asymmetry we had not previously recognized or disclosed."),
    ("5", "DeLong test misapplied to macro-averaged multiclass AUC",
     "Applying it to macro-averaged multiclass AUCs, aggregated across ten "
     "one-versus-rest problems with unmodelled between-class correlation and "
     "highly unequal class support, is not valid. Please replace with a paired "
     "bootstrap on \u0394AUC... and report the confidence interval of \u0394AUC rather than "
     "a z-statistic.",
     "Implemented exactly as requested (Methods 2.4, Table 9): paired bootstrap on "
     "\u0394 macro-AUC (2,000 resamples of the test set, model pairing preserved), "
     "reported as a 95% CI and two-sided p-value, now the primary comparison. "
     "DeLong is retained only as a secondary, explicitly labeled cross-check; both "
     "agree in direction and significance for both comparisons reported."),
    ("6", "Accuracy/AUC dissociation and tiny class supports",
     "The dissociation between accuracy and AUC... is a gap of roughly 33 points "
     "in a 10-class problem and is left unexplained... Please report per-class n "
     "with every per-class metric, add weighted-average AUC alongside macro-"
     "average, and provide bootstrap confidence intervals on per-class AUCs.",
     "Per-class n is now reported for every partition (Table 1), and per-class AUC "
     "with bootstrap CIs is reported for every model in the full results tables "
     "(results_summary/advanced_model_metrics.csv, summarized as needed in the "
     "main text). We have not added a weighted-average AUC column in this revision "
     "cycle; given that 4 of 10 classes have zero test weight under the corrected "
     "split (Table 1), a support-weighted average over only the 6 evaluable "
     "classes would numerically resemble macro-AUC weighted toward the two "
     "largest of those six classes (Microtubule stabilizers and destabilizers, "
     "which together account for over 60% of the test set) rather than adding new "
     "information; we are glad to add it explicitly if the reviewer considers it "
     "still useful. The specific 33-point accuracy/AUC gap the reviewer flagged in "
     "our original submission does not have a single clean re-measurement in the "
     "revised results, since both the split and the features changed; the "
     "underlying mechanism the reviewer identified (macro-averaging is sensitive "
     "to small per-class supports) remains directly relevant and is now visible "
     "explicitly via Table 1's per-class n rather than requiring the reader to "
     "back-calculate it from confusion-matrix row sums as before."),
    ("7", "Headline scratch result drawn from the worst of six runs",
     "The value reported here... falls below the entire 0.75-0.96 range that the "
     "Discussion reports for five re-initialisations of the same model... Please "
     "report all deep models as mean \u00b1 SD and range over at least five seeds, and "
     "recompute every downstream comparison on that basis.",
     "Implemented exactly as requested for both ResNet-18 conditions (Methods 2.3, "
     "Table 3, Results 3.2-3.3): five seeds, mean \u00b1 SD and range reported as the "
     "primary result everywhere both models appear, and the statistical comparison "
     "(Section 3.3) is computed on this basis via both the single-run paired "
     "bootstrap/DeLong tests and an independent five-seed group-level test (Welch "
     "and Mann-Whitney). We have not extended multi-seed evaluation to the compact "
     "CNN in this revision cycle and say so explicitly (Limitations 4.4) rather "
     "than presenting its single run as more representative than it is."),
    ("8", "Ablation collinearity, pooled dose-response, unquantified batch correction",
     "(i) The 'top five' ablation removes only three independent quantities... "
     "(ii) Spearman correlation... pooled across compounds of differing potency is "
     "not interpretable... (iii) The claim that normalisation confirms 'effective "
     "removal of technical variation' is asserted from visual inspection of a PCA "
     "scatter; please quantify...",
     "All three implemented. (i) Ablation now deduplicates collinear feature pairs "
     "before ranking, and removes a collinear partner alongside any feature "
     "selected for removal, so \u2018N features removed\u2019 always means N independent "
     "degrees of freedom (Methods 2.4, Table 5). (ii) Dose-response correlation is "
     "now computed within each compound and summarized by median rho and "
     "sign-consistency fraction across compounds (Methods 2.4, Table 6), replacing "
     "the pooled-across-compounds statistic. (iii) Batch-effect correction is now "
     "quantified via silhouette score on batch labels and 3-fold cross-validated "
     "batch-identity prediction accuracy, before and after normalization (Methods "
     "2.4, Table 8), in addition to the visual PCA plots (now with %-variance-"
     "explained axis labels and a sorted legend, addressing comment 14 below); we "
     "report the improvement as modest by these metrics rather than overstating "
     "normalization's effect."),
    ("9", "Hardware specification and total epoch count for cost table",
     "No hardware specification is given... The explanation that the pretrained "
     "model trained faster... holds only for the ten frozen epochs, and the total "
     "epoch count is never stated.",
     "Hardware is now stated explicitly (Methods 2.3, Results 3.6): a single "
     "CPU-only workstation, no GPU. Total epoch counts for both ResNet-18 "
     "conditions are now stated and matched at 16 (comment 4 above), which also "
     "corrects the original speed explanation: the pretrained model's shorter "
     "training time is no longer confounded with it also seeing fewer total "
     "training epochs."),
    ("10", "Unsupported ROS-mediated-cytotoxicity attribution",
     "The statement that the dominant ablation features 'reflect cellular-state "
     "changes associated with ROS-mediated cytotoxicity' attributes a redox "
     "interpretation to spot count, mean intensity and area-covered ratio with no "
     "supporting evidence. Please remove or substantiate.",
     "Removed. The Discussion no longer attributes a redox mechanism to any "
     "specific ablation-ranked feature (Discussion 4.3)."),
    ("11", "Seed-sensitivity disclosure placement reads as selective reporting",
     "Reported as written -- worst run in the tables, full range in the "
     "Discussion -- many readers will read this as selective reporting. It should "
     "be corrected rather than caveated.",
     "Corrected rather than caveated: the five-seed distribution is now the "
     "reported result in the tables themselves (Table 3), not a single run with "
     "the range disclosed only in the Discussion (Discussion 4.2)."),
    ("12", "Conclusion overstates redox/pulmonary characterization",
     "Nothing in the study supports [characterizing redox-active compounds "
     "relevant to pulmonary pharmacology]... Please scope the conclusion to what "
     "was demonstrated, or add the ROS-annotation analysis proposed in the "
     "Discussion...",
     "The Conclusion has been rewritten to scope the claimed contribution to what "
     "was demonstrated: a leakage-controlled, per-channel MOA classification "
     "benchmark and methodology. We did not add the proposed ROS-annotation "
     "analysis in this revision cycle (it would require an additional, separate "
     "data-annotation effort) and instead moderated the claim, consistent with our "
     "response to Comment 2 above."),
    ("13", "Figure 3/5 legend values disagree with tables (mislabeled averaging)",
     "What is plotted is presumably a micro-averaged (pooled) curve, a different "
     "quantity... Please plot per-class curves with the macro-average overlaid, or "
     "label the micro-average correctly, and reconcile all values.",
     "Figures 3 and 5 are regenerated as genuine macro-averaged ROC curves "
     "(interpolated per-class TPR averaged over all test-evaluable classes at each "
     "FPR, per the standard macro-ROC construction), with legend AUC values that "
     "now match Table 2 exactly by construction, since both are computed from the "
     "same saved probability arrays. We additionally provide per-class ROC plots "
     "with the macro-average explicitly overlaid (Figures 3b, 5b) exactly as "
     "requested, so the relationship between per-class and macro performance is "
     "directly visible rather than collapsed into one ambiguous curve."),
    ("14", "Figure 8/S8 missing axis labels and unsorted batch legend",
     "Both lack axis labels... and variance-explained annotations for PC1 and "
     "PC2. Batch labels are unsorted, with Week1, Week3 and Week2 interleaved.",
     "Fixed directly: both panels now label PC1/PC2 with %-variance-explained, and "
     "the batch legend is sorted (Methods 2.4, Figure 8, Table 8)."),
    ("15", "Supplementary Figure S7: overfitting shape, BCE-vs-softmax, unexplained epoch-7 drop, missing ResNet curves",
     "(i) the network never generalises better than its initialisation... (ii) the "
     "outputs are not a posterior distribution over the ten classes... (iii) the "
     "sharp fall at epoch 7... is unexplained... (iv) No curves are shown for "
     "either ResNet-18...",
     "The reviewer's technical diagnosis in (ii) was exactly correct, and this was "
     "the single most serious specific finding in either review. We traced the "
     "figure's generation and confirmed that Supplementary Figure S7 plotted the "
     "loss curve of a separate, abandoned exploratory script "
     "(deep_learning_pipeline.py) that trains a binary classifier "
     "(BCEWithLogitsLoss, 8 epochs, matching the figure's epoch count and initial-"
     "loss value exactly, as the reviewer inferred) and was never used to produce "
     "any result reported in the manuscript. The actual compact CNN whose results "
     "appear in Tables 2-4 is trained with softmax cross-entropy over the 10 "
     "classes (6 epochs) in a different script, which had no loss-curve logging at "
     "all. We have added loss-history logging to the correct training function and "
     "regenerated Figure 7 (renumbered from Supplementary S7) directly from that "
     "model. We also now provide loss curves for both ResNet-18 conditions "
     "(Figures 7b-c), addressing point (iv); the stage-transition epoch is marked "
     "explicitly on the pretrained curve. Because the original figure never "
     "reflected any reported result, we cannot retrospectively explain the "
     "specific overfitting shape or epoch-7 drop the reviewer described in points "
     "(i) and (iii) -- those were both artifacts of the wrong script's output being "
     "plotted, not properties of a model whose numbers appear anywhere in this "
     "manuscript, and neither the Table 2-4 ECE nor accuracy values require any "
     "correction as a result, since they were never computed from that script."),
]

for num, title_text, comment_text, response_text in items:
    h2(f"Comment {num} \u2014 {title_text}")
    comment(comment_text)
    response(response_text)

doc.save(r"C:\Users\nanda\Downloads\BBBC021_revision\Response_to_Reviewers.docx")
print("Response letter saved.")
