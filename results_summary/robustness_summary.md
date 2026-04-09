# Robustness Summary

- Batches tested: Week1_22123, Week1_22141, Week2_24121
- Images processed: 450
- Configurations tested: sensitive, default, conservative

## Classification by Configuration
- sensitive: LR acc=0.7556, LR AUC=0.8354, RF acc=0.8889, RF AUC=0.9547
- default: LR acc=0.6222, LR AUC=0.7531, RF acc=0.8667, RF AUC=0.9352
- conservative: LR acc=0.6444, LR AUC=0.6955, RF acc=0.8667, RF AUC=0.9270

## Output Files
- final_figures/figure6_batch_robustness.png
- final_figures/figure7_threshold_sensitivity.png
- final_figures/figure8_robustness_classification.png
- final_tables/robustness_feature_table.csv
- results_summary/robustness_batch_summary.csv
- results_summary/robustness_classification.csv
- results_summary/robustness_config_summary.csv