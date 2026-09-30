# Measured Waste Classification Results

These results were generated from this experiment. Inspect them and their limitations before adding them to a manuscript.

## Experimental status

Completed runs: 12. Full standard 4-model × 3-seed study: True. CPU latency available for all runs: True.

## Model comparison

model,accuracy_mean,accuracy_std,macro_precision_mean,macro_precision_std,macro_recall_mean,macro_recall_std,macro_f1_mean,macro_f1_std,weighted_f1_mean,weighted_f1_std,training_seconds_mean,training_seconds_std,model_size_mib_mean,model_size_mib_std,parameters_mean,parameters_std,model_median_ms_mean,model_median_ms_std,model_p95_ms_mean,model_p95_ms_std,end_to_end_median_ms_mean,end_to_end_median_ms_std
custom_cnn,0.56349,0.0347,0.58795,0.03873,0.57276,0.03018,0.53536,0.02954,0.56624,0.03665,760.57935,243.81757,0.47366,0.0,110534.0,0.0,15.94618,0.30996,18.73524,0.82449,17.92238,0.35797
efficientnetb0,0.8739,0.01222,0.84647,0.01559,0.8653,0.01904,0.85256,0.01658,0.87612,0.0119,1338.69475,764.72898,16.296,0.0,4057257.0,0.0,19.57802,0.10926,22.63141,0.54057,21.48273,0.04661
mobilenetv2,0.84656,0.00916,0.81764,0.01757,0.82604,0.00939,0.81935,0.01343,0.847,0.00769,975.28751,116.78231,9.21757,0.0,2265670.0,0.0,12.76398,0.43643,15.36612,0.34742,14.57218,0.49604
resnet50,0.89506,0.00764,0.87859,0.00903,0.87861,0.01487,0.87708,0.01246,0.8947,0.00803,5476.47107,5397.13549,90.66713,0.0,23600006.0,0.0,93.98483,0.33494,100.79838,1.00944,95.91015,0.58315


Values in CSV tables are fractions for accuracy and F1 (multiply by 100 for percentages). Standard deviation describes training-seed variability on one fixed split, not uncertainty across datasets. Model size is the .keras inference archive including preprocessing and weights. Training time includes validation prediction for macro F1 and checkpoint writing.

## Results paragraph to edit

Among the completed configurations, resnet50 obtained the highest mean test macro F1 (0.8771). This ranking is specific to the fixed dataset split and training protocol. Class-wise metrics and paired uncertainty analysis should be considered before interpreting a small difference as meaningful. No practical sorting reliability is established by this result alone.

## Figures and tables

comparison_macro_f1.png: Mean test macro F1 with one training-seed standard deviation.
performance_vs_latency.png: CPU median inference latency versus mean macro F1, when timing is available.
class_metrics_all.csv: Per-class precision, recall and F1 for every run.
robustness_summary.csv: Mean and standard deviation under predeclared image corruptions.
bootstrap_differences.csv: Exploratory paired stratified test-set confidence intervals.
Each run folder contains accuracy/loss curves and a confusion matrix.

## Limitations paragraph to edit

The evaluation uses a small dataset with controlled backgrounds and an imbalanced trash category. Three seeds share one split. Controlled corruptions simulate image degradation and do not demonstrate cross-dataset generalization. Inference timings apply only to the recorded CPU and software configuration. The scratch baseline and pretrained networks differ in architecture as well as initialization, so this experiment does not isolate a causal transfer-learning effect.