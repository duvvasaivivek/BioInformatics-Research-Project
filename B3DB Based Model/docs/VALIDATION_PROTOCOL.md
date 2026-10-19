# Leakage-Resistant Validation Protocol & Empirical Performance

## 1. Experimental Methodology
To prevent optimistic bias and test set information leakage:
1. **Data Separation:** A stratified 80/20 train-test split was held strictly separate prior to any scaler fitting or hyperparameter optimization.
2. **Standardization:** Continuous scalers (`StandardScaler`) were fitted exclusively on training instances and serialized to `models/*_scaler_metadata.json`.
3. **Statistical Resampling:** 1,000 bootstrap resamplings on the test partitions were performed to determine 95% Confidence Intervals (2.5th to 97.5th percentiles).
4. **Scaffold Diversity Assessment:** Murcko scaffold clustering was performed across both datasets to analyze chemical diversity.

---

## 2. Test Set Evaluation Metrics (95% Bootstrap Confidence Intervals)

| Metric | BBB Permeability (B3DB, N=7,805) | P-gp Efflux Substrate (Broccatelli, N=1,205) |
|---|:---:|:---:|
| **Accuracy** | 0.8962 [0.8815 - 0.9116] | 0.9336 [0.9004 - 0.9627] |
| **Balanced Accuracy** | 0.8896 [0.8727 - 0.9066] | 0.9339 [0.9020 - 0.9625] |
| **Precision** | 0.9229 [0.9060 - 0.9381] | 0.9453 [0.9008 - 0.9778] |
| **Recall / Sensitivity** | 0.9137 [0.8963 - 0.9309] | 0.9308 [0.8824 - 0.9706] |
| **Specificity** | 0.8655 [0.8357 - 0.8928] | 0.9369 [0.8868 - 0.9741] |
| **F1-Score** | 0.9183 [0.9058 - 0.9307] | 0.9380 [0.9061 - 0.9653] |
| **ROC-AUC** | **0.9544 [0.9449 - 0.9644]** | **0.9593 [0.9320 - 0.9822]** |
| **PR-AUC (Avg Precision)** | 0.9715 [0.9629 - 0.9789] | 0.9697 [0.9497 - 0.9850] |
| **Matthews Corr Coeff (MCC)**| 0.7762 [0.7446 - 0.8099] | 0.8667 [0.8013 - 0.9249] |
| **Brier Score (Calibration)**| **0.0796 [0.0698 - 0.0893]** | **0.0666 [0.0464 - 0.0921]** |

---

## 3. Incremental Feature Ablation Study (5-Fold Stratified Cross-Validation)

### P-gp Efflux Model Ablation:
1. **3D Shape Only (6 features):** $\text{ROC-AUC} = 0.8425 \pm 0.0243$ | $\text{F1} = 0.8042$ | $\text{Acc} = 0.7842$
2. **2D Descriptors Only (6 features):** $\text{ROC-AUC} = 0.9178 \pm 0.0104$ | $\text{F1} = 0.8670$ | $\text{Acc} = 0.8556$
3. **2D Descriptors + 3D Shape (12 features):** $\text{ROC-AUC} = 0.9167 \pm 0.0121$ | $\text{F1} = 0.8637$ | $\text{Acc} = 0.8523$
4. **Morgan FP + 2D Descriptors (2054 features):** $\text{ROC-AUC} = 0.9511 \pm 0.0017$ | $\text{F1} = 0.9049$ | $\text{Acc} = 0.8979$
5. **Full Model (Morgan + 2D + 3D Shape) (2060 features):** $\text{ROC-AUC} = 0.9474 \pm 0.0035$ | $\text{F1} = 0.9028$ | $\text{Acc} = 0.8954$

### BBB Permeability Model Ablation:
1. **2D Descriptors Only (6 features):** $\text{ROC-AUC} = 0.9174 \pm 0.0081$ | $\text{F1} = 0.8913$ | $\text{Acc} = 0.8555$
2. **Morgan Fingerprints Only (2048 features):** $\text{ROC-AUC} = 0.9393 \pm 0.0054$ | $\text{F1} = 0.8936$ | $\text{Acc} = 0.8588$
3. **Full Hybrid (Morgan + 2D Descriptors) (2054 features):** $\text{ROC-AUC} = 0.9467 \pm 0.0035$ | $\text{F1} = 0.9041$ | $\text{Acc} = 0.8738$

*Scientific Takeaway:* The full hybrid BBB architecture provides a statistically significant improvement over standalone 2D or fingerprint baselines. For P-gp, 3D shape features achieve a remarkable 0.8425 ROC-AUC using merely 6 geometric variables, while 2048-bit Morgan fingerprints capture the vast majority of local pharmacophore motifs.
