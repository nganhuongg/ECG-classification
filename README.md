# Interpretable Multi-Label ECG Classification (ECGDeli Features)

## Problem

Build an interpretable multi-label ECG classification pipeline using structured ECGDeli features (~459 selected features after domain pruning) for ~21k recordings with long-tail diagnostic labels.

---

## Representation Choice

Instead of raw ECG waveforms (high-dimensional time series), we use fiducial-based ECGDeli features:

- Physiologically meaningful intervals (PR, QRS, QT, RR)
- Aggregated waveform statistics
- Tabular structure suitable for classical ML
- Clinically interpretable

Missing values followed a structured block pattern consistent with delineation failure (MAR).  

---

## Baseline Model

**Model:** One-vs-Rest Logistic Regression (L2-regularized)  
**Preprocessing:** Median imputation + Standard scaling  

**Baseline performance:**
- Macro F1 ≈ 0.255  
- Micro F1 ≈ 0.64  

Observation:
- Strong performance on dominant labels (e.g., SR, NORM)
- Multiple rare labels with F1 = 0  
- Large gap between macro and micro F1 → imbalance suspected

---

## Hypothesis Testing

### 1. Multicollinearity

Correlation filtering (|r| > 0.9):

- ~297 features removed
- Macro F1 → 0.2577 (+0.002)

**Conclusion:**  
Multicollinearity exists but is not the primary bottleneck.  
L2 regularization already mitigates most redundancy.

---

### 2. Class Imbalance

Logistic Regression with:

```python
class_weight="balanced"
```

Performance:

Macro F1 → 0.2712 (+0.016 over baseline)

Rare-class F1 improved substantially (only 2 labels remained at F1 = 0).

**Conclusion:**
Class imbalance is the dominant performance constraint.


---

### 3. Nonlinear Modeling

Random Forest (200 trees, balanced):

Macro F1 = 0.1186

Micro F1 = 0.6132

Random Forest underperformed Logistic Regression.

**Conclusion:**
The ECGDeli feature space carries predominantly linear signal.
Increasing model complexity does not solve the main limitation.

---

## Final Insights

Feature engineering is structurally sound.

Multicollinearity has limited empirical impact.

Class imbalance is the primary bottleneck.

Linear models outperform tree-based models in this feature space.

Performance ceiling is data-distribution driven, not model-capacity driven.

This project demonstrates controlled experimental reasoning and bottleneck diagnosis rather than blind performance tuning.