# 🧠 Wafer Scratch Detection (NI ML Assignment)

This repository presents a machine learning solution for a scratch detection task on semiconductor wafers, part of a technical assignment from **National Instruments**. The goal is to automatically identify **scratches**—elongated patterns of defective or seemingly good dies—on wafer maps using spatial data and die quality labels.

---

## 📌 Problem Context

In semiconductor fabrication, wafers contain a grid of **dies**, each representing a circuit. Some dies fail quality checks due to physical defects such as **scratches**, which appear as continuous or fragmented lines. Detecting these defects is currently manual, error-prone, and inefficient.

This project proposes a data-driven approach to **automate scratch detection**, improving quality control and reducing inspection time and cost.

---

## 🎯 Project Goals

* Detect dies associated with a **scratch**, even if the die is still marked as functional ("Ink")
* Build a robust model using wafer map features (X, Y, category, etc.)
* Predict scratch labels on an unlabeled test set and submit predictions
* Demonstrate practical skills in **feature engineering, modeling, and real-world ML deployment**

---

## 🔍 Key Challenges & Solutions

| Challenge                    | Solution                                                               |
| ---------------------------- | ---------------------------------------------------------------------- |
| Severe class imbalance       | Applied resampling and optimized metrics like F1 and precision         |
| Spatial structure of dies    | Engineered geometric and neighbor-based spatial features               |
| Disconnected scratches       | Used local clustering + die context to group spatially coherent points |
| Runtime and scale efficiency | Parallelized processing with `joblib`                                  |

---

## 🚀 Methodology

1. **Data Analysis**
   Explored wafer layout, class distributions, and defect patterns.

2. **Feature Engineering**
   Created spatial features: normalized coordinates, die neighborhood structure, die counts.

3. **Model Selection**
   Benchmarked models:

   * `Random Forest` (baseline)
   * `LightGBM` (final model – best performance)

4. **Cross-Validation Strategy**

   * Used GroupKFold to preserve wafer-level separation.
   * Optimized for **F1 score** to balance precision and recall.

5. **Prediction & Postprocessing**

   * Predicted die classes on test data
   * Refined predictions using die grouping and domain rules

6. **Submission**
   Generated `scratch_predictions.csv` for submission, formatted per spec.

---

## 🧰 Technologies Used

* Python 3.x
* Jupyter Notebook
* pandas, numpy
* scikit-learn
* LightGBM
* matplotlib, seaborn
* joblib (parallel processing)

---

## 🙏 Acknowledgment

Developed as part of a **Machine Learning Student** candidate assessment for **National Instruments**.

---

## ⚠️ Disclaimer

This repository is for educational demonstration only. All data and problem definitions remain the intellectual property of **National Instruments**.
