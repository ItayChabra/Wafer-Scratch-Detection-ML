# Wafer Scratch Detection (NI Assignment)

This repository contains the solution for a data science assignment given by **National Instruments (NI)**, aimed at building a machine learning model to automatically detect **scratches on semiconductor wafers** using wafer map data.

---

## 🧾 Project Overview

In semiconductor manufacturing, wafers consist of many small units called **dies**, some of which can become defective. **Scratches** are elongated patterns of bad dies (and occasionally good ones) that often indicate physical damage. Identifying these scratches is essential to ensure quality and yield.

Currently, the scratch detection process is manual, time-consuming, and error-prone. The purpose of this project is to **automate scratch detection** by predicting which dies—both good and bad—belong to a scratch, helping reduce costs and improve decision-making.

---

## 🎯 Objectives

* Predict which dies on a wafer belong to a scratch (whether they are bad - "Scratch" or good - "Ink")
* Build an ML model using provided wafer map data (coordinates, die quality, etc.)
* Make predictions on a test set with unknown scratch labels
* Output a submission CSV with predictions
* Ensure the solution supports business goals of **automation** and **quality control**

---

## 🧠 Challenges & Solutions

* **Class imbalance**: Few scratch/ink dies relative to normal ones → used resampling and metric tuning
* **Spatial structure**: Converted X, Y coordinates into spatial features
* **Non-continuous scratches**: Developed logic to detect disconnected but related points
* **Performance constraints**: Employed multiprocessing via `joblib` to parallelize evaluation and predictions

---

## 🧪 Approach

1. **Data Exploration**: Visualized wafer maps, distribution of scratch/ink/good dies
2. **Feature Engineering**: Encoded spatial and neighborhood features
3. **Modeling**: Trained models including Random Forest, XGBoost
4. **Validation**: Cross-validation using F1 score, precision, recall
5. **Testing**: Applied the best model to the test set
6. **Submission**: Generated CSV of predictions

---

## 📦 Files

* `notebook.ipynb` — Main notebook containing all steps from exploration to modeling and submission
* `scratch_predictions.csv` — Final submission file with predictions for each die in the test set
* `README.md` — This file

---

## ⚙️ Tech Stack

* Python 3.x
* Jupyter Notebook
* Pandas, NumPy
* Scikit-learn
* XGBoost
* Seaborn & Matplotlib (for visualization)
* Joblib (for parallel processing)

---
