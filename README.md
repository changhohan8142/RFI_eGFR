# Explainable Foundation Model-Based Optimization for Renal Function Prediction Using Retinal Fundus Images

This repository contains the official implementation of the study:  
**"Explainable foundation model-based optimization for renal function prediction using retinal fundus images"**

The repository provides code for model training, inference, performance evaluation, and explainability analysis used in the manuscript.

---

## 🧩 Installation

This codebase requires Python 3.11.8.  
Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
````

---

## 🧠 Model Training

Model training can be initiated using the provided shell script:

```bash
bash egfr.sh
```

The training process relies on the following key components:

* `trainer.py` — main training loop and optimization logic
* `egfr.sh` — shell script for launching training across architectures and fine-tuning strategies

---

## 🔍 Inference and Performance Evaluation

Model inference and predictive performance evaluation are implemented in:

* **`trainer.py`** — saves inference outputs on validation and test sets
* **`saliency and evaluation.ipynb`** — summarizes model performance and evaluates regression/classification metrics

---

## 💡 Explainability Analysis

For saliency map generation, regional saliency quantification, and visualization, see:

* **`saliency and evaluation.ipynb`**

---

## 📘 Citation

If you use this code or reproduce results from the study, please cite our paper:

```text
to be updated
```
