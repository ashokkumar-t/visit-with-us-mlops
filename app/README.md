# 🌍 Visit With Us - Wellness Tourism Prediction

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Live-red)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-yellow)
![GitHub Actions](https://img.shields.io/badge/GitHub-Actions-success)

---

# Overview

Visit With Us is an end-to-end Machine Learning Operations (MLOps) project that predicts whether a customer is likely to purchase the newly introduced **Wellness Tourism Package** before being contacted by the sales team.

The project demonstrates a complete MLOps workflow, including:

- Data Registration
- Data Preparation
- Feature Engineering
- Model Training
- Hyperparameter Tuning
- Model Evaluation
- Model Registry
- Streamlit Deployment
- GitHub Actions CI/CD

---

# Project Architecture

```
                GitHub
                   │
                   ▼
          GitHub Actions
                   │
      ┌────────────┼────────────┐
      ▼            ▼            ▼
Data Prep      Train Model    Evaluate
      │            │            │
      └────────────┼────────────┘
                   ▼
        Hugging Face Model Hub
                   │
                   ▼
          Streamlit Application
                   │
                   ▼
             Customer Prediction
```

---

# Folder Structure

```
app/
│
├── app.py
├── config.py
├── model_loader.py
├── predict.py
├── requirements.txt
├── Dockerfile
├── runtime.txt
└── README.md
```

---

# Machine Learning Algorithm

The deployed model uses:

**Bagging Classifier**

The model was selected after comparing:

- Decision Tree
- Random Forest
- Bagging
- AdaBoost
- Gradient Boosting
- XGBoost

using cross-validation and hyperparameter tuning.

---

# Deployment

The application automatically downloads the latest trained model from the Hugging Face Model Hub.

```
User
   │
   ▼
Streamlit
   │
   ▼
Model Loader
   │
   ▼
Hugging Face Model Hub
   │
   ▼
Prediction
```

---

# Running Locally

## Clone Repository

```bash
git clone https://github.com/itisashokkumar/visit-with-us.git
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run Streamlit

```bash
streamlit run app.py
```

---

# Docker

## Build

```bash
docker build -t visit-with-us .
```

## Run

```bash
docker run -p 7860:7860 visit-with-us
```

---

# Hugging Face

## Dataset

https://huggingface.co/datasets/itisashokkumar/visit-with-us-dataset

---

## Model Hub

https://huggingface.co/itisashokkumar/visit-with-us-models

---

## Streamlit Space

https://huggingface.co/spaces/itisashokkumar/visit-with-us

---

# GitHub Repository

https://github.com/itisashokkumar/visit-with-us

---

# Input Features

The model expects the following customer information:

- Age
- Type of Contact
- City Tier
- Occupation
- Gender
- Number of Persons Visiting
- Preferred Property Star
- Marital Status
- Number of Trips
- Passport
- Own Car
- Number of Children Visiting
- Designation
- Monthly Income
- Pitch Satisfaction Score
- Product Pitched
- Number of Followups
- Duration of Pitch

---

# Output

The application predicts:

- Likely to Purchase
- Not Likely to Purchase

If supported by the model, it also displays the purchase probability.

---

# Technologies

- Python 3.11
- Scikit-Learn
- Bagging Classifier
- Streamlit
- Hugging Face Hub
- Docker
- GitHub Actions

---

# Future Improvements

- Batch prediction using CSV upload
- Explainable AI (SHAP)
- Model monitoring
- Automatic retraining
- Drift detection
- REST API using FastAPI

---

# Author

**Ashokkumar T**

Software Engineer

---

# License

MIT License