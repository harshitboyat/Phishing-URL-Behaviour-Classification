# 🔐 Phishing URL Behaviour Classification

A Machine Learning project that classifies web URLs as **Legitimate or Phishing** based on their structural characteristics.

This project was developed as part of my **LearnDepth Academy Machine Learning Internship** and covers the complete Machine Learning workflow from data preprocessing and feature selection to model training, evaluation, and Streamlit deployment.

---

## ❗ Problem Statement

Phishing attacks use deceptive web addresses to trick users into visiting malicious websites or exposing sensitive information.

The problem addressed by this project is to **automatically classify a given URL as either legitimate or phishing using Machine Learning**.

Instead of relying on paid security APIs, the system analyzes structural characteristics of a URL, such as:

- URL length
- Domain length
- Number of subdomains
- Number of digits
- Digit ratio
- Special characters
- Query parameters
- HTTPS usage
- Whether the domain is an IP address

The goal is to develop a lightweight URL-based classification system that can identify suspicious URLs and provide a prediction to the user.

---

## 🎯 Project Objectives

- Analyze a phishing URL dataset
- Perform data preprocessing and exploratory data analysis
- Identify relevant URL-based features
- Train Machine Learning classification models
- Compare model performance
- Perform error analysis
- Save the trained model and preprocessing objects
- Build a Streamlit application for real-time URL classification

---

## 📊 Dataset

The project uses the **PhiUSIIL Phishing URL Dataset** from the UCI Machine Learning Repository.

### Original Dataset

| Property | Value |
|---|---:|
| Total Instances | 235,795 |
| Features | 54 |
| Legitimate URLs | 134,850 |
| Phishing URLs | 100,945 |
| Missing Values | None reported by UCI |

The dataset contains URL-based characteristics as well as features related to webpage content.

For this project, the final deployment model uses only features that can be obtained directly from a URL.

### Dataset Source

UCI Machine Learning Repository:

https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset

> The complete dataset is not included in this repository because of its large size. A smaller sample may be included for demonstration purposes.

---

## 🧠 Features Used

The final URL-based model uses the following 14 features:

| # | Feature |
|---:|---|
| 1 | URLLength |
| 2 | DomainLength |
| 3 | IsDomainIP |
| 4 | TLDLength |
| 5 | NoOfSubDomain |
| 6 | NoOfLettersInURL |
| 7 | NoOfDegitsInURL |
| 8 | DegitRatioInURL |
| 9 | NoOfEqualsInURL |
| 10 | NoOfQMarkInURL |
| 11 | NoOfAmpersandInURL |
| 12 | NoOfOtherSpecialCharsInURL |
| 13 | SpacialCharRatioInURL |
| 14 | IsHTTPS |

These features represent structural properties of a URL that can be extracted without requiring a paid security API.

---

## 🔬 Machine Learning Approach

The project follows this workflow:

```text
Dataset
   ↓
Data Preprocessing
   ↓
Exploratory Data Analysis
   ↓
Feature Relevance Analysis
   ↓
URL-Based Feature Selection
   ↓
Train-Test Split
   ↓
Feature Scaling
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Error Analysis
   ↓
Model Saving
   ↓
Streamlit Deployment
