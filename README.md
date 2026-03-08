
<p align="center">
  <a href="https://git.io/typing-svg">
    <img
      src="https://readme-typing-svg.demolab.com?font=Poppins&weight=700&size=40&pause=1000&duration=2500&color=8A2BE2&center=true&vCenter=true&width=1100&height=110&lines=Student+Mental+Health+Analysis;Statistical+Analysis+%26+Predictive+Modeling;Does+CGPA+Influence+Mental+Health%3F"
      alt="Typing SVG"
    />
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white" />
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/SciPy-Chi--Square-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white" />
  <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white" />
  <img src="https://img.shields.io/badge/Status-Completed-2ecc71?style=for-the-badge" />
</p>


<h1 align="center">Student Mental Health Analysis</h1>



<p align="center">
  <i>Uncovering the hidden relationship between academic performance and student mental health</i>
</p>

---

## Table of Contents

- [About The Project](#about-the-project)
- [Dataset & Features](#dataset--features)
- [Project Pipeline](#project-pipeline)
- [Step-by-Step Breakdown](#step-by-step-breakdown)
  - [1. Libraries & Data Loading](#1-libraries--data-loading)
  - [2. Data Summary](#2-data-summary)
  - [3. Data Cleaning](#3-data-cleaning)
  - [4. Exploratory Data Analysis](#4-exploratory-data-analysis-eda)
  - [5. Feature Encoding](#5-feature-encoding)
  - [6. Hypothesis Testing](#6-hypothesis-testing--chi-square-test)
  - [7. Model Training & Evaluation](#7-model-training--evaluation)
- [Key Findings & Insights](#key-findings--insights)
- [Results Summary](#results-summary)
- [Tech Stack](#tech-stack)
- [How to Run](#how-to-run)
- [Future Improvements](#future-improvements)
- [Author](#author)

---

## About The Project

Mental health among university students is a growing concern worldwide, yet it remains widely underreported and under-studied. This project analyzes a **student mental health survey** conducted at a Malaysian university to answer:

> *"Do mental health problems like depression, anxiety, and panic attacks affect students' academic performance (CGPA)?"*

The surprising answer challenges the original hypothesis — and reveals a deeper, more nuanced relationship between academic pressure and mental health.

The notebook covers raw data cleaning, statistical hypothesis testing with Chi-Square tests, rich EDA with 10+ visualizations, and a Logistic Regression model to predict student depression.

**GitHub Repository:** [Student Mental Health](https://github.com/mdnaim2004/Student_Mental_Health)

---

## Dataset & Features

| Property | Details |
|---|---|
| **Name** | Student Mental Health Survey |
| **Source** | [Kaggle — Student Mental Health](https://www.kaggle.com/datasets/shariful07/student-mental-health) |
| **File** | `Student Mental health.csv` |
| **Survey Period** | July 8–18, 2020 |
| **Respondents** | ~101 university students |
| **Target Variable** | `depression` — Whether the student has depression (Yes/No) |

### Feature Descriptions

| Feature | Description |
|---|---|
| `timestamp` | Date and time of survey response |
| `gender` | Respondent's gender |
| `age` | Respondent's age |
| `course` | Enrolled course/major |
| `study_year` | Current year of study (Year 1–4) |
| `cgpa` | Academic performance in GPA range |
| `marital_status` | Whether the student is married |
| `depression` | Does the student have depression? (Yes/No) |
| `anxiety` | Does the student have anxiety? (Yes/No) |
| `panic_attack` | Does the student have panic attacks? (Yes/No) |
| `seek_treatment` | Has the student sought specialist treatment? (Yes/No) |

---

## Project Pipeline

```
Data Load  ──►  Data Summary  ──►  Data Cleaning  ──►  EDA  ──►  Encoding  ──►  Chi-Square Test  ──►  Model
```

---

## Step-by-Step Breakdown

### 1. Libraries & Data Loading

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
```

Dataset loaded from Kaggle's input directory and explored with `.shape`, `.info()`, and `.head()`.

---

### 2. Data Summary

Initial inspection revealed key issues:

- 1 missing value in `age` column
- `course` column: 40+ spelling variations for same courses
- `study_year`: `"year 1"` vs `"Year 1"` treated as different categories
- `cgpa`: duplicate categories due to trailing whitespace

---

### 3. Data Cleaning

#### Column Renaming
All columns renamed for clarity (`'Choose your gender'` → `'gender'`, etc.)

#### Course Category Mapping
40+ raw course entries grouped into faculty categories:

| Faculty | Example Raw Inputs |
|---|---|
| IT | BCS, BIT, IT |
| Engineering | Engineering, KOE, Engine, engin |
| Medicine | Biomedical science, Biotechnology, MHSC |
| Law | Laws |
| Economics and management | Econs |
| RSEP | psychology, Islamic Education, Human Sciences |

#### Study Year Fix
```python
df['study_year'] = df['study_year'].str.lower().str.strip()
```

#### CGPA Whitespace Fix
```python
df.cgpa = df.cgpa.str.strip()
```

#### Missing Age Value
Filled with column mean (~20) — appropriate since respondents are mostly 18–22 year old students:
```python
df['age'] = df['age'].fillna(20).astype(int)
```

#### Timestamp Parsing
`timestamp` column parsed into `day`, `month`, `year` — survey ran only July 8–18, 2020, so no further time analysis needed.

---

### 4. Exploratory Data Analysis (EDA)

#### Univariate Analysis

| Variable | Key Finding |
|---|---|
| `gender` | ~75% of participants are female |
| `marital_status` | Only 16 students are married |
| `depression` | 35 students reported depression |
| `anxiety` | Notable portion reported anxiety |
| `panic_attack` | Notable number reported panic attacks |
| `seek_treatment` | Very few sought professional help |
| `study_year` | Majority are first-year students |
| `cgpa` | Most fall in the 3.50–4.00 range |
| `course_category` | IT faculty has the most participants |
| `age` | Most participants are 18 years old |

#### Bivariate Analysis — CGPA vs Mental Health (Grouped Bar Charts)

| Mental Health Issue | Most Affected CGPA Group |
|---|---|
| Depression | 3.00–3.49 (19/43 students = 44%) |
| Anxiety | 3.50–4.00 (18/48 students = 38%) |
| Panic Attacks | 3.50–4.00 (19/48 students = 40%) |
| Seek Treatment | Only 4 students from top CGPA group |

> **Surprising Finding:** High-achieving students are NOT immune. In fact, they show higher rates of anxiety and panic attacks.

#### Mental Health by Study Year & CGPA (Heatmaps)
- Depression peaks in Year 1, especially CGPA 3.00–3.49
- Anxiety and panic attacks highest among Year 1 high-CGPA students
- Help-seeking remains very low across all groups

#### Mental Health by Course Category & CGPA (Heatmaps)
- Engineering and IT students report the highest mental health issues
- Anxiety is the most common condition among high-CGPA students

#### Mental Health by Age & CGPA (Heatmaps)
- Depression and anxiety most common at ages 18–19 with high CGPA
- Help-seeking remains critically low even in the most affected groups

---

### 5. Feature Encoding

#### CGPA Grouping
Low-frequency categories merged to meet Chi-Square requirements (min 5 per cell):

| Group | CGPA Ranges |
|---|---|
| Group 1 | 0–1.99, 2.00–2.49, 2.50–2.99 |
| Group 2 | 3.00–3.49 |
| Group 3 | 3.50–4.00 |

#### Label Encoding
```python
df['depression_encoded'] = le.fit_transform(df['depression'])
df['gender_encoded'] = le.fit_transform(df['gender'])
df['study_year_encoded'] = le.fit_transform(df['study_year'])
df['course_category_encoded'] = le.fit_transform(df['course_category'])
```

**Final feature set for modeling:**
```python
feature_cols = ['age', 'gender_encoded', 'study_year_encoded',
                'course_category_encoded', 'cgpa_grouped']
target_col = 'depression_encoded'
```

Train/Test Split: **80% / 20%** with `random_state=42`

---

### 6. Hypothesis Testing — Chi-Square Test

Formal statistical tests using `scipy.stats.chi2_contingency`:

| Test | H₀ (Null) | p-value | Conclusion |
|---|---|---|---|
| CGPA vs Depression | No relationship | 0.219 | No significant relationship |
| CGPA vs Anxiety | No relationship | > 0.05 | No significant relationship |
| CGPA vs Panic Attack | No relationship | > 0.05 | No significant relationship |
| Study Year vs Depression | Independent | > 0.05 | No significant relationship |
| Course Category vs Depression | Independent | > 0.05 | No significant relationship |

> **Interpretation:** Despite visual patterns in EDA, none of the relationships reached statistical significance (p < 0.05). Academic/demographic factors alone do not statistically predict mental health outcomes in this dataset.

---

### 7. Model Training & Evaluation

**SMOTE** was applied to handle class imbalance before training.

```python
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
```

**Evaluation Metrics:**
- Confusion Matrix — heatmap with Blues colormap
- Accuracy Score — `accuracy_score(y_test, y_pred)`
- Classification Report — Precision, Recall, F1-Score per class
- ROC Curve & AUC — measures overall discriminatory power

---

## Key Findings & Insights

**1. High achievers face mental health pressure too**
Students with CGPA 3.50–4.00 show the highest rates of anxiety and panic attacks — academic pressure may cause mental health issues, not the other way around.

**2. First-year students are the most vulnerable**
Year 1 dominates mental health issue reports across all categories — the transition to university life is a critical risk period.

**3. Engineering and IT students are the most affected**
Heavy workloads and high academic expectations in these fields correlate with higher reported mental health issues.

**4. Help-seeking is critically low**
Even in the most affected groups, very few students sought professional treatment — a major gap in campus mental health support.

**5. Statistical tests show no significant association**
Despite clear visual trends, Chi-Square tests found no statistically significant relationships — the dataset's small size (101 samples) limits statistical power.

**6. Gender sampling bias exists**
~75% of respondents are female — results should be interpreted with this imbalance in mind.

---

## Results Summary

| Metric | Details |
|---|---|
| Dataset Size | ~101 student responses |
| Target Variable | `depression` (Yes/No) |
| Model | Logistic Regression |
| Class Imbalance Handling | SMOTE |
| Train/Test Split | 80% / 20% |
| Statistical Test | Chi-Square |
| Chi-Square Significant? | No (all p > 0.05) |
| Most Affected Group | Year 1, High-CGPA, Engineering/IT |
| Most Reported Issue | Anxiety |
| Help-Seeking Rate | Very Low |

---

## Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Data loading, cleaning, manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | Base plotting |
| `seaborn` | Statistical visualizations |
| `scipy.stats` | Chi-Square hypothesis testing |
| `scikit-learn` | Label encoding, modeling, evaluation |
| `imbalanced-learn` | SMOTE for class imbalance |

---

## How to Run

**Option A — Run on Kaggle**
1. Open the notebook on Kaggle
2. Click **"Copy & Edit"** and run all cells.

**Option B — Run Locally**

```bash
git clone https://github.com/mdnaim2004/Student_Mental_Health.git
cd Student_Mental_Health
pip install pandas numpy matplotlib seaborn scipy scikit-learn imbalanced-learn
jupyter notebook student-mental-health.ipynb
```

Download `Student Mental health.csv` from [Kaggle](https://www.kaggle.com/datasets/shariful07/student-mental-health) and update the data path accordingly.

---

## Future Improvements

- Collect a larger dataset — 101 samples limits statistical power
- Add gender-stratified analysis
- Try Random Forest or XGBoost for better accuracy
- Apply SHAP values for interpretable predictions
- Multi-target classification — predict all 3 conditions simultaneously
- Build a Streamlit web app for student mental health screening

---

## Author

[![Kaggle](https://img.shields.io/badge/Kaggle-mdnaimislam165436-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/mdnaimislam165436)
&nbsp;
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Md.%20Naim%20Islam-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/md-naim-00a164381/)
&nbsp;
[![Gmail](https://img.shields.io/badge/Gmail-naim.cse2004@gmail.com-EA4335?style=for-the-badge&logo=gmail&logoColor=white)](mailto:naim.cse2004@gmail.com)

---

## License

This project is open source and available under the [MIT License](LICENSE).

---

<p align="center">
  If this project helped you, please consider giving it a star on GitHub!
</p>

<p align="center">
  Made with dedication by <b>Md. Naim Islam</b>
</p>
