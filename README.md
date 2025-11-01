# 🧠 Credit Risk Modelling using Machine Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-FF4B4B.svg)](https://streamlit.io)
[![Optuna](https://img.shields.io/badge/Optuna-3.0+-00ADD8.svg)](https://optuna.org)

This project implements an enterprise-grade credit risk assessment system that predicts loan default probability using advanced machine learning techniques, comprehensive statistical analysis, and sophisticated feature engineering methodologies.

### The Challenge
- Traditional credit scoring systems lack flexibility and fail to capture complex patterns
- Manual feature engineering is time-consuming and may miss important relationships
- Hyperparameter tuning requires extensive expertise and computational resources
- Model interpretability is crucial for regulatory compliance and business decisions

### Our Solution
A state-of-the-art ML pipeline featuring:
- *Comprehensive EDA*: Statistical tests, distribution analysis, and correlation studies
- *Advanced Feature Engineering*: WoE (Weight of Evidence) and IV (Information Value) transformation
- *Multiple Optimization Strategies*: Optuna, GridSearchCV, and RandomizedSearchCV
- *Statistical Validation*: KS (Kolmogorov-Smirnov) statistic for model performance
- *Multiple Encoding Techniques*: One-hot, label, target, frequency, and WoE encoding
- *Production Deployment*: Interactive Streamlit dashboard for real-time predictions

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Streamlit-FF4B4B?style=for-the-badge)](https://credit-risk-modeling-ml.streamlit.app/)

## ✨ Key Features

<table>
<tr>
<td width="50%">

### 📊 Exploratory Data Analysis
- *Statistical Analysis*
  - Descriptive statistics
  - Distribution analysis (skewness, kurtosis)
  - Normality tests (Shapiro-Wilk, Anderson-Darling)
  - Chi-square tests for categorical associations
- *Visualization*
  - Univariate analysis (histograms, box plots)
  - Bivariate analysis (scatter plots, correlation heatmaps)
  - Multivariate analysis (pair plots, 3D visualizations)
- *Missing Value Analysis*
  - Missing data patterns
  - Imputation strategies
  - Impact assessment

</td>
<td width="50%">

### 🎨 Advanced Feature Engineering
- *WoE & IV Transformation*
  - Weight of Evidence (WoE) encoding
  - Information Value (IV) calculation
  - Optimal binning for continuous variables
  - Risk score generation
- *Derived Features*
  - Financial ratios (debt-to-income, credit utilization)
  - Aggregated bureau metrics
  - Temporal features
  - Interaction terms
- *Feature Selection*
  - IV-based selection (IV > 0.02 threshold)
  - Correlation filtering
  - Variance threshold
  - Recursive Feature Elimination (RFE)

</td>
</tr>
<tr>
<td width="50%">

### ⚡ Hyperparameter Optimization
- *Optuna*
  - Bayesian optimization with TPE
  - Multi-objective optimization
  - Pruning for efficiency
  - Visualization of optimization history
- *GridSearchCV*
  - Exhaustive parameter search
  - Cross-validation integration
  - Best parameter extraction
- *RandomizedSearchCV*
  - Random sampling from distributions
  - Time-efficient exploration
  - Broad search space coverage
- *Comparison & Selection*
  - Performance comparison across methods
  - Time vs accuracy trade-offs

</td>
<td width="50%">

### 🔍 Multiple Encoding Strategies
- *One-Hot Encoding*
  - For low-cardinality nominal features
  - Dummy variable creation
- *Label Encoding*
  - For ordinal features
  - Maintains order relationships
- *Target Encoding*
  - Mean target encoding
  - Smoothing techniques
  - Cross-validation to prevent overfitting
- *Frequency Encoding*
  - Category frequency mapping
  - Captures prevalence patterns
- *WoE Encoding*
  - Log-odds transformation
  - Strong predictive power for credit risk
  - Business interpretability

</td>
</tr>
<tr>
<td width="50%">

### 📈 Statistical Model Validation
- *KS Statistic*
  - Kolmogorov-Smirnov test
  - Measures separation between classes
  - KS > 0.4 indicates good model
- *Performance Metrics*
  - Accuracy, Precision, Recall, F1-Score
  - ROC-AUC and Precision-Recall AUC
  - Gini coefficient
  - Confusion matrix analysis
- *Cross-Validation*
  - K-Fold validation
  - Stratified K-Fold for imbalanced data
  - Time-series split for temporal data

</td>
<td width="50%">

### 🚀 Production Deployment
- *Streamlit Application*
  - Interactive web interface
  - Real-time predictions
  - User-friendly input forms
  - Visual result presentation
- *Model Persistence*
  - Joblib serialization
  - Efficient model loading
  - Version control
- *Scalability*
  - Modular code structure
  - Easy integration with APIs
  - Cloud deployment ready

</td>
</tr>
</table>

---

## 📊 Dataset

The project integrates three interconnected datasets for comprehensive credit risk analysis:

### Data Sources

| Dataset | Description |
|---------|-------------|
| *bureau_data.csv* | Credit bureau data with payment history, credit utilization, and delinquency records |
| *customers.csv* | Customer demographics including income, employment, and personal details |
| *loans.csv* | Loan details with amount, term, interest rate, and default status (target variable) |

### Key Features by Category

<details>
<summary><b>📋 Bureau Data Features</b></summary>

*Credit History:*
- Credit account age
- Number of active/closed accounts
- Credit mix (mortgage, auto, personal loans)

*Payment Behavior:*
- Payment history (on-time vs late)
- Number of delinquencies (30, 60, 90+ days)
- Derogatory marks

*Credit Utilization:*
- Total credit limit
- Total outstanding balance
- Utilization ratio

*Inquiries:*
- Hard inquiry count (last 6 months)
- Soft inquiry count

</details>

<details>
<summary><b>👤 Customer Data Features</b></summary>

*Demographics:*
- Age
- Gender
- Marital status
- Number of dependents

*Employment:*
- Employment status
- Employment length
- Job title/category
- Income stability

*Financial:*
- Annual income
- Monthly income
- Income source

*Geographic:*
- State/City
- ZIP code
- Urban/Rural classification

</details>

<details>
<summary><b>💰 Loan Data Features</b></summary>

*Loan Details:*
- Loan amount requested
- Loan term (months)
- Interest rate
- Monthly installment

*Loan Characteristics:*
- Loan purpose (debt consolidation, home improvement, etc.)
- Loan grade/sub-grade
- Application type

*Status:*
- Loan status (current, fully paid, charged off)
- *Target Variable*: Default (0 = No Default, 1 = Default)

*Dates:*
- Application date
- Issue date
- Last payment date

</details>
