# Customer Retention Model: Model Fitness Analysis

---

## 🌟 **Introduction**

Model Fitness, a gym chain, aims to improve customer retention by analyzing factors that influence churn and formulating data-driven strategies. The main objectives of this project are:

- **Analyze customer profiles** to understand behavioral patterns.
- **Predict churn probability** using classification models.
- **Segment customers** to design personalized strategies.
- **Recommend actions based on the findings.**

---

## 📊 **Step 1: Data Exploration and Preparation**

### 1.1 **Libraries and Dataset Loading**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage

# Load dataset
df = pd.read_csv('/datasets/gym_churn_us.csv')
```

### 1.2 **Initial Dataset Review**
- **Size and Columns:** 4000 records, 14 columns.
- **Relevant Columns:** Gender, Age, Contract Duration, Average Additional Charges, Visit Frequency, among others.
- **Complete Data:** No missing or duplicate values.

```python
print(df.info())
print(df.describe())
```

---

## 🌄 **Step 2: Exploratory Analysis**

### 2.1 **Variable Distribution**

#### Discrete Variables
- Customers near the location are less likely to churn.
- Longer contracts are linked to lower churn rates.

#### Continuous Variables
- **Age:** Younger customers are more likely to churn.
- **Average Additional Charges:** Higher additional spending is associated with retention.
- **Class Frequency:** Customers with higher attendance show lower churn rates.

```python
# Histograms for continuous variables
for column in ['Age', 'Avg_additional_charges_total']:
    sns.histplot(data=df, x=column, hue='Churn', kde=True)
    plt.show()
```

### 2.2 **Correlation Matrix**
- **Key Observations:**
  - Variables like "Contract Duration" and "Visit Frequency" have a strong negative correlation with "Churn".

```python
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()
```

---

## 🔧 **Step 3: Predictive Models**

### 3.1 **Data Splitting and Scaling**
```python
X = df.drop(columns=['Churn'])
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_st = scaler.fit_transform(X_train)
X_test_st = scaler.transform(X_test)
```

### 3.2 **Model Evaluation**
#### Logistic Regression
```python
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_st, y_train)
lr_predictions = lr_model.predict(X_test_st)
evaluate_model(y_test, lr_predictions)
```

#### Random Forest
```python
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
evaluate_model(y_test, rf_predictions)
```

### 3.3 **Evaluation Results**
| Model                | Accuracy | Precision | Recall |
|-----------------------|-----------|-------------|--------------|
| Logistic Regression | 0.92      | 0.87        | 0.78         |
| Random Forest       | 0.91      | 0.85        | 0.78         |

---

## 🌿 **Step 4: Segmentation Analysis**

### 4.1 **Clustering with K-Means**
- **Cluster Configuration:** 5 clusters were defined using the "Elbow Method".
- **Cluster Analysis:**
  - Customers with longer contracts and higher frequency are less likely to churn.
  - Clusters with higher churn have lower group class participation and additional spending.

```python
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)
```

---

## 🌟 **Conclusion and Recommendations**

### Key Observations
1. **Visit Frequency:** Customers with higher attendance are less likely to churn.
2. **Contract Duration:** Longer contracts are associated with lower churn.
3. **Segmentation:** Clusters with lower churn exhibit high class participation and additional spending.

### Recommendations
1. **Encourage Long-Term Contracts:** Offer discounts or exclusive benefits for 6 or 12-month contracts.
2. **Promote Group Classes:** Increase offerings and run promotions.
3. **Effective Segmentation:** Personalize strategies based on identified clusters.
4. **Proactive Monitoring:** Identify at-risk customers and provide incentives for retention.

---

**Date: 2025-01-06**  
**Francisco SLG**

