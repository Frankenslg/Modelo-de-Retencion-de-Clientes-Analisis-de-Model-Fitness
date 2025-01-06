# Modelo de Retención de Clientes: Análisis de Model Fitness

---

## 🌟 **Introducción**

Model Fitness, una cadena de gimnasios, busca mejorar la retención de clientes analizando factores que influyen en la rotación ("churn") y formulando estrategias basadas en datos. Este proyecto tiene como objetivos principales:

- **Analizar perfiles de clientes** para entender patrones de comportamiento.
- **Predecir la probabilidad de rotación** utilizando modelos de clasificación.
- **Segmentar clientes** para diseñar estrategias personalizadas.
- **Recomendar acción basadas en los resultados obtenidos.**

---

## 📊 **Paso 1: Exploración y Preparación de Datos**

### 1.1 **Carga de Librerías y Dataset**
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

# Carga del dataset
df = pd.read_csv('/datasets/gym_churn_us.csv')
```

### 1.2 **Revisión Inicial del Dataset**
- **Tamaño y Columnas:** 4000 registros, 14 columnas.
- **Columnas relevantes:** Género, Edad, Duración del contrato, Gasto adicional promedio, Frecuencia de visitas, entre otras.
- **Datos completos:** No se encontraron valores nulos ni duplicados.

```python
print(df.info())
print(df.describe())
```

---

## 🌄 **Paso 2: Análisis Exploratorio**

### 2.1 **Distribución de Variables**

#### Variables Discretas
- Los clientes cerca de la ubicación tienen menor probabilidad de cancelar.
- Contratos más largos están vinculados a menores tasas de cancelación.

#### Variables Continuas
- **Edad:** Los clientes jóvenes tienen mayor probabilidad de cancelar.
- **Gasto Adicional Promedio:** Un mayor gasto adicional está asociado con retención.
- **Frecuencia de Clases:** Clientes con mayor asistencia muestran menor probabilidad de cancelar.

```python
# Histogramas de variables continuas
for column in ['Age', 'Avg_additional_charges_total']:
    sns.histplot(data=df, x=column, hue='Churn', kde=True)
    plt.show()
```

### 2.2 **Matriz de Correlación**
- **Observaciones Clave:**
  - Las variables "Duración del Contrato" y "Frecuencia de Visitas" tienen una fuerte correlación negativa con "Churn".

```python
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()
```

---

## 🔧 **Paso 3: Modelos Predictivos**

### 3.1 **División de Datos y Escalado**
```python
X = df.drop(columns=['Churn'])
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_st = scaler.fit_transform(X_train)
X_test_st = scaler.transform(X_test)
```

### 3.2 **Evaluación de Modelos**
#### Regresión Logística
```python
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_st, y_train)
lr_predictions = lr_model.predict(X_test_st)
evaluate_model(y_test, lr_predictions)
```

#### Bosque Aleatorio
```python
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
evaluate_model(y_test, rf_predictions)
```

### 3.3 **Resultados de Evaluación**
| Modelo                | Exactitud | Precisión | Sensibilidad |
|-----------------------|-----------|-------------|--------------|
| Regresión Logística | 0.92      | 0.87        | 0.78         |
| Bosque Aleatorio      | 0.91      | 0.85        | 0.78         |

---

## 🌿 **Paso 4: Análisis de Segmentación**

### 4.1 **Clustering con K-Means**
- **Configuración de Clústeres:** Se definieron 5 clústeres utilizando el método de "Elbow".
- **Análisis de clústeres:**
  - Clientes con contratos largos y alta frecuencia tienen menor probabilidad de cancelar.
  - Los clústeres con mayor "Churn" tienen menor participación en clases grupales y menos gasto adicional.

```python
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)
```

---

## 🌟 **Conclusión y Recomendaciones**

### Observaciones Principales
1. **Frecuencia de Visitas:** Los clientes con mayor asistencia tienen menor probabilidad de cancelar.
2. **Duración del Contrato:** Contratos largos están asociados a menor "Churn".
3. **Segmentación:** Los clústeres con menor "Churn" muestran alta participación en clases y gasto adicional.

### Recomendaciones
1. **Incentivar Contratos Largos:** Ofrecer descuentos o beneficios exclusivos para contratos de 6 o 12 meses.
2. **Promover Clases Grupales:** Incrementar la oferta y hacer promociones.
3. **Segmentación Efectiva:** Personalizar estrategias basadas en los clústeres identificados.
4. **Monitoreo Proactivo:** Identificar clientes en riesgo y ofrecer incentivos para su retención.

---

**Fecha: 2025-01-06**
**Francisco SLG**



