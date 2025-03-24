import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
dsn = pd.read_csv("product_selection_dataset.csv")

# Separating features and target
X = dsn.drop(columns=['Product_Choice'])
y = dsn['Product_Choice']

# Encoding categorical variables if any
X = pd.get_dummies(X, drop_first=True)

# Standardizing the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Performing Factor Analysis
fa = FactorAnalysis(n_components=3, random_state=42)
X_factors = fa.fit_transform(X_scaled)

# Splitting into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_factors, y, test_size=0.2, random_state=42)

# Training Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Making predictions
y_pred = rf_model.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Displaying results
print("Random Forest Model Accuracy After Factor Analysis:", accuracy)
print("\nConfusion Matrix After Factor Analysis:\n", conf_matrix)
print("\nClassification Report After Factor Analysis:\n", class_report)
