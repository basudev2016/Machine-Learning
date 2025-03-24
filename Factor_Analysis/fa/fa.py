import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler

# Load dataset
dsn = pd.read_csv("product_selection_dataset.csv")

# Separating features and target
X = dsn.drop(columns=['Product_Choice'])

# Encoding categorical variables if any
X = pd.get_dummies(X, drop_first=True)

# Standardizing the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Performing Factor Analysis
fa = FactorAnalysis(n_components=3, random_state=42)
X_factors = fa.fit_transform(X_scaled)

# Displaying factor loadings
factor_loadings = pd.DataFrame(fa.components_, columns=X.columns, index=["Factor 1", "Factor 2", "Factor 3"])
print("\nFactor Loadings:")
print(factor_loadings)

# Visualizing factor loadings
plt.figure(figsize=(10, 6))
sns.heatmap(factor_loadings, annot=True, cmap='coolwarm', center=0)
plt.title("Factor Loadings Heatmap")
plt.xlabel("Features")
plt.ylabel("Factors")
plt.show()

# Explained Variance (approximated by sum of squared loadings)
explained_variance = np.sum(factor_loadings ** 2, axis=1)
explained_variance_ratio = explained_variance / np.sum(explained_variance) * 100
print("@@@@@@@@@@@@@@@",explained_variance_ratio)

print("\nExplained Variance by Each Factor:")
for i, var in enumerate(explained_variance_ratio):
    print(f"Factor {i+1}: {var:.2f}% variance explained")