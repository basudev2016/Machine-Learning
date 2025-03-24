import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.cluster import KMeans

# Load transaction dataset
df = pd.read_csv("BFSI_Transactions.csv")

# Convert TransactionDate to datetime
df["TransactionDate"] = pd.to_datetime(df["TransactionDate"])

# Get the most recent transaction date
latest_date = df["TransactionDate"].max()

# Compute RFM metrics while ensuring CustomerID is preserved
rfm_data = df.groupby("CustomerID").agg({
    "TransactionDate": lambda x: (latest_date - x.max()).days,  # Recency
    "CustomerID": "count",  # Frequency
    "TransactionAmount": "sum"  # Monetary
}).rename(columns={
    "TransactionDate": "Recency",
    "CustomerID": "Frequency",
    "TransactionAmount": "Monetary"
}).reset_index()  # Reset index to retain original CustomerID

# Plot and save RFM distributions
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

sns.histplot(rfm_data["Recency"], bins=30, kde=True, ax=axes[0])
axes[0].set_title("Recency Distribution")

sns.histplot(rfm_data["Frequency"], bins=30, kde=True, ax=axes[1])
axes[1].set_title("Frequency Distribution")

sns.histplot(rfm_data["Monetary"], bins=30, kde=True, ax=axes[2])
axes[2].set_title("Monetary Distribution")

plt.tight_layout()
plt.savefig("RFM_Distribution.png")  # Save the plot
plt.show()

# Assign RFM scores based on quartiles (1 to 5)
rfm_data["R_Score"] = pd.qcut(rfm_data["Recency"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
rfm_data["F_Score"] = pd.qcut(rfm_data["Frequency"], 5, labels=[1, 2, 3, 4, 5]).astype(int)
rfm_data["M_Score"] = pd.qcut(rfm_data["Monetary"], 5, labels=[1, 2, 3, 4, 5]).astype(int)

# Compute Concatenation-based RFM Score
rfm_data["Concatenated_RFM_Score"] = (
    rfm_data["R_Score"].astype(str) + 
    rfm_data["F_Score"].astype(str) + 
    rfm_data["M_Score"].astype(str)
)

# Compute Weighted RFM Score (Adjust weights based on business importance)
rfm_data["Weighted_RFM_Score"] = (
    0.3 * rfm_data["R_Score"] +
    0.3 * rfm_data["F_Score"] +
    0.4 * rfm_data["M_Score"]
)

# Apply K-Means Clustering for automated segmentation
rfm_clustering = rfm_data[["R_Score", "F_Score", "M_Score"]]
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
rfm_data["KMeans_Cluster"] = kmeans.fit_predict(rfm_clustering)

# Compute RFM Percentile Score (for continuous ranking)
rfm_data["RFM_Percentile"] = (
    rfm_data["R_Score"].rank(pct=True) +
    rfm_data["F_Score"].rank(pct=True) +
    rfm_data["M_Score"].rank(pct=True)
)

# Classify customers into segments based on Weighted RFM Score
def classify_customer(score):
    if score >= 4.0:
        return "Best Customers"
    elif score >= 3.0:
        return "Loyal Customers"
    elif score >= 2.0:
        return "At-Risk Customers"
    else:
        return "Lost Customers"

rfm_data["CustomerSegment"] = rfm_data["Weighted_RFM_Score"].apply(classify_customer)

# Ensure CustomerID is properly included in all outputs
rfm_data = rfm_data.sort_values(by="CustomerID")

# Save separate CSV files with preserved CustomerID
rfm_data[["CustomerID", "R_Score", "F_Score", "M_Score", "Concatenated_RFM_Score"]].to_csv(
    "Customer_RFM_Concatenation.csv", index=False)

rfm_data[["CustomerID", "R_Score", "F_Score", "M_Score", "Weighted_RFM_Score", "CustomerSegment"]].to_csv(
    "Customer_RFM_Segmentation.csv", index=False)

rfm_data[["CustomerID", "KMeans_Cluster"]].to_csv(
    "Customer_KMeans_Clustering.csv", index=False)

rfm_data[["CustomerID", "RFM_Percentile"]].to_csv(
    "Customer_RFM_Percentile.csv", index=False)

print("✅ RFM Analysis Completed")
print("📊 RFM Distribution Plot Saved as 'RFM_Distribution.png'")
print("📂 Concatenation-based RFM Saved as 'Customer_RFM_Concatenation.csv'")
print("📂 Weighted RFM Segmentation Saved as 'Customer_RFM_Segmentation.csv'")
print("📂 K-Means Clustering Data Saved as 'Customer_KMeans_Clustering.csv'")
print("📂 RFM Percentile Data Saved as 'Customer_RFM_Percentile.csv'")
