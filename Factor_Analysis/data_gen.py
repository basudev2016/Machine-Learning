import pandas as pd
import numpy as np
import random

# Setting seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 500

# Generating product choices
product_choices = ['P1', 'P2', 'P3']

# Creating features that influence product selection
age = np.random.randint(18, 65, n_samples)  # Age of the buyer
income = np.random.randint(20000, 150000, n_samples)  # Annual income
brand_loyalty = np.random.randint(1, 10, n_samples)  # Loyalty score (1-10)
price_sensitivity = np.random.randint(1, 10, n_samples)  # Price sensitivity (1-10)
product_reviews = np.random.randint(1, 10, n_samples)  # Average product rating from online sources (1-10)
advertisement_exposure = np.random.randint(1, 10, n_samples)  # Frequency of ads seen (1-10)
discount_affinity = np.random.randint(1, 10, n_samples)  # Preference for discounts (1-10)
feature_importance = np.random.randint(1, 10, n_samples)  # Importance of product features (1-10)
peer_recommendation = np.random.randint(1, 10, n_samples)  # Influence of peer recommendations (1-10)

# Logic to influence product choice
choices = []
for i in range(n_samples):
    if brand_loyalty[i] > 7 and advertisement_exposure[i] > 7:
        choices.append('P1')  # Strong brand recognition and high ad exposure favor P1
    elif price_sensitivity[i] > 6 and discount_affinity[i] > 6:
        choices.append('P2')  # Price-sensitive buyers tend to choose P2
    elif feature_importance[i] > 7 and peer_recommendation[i] > 7:
        choices.append('P3')  # Feature-conscious buyers prefer P3
    else:
        choices.append(random.choice(product_choices))  # Random selection for mixed cases

# Creating DataFrame
df_product_selection = pd.DataFrame({
    'Age': age,
    'Income': income,
    'Brand_Loyalty': brand_loyalty,
    'Price_Sensitivity': price_sensitivity,
    'Product_Reviews': product_reviews,
    'Advertisement_Exposure': advertisement_exposure,
    'Discount_Affinity': discount_affinity,
    'Feature_Importance': feature_importance,
    'Peer_Recommendation': peer_recommendation,
    'Product_Choice': choices  # Target variable
})

# Saving to CSV
df_product_selection.to_csv("product_selection_dataset.csv", index=False)

print("Dataset successfully created and saved as 'product_selection_dataset.csv'.")
