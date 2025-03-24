import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# STEP 1: Load the dataset
print("\n📥 STEP 1: Loading the retail transaction dataset...")
df = pd.read_csv("MBA_Retail_Dataset.csv")
print(f"✅ Loaded dataset with {df.shape[0]} transactions.")
print("🧾 First few rows of the dataset:")
print(df.head())

# STEP 2: Drop non-item columns (Transaction_ID, Transaction list)
print("\n🧹 STEP 2: Removing non-item columns for Apriori analysis...")
df_items = df.drop(columns=['Transaction_ID', 'Transaction'])
print(f"✅ Retained columns: {list(df_items.columns)}")

# STEP 3: Generate frequent itemsets using Apriori
print("\n⚙️ STEP 3: Running Apriori algorithm (min_support=0.2)...")
frequent_itemsets = apriori(df_items, min_support=0.2, use_colnames=True)
print(f"✅ Found {len(frequent_itemsets)} frequent itemsets:")
print(frequent_itemsets)

# STEP 4: Generate association rules
print("\n🔍 STEP 4: Generating association rules (min_confidence=0.5)...")
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
print(f"✅ Generated {len(rules)} rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# STEP 5: Filter strong rules
print("\n📊 STEP 5: Filtering strong rules (lift > 1 and confidence > 0.6)...")
strong_rules = rules[(rules['lift'] > 1.0) & (rules['confidence'] > 0.6)]
print(f"✅ {len(strong_rules)} strong rules found:")
print(strong_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# STEP 6: Save the results
print("\n💾 STEP 6: Saving strong rules to 'Retail_Strong_Association_Rules.csv'...")
strong_rules.to_csv("Retail_Strong_Association_Rules.csv", index=False)
print("🎉 DONE! You can now inspect the rule file for actionable retail insights.")

# STEP 7: Explain metrics
print("\n📚 METRIC RECAP:")
print("• Support = P(A ∩ B): How often A and B appear together")
print("• Confidence = P(B|A): Likelihood of B given A is purchased")
print("• Lift = Confidence / P(B): How much more likely B is with A compared to random chance")
