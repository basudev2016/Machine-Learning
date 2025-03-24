# -------------------------------
# Association Rule Mining with Apriori
# -------------------------------

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# STEP 1: Load the dataset
print("\n📥 STEP 1: Loading the dataset...")
df = pd.read_csv("Healthcare_Symptom_Association.csv")
print(f"✅ Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")
print(df.head())

# STEP 2: Drop patient_ID column if present
if 'Patient_ID' in df.columns:
    print("\n🧹 STEP 2: Dropping 'patient_ID' column (not needed for Apriori)...")
    df = df.drop(columns=['Patient_ID'])
else:
    print("\n🧹 STEP 2: 'patient_ID' column not found — skipping drop.")

# STEP 3: Apply Apriori Algorithm
print("\n⚙️ STEP 3: Running Apriori to find frequent itemsets...")
frequent_itemsets = apriori(df, min_support=0.1, use_colnames=True)
print(f"✅ Found {len(frequent_itemsets)} frequent itemsets (support ≥ 0.1):")
print(frequent_itemsets.head())

# STEP 4: Generate Association Rules
print("\n🔍 STEP 4: Generating association rules (confidence ≥ 0.5)...")
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
print(f"✅ Generated {len(rules)} rules.")
print("📝 Sample rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())

# STEP 5: Explain lift, confidence, support 
print("\n📚 RULE METRIC EXPLANATIONS:")
print("• Support(A → B) = P(A ∪ B) = Fraction of transactions containing both A and B")
print("• Confidence(A → B) = P(B|A) = Fraction of transactions with A that also have B")
print("• Lift(A → B) = Confidence(A → B) / P(B)")
print("  - Lift > 1: Positive correlation between A and B")
print("  - Lift = 1: A and B are independent")
print("  - Lift < 1: A and B are negatively correlated")

# STEP 6: Filter strong rules
print("\n📊 STEP 6: Filtering strong rules (lift > 1.2 and confidence > 0.6)...")
strong_rules = rules[(rules['lift'] > 1.2) & (rules['confidence'] > 0.6)]
print(f"✅ Found {len(strong_rules)} strong rules:")
print(strong_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# STEP 7: Save results
print("\n💾 STEP 7: Saving strong rules to CSV file...")
strong_rules.to_csv("Healthcare_Strong_Association_Rules.csv", index=False)
print("🎉 DONE: Rules saved as 'Healthcare_Strong_Association_Rules.csv'")
