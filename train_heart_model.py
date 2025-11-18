import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

print("="*60)
print("TRAINING HEART DISEASE PREDICTION MODEL")
print("="*60)

# Set random seed for reproducibility
np.random.seed(42)
n_samples = 1000

print(f"\n✓ Generating {n_samples} synthetic patient records...")

# Create synthetic heart disease dataset
data = {
    'age': np.random.randint(30, 80, n_samples),
    'sex': np.random.randint(0, 2, n_samples),
    'cp': np.random.randint(0, 4, n_samples),
    'trestbps': np.random.randint(90, 200, n_samples),
    'chol': np.random.randint(120, 400, n_samples),
    'fbs': np.random.randint(0, 2, n_samples),
    'restecg': np.random.randint(0, 3, n_samples),
    'thalach': np.random.randint(70, 200, n_samples),
    'exang': np.random.randint(0, 2, n_samples),
    'oldpeak': np.random.uniform(0, 6, n_samples).round(1),
    'slope': np.random.randint(0, 3, n_samples),
    'ca': np.random.randint(0, 4, n_samples),
    'thal': np.random.randint(0, 4, n_samples)
}

df = pd.DataFrame(data)

print("✓ Data generated successfully")
print(f"  Shape: {df.shape}")

# Create target variable based on risk factors
# Initialize all as 0 (no disease)
df['target'] = 0

# Define high risk conditions
high_risk_mask = (
    (df['age'] > 60) |
    (df['chol'] > 280) |
    (df['trestbps'] > 150) |
    (df['thalach'] < 100) |
    (df['cp'] == 3) |
    (df['exang'] == 1)
)

# Set high risk patients to 1
df.loc[high_risk_mask, 'target'] = 1

# Add some randomness to make it more realistic
# Flip some labels randomly
random_flip_indices = np.random.choice(df.index, size=int(0.15 * len(df)), replace=False)
df.loc[random_flip_indices, 'target'] = 1 - df.loc[random_flip_indices, 'target']

print(f"✓ Target variable created")
print(f"  Disease cases: {df['target'].sum()} ({df['target'].sum()/len(df)*100:.1f}%)")
print(f"  No disease cases: {(df['target']==0).sum()} ({(df['target']==0).sum()/len(df)*100:.1f}%)")

# Check for NaN values
if df.isnull().any().any():
    print("\n⚠️  Warning: Found NaN values. Removing them...")
    df = df.dropna()
    print(f"  New shape: {df.shape}")
else:
    print("✓ No NaN values found")

# Prepare features and target
X = df.drop('target', axis=1)
y = df['target']

print(f"\n✓ Splitting data into train and test sets...")
# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"  Training samples: {len(X_train)}")
print(f"  Testing samples: {len(X_test)}")

# Scale features
print(f"\n✓ Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest model
print(f"\n✓ Training Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100, 
    random_state=42, 
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2
)

model.fit(X_train_scaled, y_train)

# Evaluate
train_score = model.score(X_train_scaled, y_train)
test_score = model.score(X_test_scaled, y_test)

print(f"\n{'='*60}")
print("MODEL TRAINING COMPLETE!")
print(f"{'='*60}")
print(f"Training Accuracy: {train_score*100:.2f}%")
print(f"Testing Accuracy: {test_score*100:.2f}%")

# Save model and scaler
print(f"\n✓ Saving model files...")
with open('heart_disease_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print(f"✓ Files saved successfully!")
print(f"  - heart_disease_model.pkl")
print(f"  - scaler.pkl")

# Feature importance
print(f"\n{'='*60}")
print("TOP 5 MOST IMPORTANT FEATURES:")
print(f"{'='*60}")
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for idx, row in feature_importance.head(5).iterrows():
    print(f"  {row['feature']:15s} : {row['importance']:.4f}")

print(f"\n{'='*60}")
print("✅ MODEL READY TO USE!")
print(f"{'='*60}")
print("\nNext steps:")
print("  1. Run: python app.py")
print("  2. Open: http://127.0.0.1:5000/")
print(f"{'='*60}\n")