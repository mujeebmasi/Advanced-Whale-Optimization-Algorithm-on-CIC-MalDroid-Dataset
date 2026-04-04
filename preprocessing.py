import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ─── LOAD ───
print("Loading CIC-MalDroid-2020...")
df = pd.read_csv('feature_vectors_syscallsbinders_frequency_5_Cat.csv', low_memory=False)
print(f"Original shape: {df.shape}")

class_names = {1: 'Benign', 2: 'Ransomware', 3: 'Adware', 4: 'Scareware', 5: 'SMS Malware'}

print("\nClass distribution:")
for cls, name in class_names.items():
    count = (df['Class'] == cls).sum()
    print(f"  Class {cls} - {name}: {count}")

# ─── STRATIFIED SUBSET OF 10,000 ───
print("\nSampling 10,000 rows (stratified)...")
df_sub = df.groupby('Class', group_keys=False).apply(
    lambda x: x.sample(min(len(x), max(1, int(10000 * len(x) / len(df)))), random_state=42)
).reset_index(drop=True)
print(f"Subset shape: {df_sub.shape}")

X = df_sub.drop(columns=['Class'])
y_orig = df_sub['Class'].values

# ─── CLEAN ───
X = X.fillna(X.median())
X = X.loc[:, X.std() > 0]
print(f"Features after cleaning: {X.shape[1]}")

# ─── NORMALIZE ───
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
print(f"Normalization done. Range: [{X_scaled.values.min():.3f}, {X_scaled.values.max():.3f}]")

# ─── BINARY LABELS: 0=Benign, 1=Malware ───
y_binary = np.where(y_orig == 1, 0, 1)
print(f"\nBinary — Benign: {np.sum(y_binary==0)}, Malware: {np.sum(y_binary==1)}")

# ─── SMOTE ───
print("Applying SMOTE...")
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_scaled, y_binary)
print(f"After SMOTE — Benign: {np.sum(y_res==0)}, Malware: {np.sum(y_res==1)}")

# ─── EDA PLOTS ───
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].bar([class_names[k] for k in sorted(class_names)],
            [np.sum(y_orig == k) for k in sorted(class_names)],
            color=['#4CAF50','#F44336','#FF9800','#2196F3','#9C27B0'])
axes[0].set_title('Original 5-Class Distribution')
axes[0].tick_params(axis='x', rotation=15)

axes[1].bar(['Benign','Malware'], np.bincount(y_res),
            color=['#4CAF50','#F44336'])
axes[1].set_title('After SMOTE (Binary)')

variances = pd.DataFrame(X_res, columns=X.columns).var()
axes[2].hist(variances, bins=40, color='#2196F3', edgecolor='white')
axes[2].set_title('Feature Variance Distribution')
axes[2].set_xlabel('Variance')

plt.tight_layout()
plt.savefig('eda_plots.png', dpi=150, bbox_inches='tight')
plt.show()
print("EDA plots saved.")

# ─── SAVE ───
out = pd.DataFrame(X_res, columns=X.columns)
out['label'] = y_res
out.to_csv('preprocessed_data.csv', index=False)
print(f"\n✅ Saved preprocessed_data.csv  shape: {out.shape}")
print("Features:", X_res.shape[1], "| Samples:", X_res.shape[0])
print("\nNext → python step2_cnn_ewoa.py")