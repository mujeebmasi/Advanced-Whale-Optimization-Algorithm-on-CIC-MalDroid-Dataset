import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import shap
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("  CIC-MalDroid-2020: PCA + EWOA + XGBoost + SHAP")
print("="*60)

# ════════════════════════════════════════════════
# LOAD
# ════════════════════════════════════════════════
df = pd.read_csv('preprocessed_data.csv')
y  = df['label'].values
X  = df.drop(columns=['label']).values

print(f"\n✅ Loaded: {X.shape[0]} samples, {X.shape[1]} features")
print(f"   Benign: {np.sum(y==0)} | Malware: {np.sum(y==1)}")

if len(X) > 8000:
    idx = np.random.RandomState(42).choice(len(X), 8000, replace=False)
    X, y = X[idx], y[idx]
    print(f"   Subsampled to 8000")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Train: {len(X_train)} | Test: {len(X_test)}")

# ════════════════════════════════════════════════
# STEP 2 — PCA (fast feature extraction)
# ════════════════════════════════════════════════
print("\n[STEP 2] PCA Dimensionality Reduction...")
pca          = PCA(n_components=50, random_state=42)
X_train_pca  = pca.fit_transform(X_train)
X_test_pca   = pca.transform(X_test)
explained    = np.cumsum(pca.explained_variance_ratio_)[-1] * 100
print(f"✅ {X_train.shape[1]} features → 50 PCA components ({explained:.1f}% variance retained)")

plt.figure(figsize=(8, 4))
plt.plot(np.cumsum(pca.explained_variance_ratio_)*100, color='#1F4E79', linewidth=2)
plt.axhline(y=95, color='red', linestyle='--', label='95% threshold')
plt.xlabel('Number of Components'); plt.ylabel('Cumulative Variance (%)')
plt.title('PCA — Explained Variance'); plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('pca_variance.png', dpi=150, bbox_inches='tight')
plt.show(); print("PCA plot saved.")

# ════════════════════════════════════════════════
# STEP 3 — EWOA
# ════════════════════════════════════════════════
print("\n[STEP 3] Enhanced Whale Optimisation Algorithm (EWOA)...")

class EWOA:
    def __init__(self, n_whales=10, max_iter=20, alpha=0.80, beta=0.20, max_features=10):
        self.n_whales = n_whales
        self.max_iter = max_iter
        self.alpha    = alpha
        self.beta     = beta
        self.max_features = max_features

    def _enforce_limit(self, sol):
        ones = np.where(sol > 0.5)[0]
        if len(ones) <= self.max_features:
            return sol
        keep = np.random.choice(ones, self.max_features, replace=False)
        constrained = np.zeros_like(sol)
        constrained[keep] = 1.0
        return constrained

    def _fitness(self, sol, X_tr, X_te, y_tr, y_te):
        sol = self._enforce_limit(sol)
        idx = np.where(sol > 0.5)[0]
        if len(idx) == 0:
            return 10.0
        clf = XGBClassifier(n_estimators=30, max_depth=4,
                            random_state=42, eval_metric='logloss',
                            verbosity=0, n_jobs=-1)
        clf.fit(X_tr[:, idx], y_tr)
        err   = 1 - accuracy_score(y_te, clf.predict(X_te[:, idx]))
        return self.alpha * err + self.beta * (len(idx) / X_tr.shape[1])

    def optimize(self, X_tr, X_te, y_tr, y_te):
        dim     = X_tr.shape[1]
        pop     = np.random.randint(0, 2, (self.n_whales, dim)).astype(float)
        pop     = np.array([self._enforce_limit(s) for s in pop])
        obl     = 1 - pop
        obl     = np.array([self._enforce_limit(s) for s in obl])
        all_pop = np.vstack([pop, obl]) 

        print("  Evaluating initial population (OBL)...")
        fits_all = np.array([self._fitness(s, X_tr, X_te, y_tr, y_te) for s in all_pop])
        top      = np.argsort(fits_all)[:self.n_whales]
        pop, fits = all_pop[top], fits_all[top]

        best_pos = pop[np.argmin(fits)].copy()
        best_fit = fits.min()
        history  = []

        for t in range(self.max_iter):
            a  = 2 - 2 * (t / self.max_iter)
            a2 = -1 - (t / self.max_iter)
            for i in range(self.n_whales):
                A = 2 * a * np.random.rand() - a
                C = 2 * np.random.rand()
                p = np.random.rand()
                l = (a2 - 1) * np.random.rand() + 1
                if p < 0.5:
                    if abs(A) < 1:
                        new_pos = best_pos - A * abs(C * best_pos - pop[i])
                    else:
                        rw      = pop[np.random.randint(self.n_whales)]
                        new_pos = rw - A * abs(C * rw - pop[i])
                else:
                    new_pos = abs(best_pos - pop[i]) * np.exp(l) * np.cos(2*np.pi*l) + best_pos

                sig     = 1 / (1 + np.exp(-np.clip(new_pos, -10, 10)))
                new_pos = (sig > np.random.rand(dim)).astype(float)
                mut = np.random.rand(dim) < 0.05
                new_pos[mut] = 1 - new_pos[mut]
                new_pos = self._enforce_limit(new_pos)

                nf = self._fitness(new_pos, X_tr, X_te, y_tr, y_te)
                if nf < fits[i]:  pop[i], fits[i] = new_pos, nf
                if nf < best_fit: best_fit, best_pos = nf, new_pos.copy()

            history.append(best_fit)
            print(f"  Iter {t+1:02d}/{self.max_iter} | Fitness: {best_fit:.4f} | "
                  f"Components: {int(np.sum(best_pos>0.5))}/{dim}")

        return best_pos, best_fit, history


ewoa                     = EWOA(n_whales=10, max_iter=20)
best_sol, best_fit, conv = ewoa.optimize(X_train_pca, X_test_pca, y_train, y_test)
sel_idx                  = np.where(best_sol > 0.5)[0]

print(f"\n✅ EWOA: selected {len(sel_idx)}/50 components | Fitness: {best_fit:.4f}")

plt.figure(figsize=(8, 4))
plt.plot(conv, color='#1F4E79', linewidth=2, marker='o', markersize=4)
plt.fill_between(range(len(conv)), conv, alpha=0.1, color='#1F4E79')
plt.title('EWOA Convergence'); plt.xlabel('Iteration'); plt.ylabel('Fitness')
plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig('ewoa_convergence.png', dpi=150, bbox_inches='tight')
plt.show(); print("Convergence plot saved.")

np.save('selected_indices.npy', sel_idx)
joblib.dump(pca, 'pca_model.pkl')

# ════════════════════════════════════════════════
# STEP 4 — XGBOOST
# ════════════════════════════════════════════════
print("\n[STEP 4] XGBoost Classification...")

X_tr_sel = X_train_pca[:, sel_idx]
X_te_sel = X_test_pca[:,  sel_idx]

base     = XGBClassifier(n_estimators=100, random_state=42,
                          eval_metric='logloss', verbosity=0, n_jobs=-1)
base.fit(X_train_pca, y_train)
base_acc = accuracy_score(y_test, base.predict(X_test_pca))

clf = XGBClassifier(n_estimators=100, random_state=42,
                    eval_metric='logloss', verbosity=0, n_jobs=-1)
clf.fit(X_tr_sel, y_train)
y_pred = clf.predict(X_te_sel)
acc    = accuracy_score(y_test, y_pred)

print(f"  Baseline (50 components) : {base_acc*100:.2f}%")
print(f"  EWOA ({len(sel_idx)} components)    : {acc*100:.2f}%")
print(f"\n{classification_report(y_test, y_pred, target_names=['Benign','Malware'])}")

cm = confusion_matrix(y_test, y_pred)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].imshow(cm, cmap='Blues')
axes[0].set_xticks([0,1]); axes[0].set_xticklabels(['Benign','Malware'])
axes[0].set_yticks([0,1]); axes[0].set_yticklabels(['Benign','Malware'])
axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('Actual')
axes[0].set_title(f'Confusion Matrix | Acc: {acc*100:.2f}%')
for i in range(2):
    for j in range(2):
        axes[0].text(j, i, cm[i,j], ha='center', va='center',
                     color='white' if cm[i,j]>cm.max()/2 else 'black',
                     fontsize=14, fontweight='bold')
axes[1].bar(['Baseline\n(50 PCA)', f'EWOA\n({len(sel_idx)} selected)'],
            [base_acc*100, acc*100], color=['#90CAF9','#1F4E79'], width=0.4)
axes[1].set_ylim([max(0, min(base_acc,acc)*100-5), 101])
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title('Baseline vs EWOA')
for i, v in enumerate([base_acc*100, acc*100]):
    axes[1].text(i, v+0.2, f'{v:.2f}%', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('results.png', dpi=150, bbox_inches='tight')
plt.show()

joblib.dump(clf, 'xgb_model.pkl')
print("Models saved.")

# ════════════════════════════════════════════════
# STEP 5 — SHAP
# ════════════════════════════════════════════════
print("\n[STEP 5] SHAP — Opening the Black Box...")

pca_names = [f"PCA_C{i}" for i in sel_idx]

# ensure numeric for SHAP
X_te_sel = np.array(X_te_sel, dtype=float)

X_shap = X_te_sel[:120]
background = shap.sample(X_tr_sel, 100, random_state=42)
masker = shap.maskers.Independent(background)
explainer = shap.Explainer(lambda data: clf.predict_proba(data)[:, 1], masker, algorithm='permutation')
shap_exp = explainer(X_shap, max_evals=2 * X_shap.shape[1] + 1)
shap_array = shap_exp.values

plt.figure()
shap.summary_plot(shap_array, X_shap, feature_names=pca_names, show=False)
plt.tight_layout(); plt.savefig('shap_summary.png', dpi=150, bbox_inches='tight')
plt.show(); print("SHAP summary saved.")

plt.figure()
shap.summary_plot(shap_array, X_shap, feature_names=pca_names, plot_type='bar', show=False)
plt.tight_layout(); plt.savefig('shap_bar.png', dpi=150, bbox_inches='tight')
plt.show(); print("SHAP bar saved.")

expected_value = shap_exp.base_values[0]

shap.force_plot(expected_value, shap_array[0], X_shap[0],
                feature_names=pca_names, matplotlib=True, show=False)
plt.savefig('shap_force_plot.png', dpi=150, bbox_inches='tight')
plt.show(); print("SHAP force plot saved.")

# ════════════════════════════════════════════════a
# SUMMARY
# ════════════════════════════════════════════════
print("______________")
print("\n" + "="*60)
print("\n" + "="*60)
print("  FINAL RESULTS")
print("="*60)
print(f"  Dataset           : CIC-MalDroid-2020")
print(f"  Original Features : {X.shape[1]}")
print(f"  After PCA         : 50 components ({explained:.1f}% variance)")
print(f"  EWOA Selected     : {len(sel_idx)} / 50 components")
print(f"  Baseline Accuracy : {base_acc*100:.2f}%")
print(f"  Final Accuracy    : {acc*100:.2f}%")
print(f"  EWOA Fitness      : {best_fit:.4f}")
print("="*60)
print("\n✅ Plots: pca_variance, ewoa_convergence, results, shap_summary, shap_bar, shap_force_plot")
print("Run: streamlit run step3_dashboard.py")
#some random change to test git pull request

# End of cnn.py
#made some changess