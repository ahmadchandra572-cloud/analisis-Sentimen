import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.feature_extraction.text import TfidfVectorizer

# =========================================================================
# ⚠️ LANGKAH 0: INISIASI DATA INPUT (HARUS DIUBAH DENGAN DATA ASLI ANDA)
# Fixes: Mengatasi ValueError: np.nan is an invalid document
# =========================================================================
file_path = '/content/data_labeling.csv' 
try:
    # Coba muat data Anda yang sudah dilabeli
    data = pd.read_csv(file_path) 
    print(f"✅ Berhasil memuat data dari: {file_path}")
except FileNotFoundError:
    print("❌ File data_labeling.csv tidak ditemukan. Menggunakan data dummy.")
    data = pd.DataFrame({
        'steming_data': ['dpr korup gaji tinggi', 'kinerja baik dan bagus', np.nan, 'dpr brengsek', 'pemerintah sudah benar'],
        'label': ['Negatif', 'Positif', 'Netral', 'Negatif', 'Positif']
    })

# --- LANGKAH PERBAIKAN: Menangani NaN dan Vektorisasi ---
# 1. Mengisi nilai NaN pada kolom teks dengan string kosong dan konversi ke string
data['steming_data'] = data['steming_data'].fillna('').astype(str)
# 2. Hapus baris yang memiliki label NaN (jika ada)
data.dropna(subset=['label'], inplace=True)
# 3. Hapus baris yang teksnya kosong setelah diisi NaN
data = data[data['steming_data'].str.strip() != '']

# Vektorisasi (Mengubah teks menjadi matriks numerik menggunakan TF-IDF)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['steming_data'])
y = data['label']

print(f"✅ Data X (fitur) dan y (label) berhasil diinisiasi. Total sampel: {len(y)}")
# =========================================================================
# ⚠️ AKHIR INISIASI DATA
# =========================================================================


# --- 1. Pemisahan Data ---
# Split data into training and testing sets (80:20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("\n--- 1. Data Split ---")
print("Jumlah data training (X_train):", X_train.shape[0])
print("Jumlah data testing (X_test):", X_test.shape[0])


# =========================================================================
# --- 2. FUNGSI OPTIMASI GAM-GWO (GA + GWO) ---
# =========================================================================

# -----------------------------
# Fitness Function (Random Forest, Logistic Regression, SVM)
# -----------------------------
def fitness_rf(params):
    n_estimators, max_depth, min_samples_split, min_samples_leaf = params
    n_estimators = max(int(n_estimators), 1)
    max_depth = None if int(max_depth) == 0 else int(max_depth)
    min_samples_split = max(int(min_samples_split), 2)
    min_samples_leaf = max(int(min_samples_leaf), 1)
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, 
        min_samples_leaf=min_samples_leaf, random_state=42
    )
    score = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy', error_score='raise')
    return np.mean(score)

def fitness_lr(params):
    C = max(params[0], 0.01)
    model = LogisticRegression(C=C, max_iter=1000, solver='lbfgs', multi_class='auto')
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        score = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
    return np.mean(score)

def fitness_svm(params):
    C, gamma = params
    C = max(C, 0.01)
    gamma = max(gamma, 0.0001)
    model = SVC(C=C, gamma=gamma, kernel='rbf', probability=True) 
    score = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy', error_score='raise')
    return np.mean(score)

# -----------------------------
# GA Core Functions
# -----------------------------
def ga_initial_population(pop_size, search_space):
    return np.array([[random.uniform(search_space[j][0], search_space[j][1]) for j in range(len(search_space))]
                     for _ in range(pop_size)])
def ga_crossover(p1, p2):
    return (p1 + p2) / 2
def ga_mutate(ind):
    return ind + np.random.normal(0, 0.1, size=len(ind))
def run_ga(fitness_func, pop_size, generations, search_space):
    population = ga_initial_population(pop_size, search_space)
    for g in range(generations):
        fitness = np.array([fitness_func(ind) for ind in population])
        sorted_idx = np.argsort(fitness)[::-1]
        population = population[sorted_idx]
        best = population[0]
        new_pop = [best]
        while len(new_pop) < pop_size:
            parents = population[np.random.choice(range(min(5, pop_size)), 2, replace=False)]
            child = ga_crossover(parents[0], parents[1])
            child = ga_mutate(child)
            child = np.clip(child, [s[0] for s in search_space], [s[1] for s in search_space])
            new_pop.append(child)
        population = np.array(new_pop)
    return population[0], fitness_func(population[0])

# -----------------------------
# GWO Core Function
# -----------------------------
def gwo_optimize(fitness_func, dim, search_space, init_pos, wolves=10, iterations=10):
    alpha, beta, delta = np.zeros(dim), np.zeros(dim), np.zeros(dim)
    alpha_score, beta_score, delta_score = -1, -1, -1
    positions = np.array([init_pos + np.random.uniform(-0.1,0.1,dim) for _ in range(wolves)])
    
    for t in range(iterations):
        a = 2 - t*(2/iterations)
        for i in range(wolves):
            score = fitness_func(positions[i])
            if score > alpha_score:
                delta_score, beta_score, alpha_score = beta_score, alpha_score, score
                delta, beta, alpha = np.copy(beta), np.copy(alpha), np.copy(positions[i])
            elif score > beta_score:
                delta_score, beta_score = beta_score, score
                delta, beta = np.copy(beta), np.copy(positions[i])
            elif score > delta_score:
                delta_score = score
                delta = np.copy(positions[i])
        for i in range(wolves):
            for j in range(dim):
                r1,r2 = random.random(), random.random()
                A1, C1 = 2*a*r1 - a, 2*r2
                D_alpha = abs(C1*alpha[j] - positions[i][j])
                X1 = alpha[j] - A1*D_alpha

                r1,r2 = random.random(), random.random()
                A2, C2 = 2*a*r1 - a, 2*r2
                D_beta = abs(C2*beta[j] - positions[i][j])
                X2 = beta[j] - A2*D_beta

                r1,r2 = random.random(), random.random()
                A3, C3 = 2*a*r1 - a, 2*r2
                D_delta = abs(C3*delta[j] - positions[i][j])
                X3 = delta[j] - A3*D_delta

                positions[i][j] = (X1+X2+X3)/3
            positions[i] = np.clip(positions[i],[s[0] for s in search_space],[s[1] for s in search_space])
    return alpha, alpha_score

# =========================================================================
# --- 3. PELATIHAN & OPTIMASI MODEL ---
# =========================================================================

print("\n--- 3. Pelatihan Model Dasar dan Optimasi GAM-GWO ---")

# A. Random Forest (RF)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_search_space = [[50,500],[0,50],[2,10],[1,5]] 
best_rf_ga, score_rf_ga = run_ga(fitness_rf, pop_size=10, generations=5, search_space=rf_search_space)
best_rf_gwo_params, score_rf_gwo = gwo_optimize(fitness_rf, dim=4, search_space=rf_search_space, init_pos=best_rf_ga, wolves=5, iterations=5)
best_rf_model = RandomForestClassifier(
    n_estimators=max(int(best_rf_gwo_params[0]),1),
    max_depth=None if int(best_rf_gwo_params[1])==0 else int(best_rf_gwo_params[1]),
    min_samples_split=max(int(best_rf_gwo_params[2]),2),
    min_samples_leaf=max(int(best_rf_gwo_params[3]),1),
    random_state=42
)
best_rf_model.fit(X_train, y_train)
y_pred_rf_opt = best_rf_model.predict(X_test)


# B. Logistic Regression (LR)
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
lr_search_space = [[0.01, 100]] 
best_lr_ga, score_lr_ga = run_ga(fitness_lr, pop_size=10, generations=5, search_space=lr_search_space)
best_lr_gwo_params, score_lr_gwo = gwo_optimize(fitness_lr, dim=1, search_space=lr_search_space, init_pos=best_lr_ga, wolves=5, iterations=5)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    best_lr_model = LogisticRegression(C=max(best_lr_gwo_params[0], 0.01), max_iter=1000, solver='lbfgs', multi_class='auto')
    best_lr_model.fit(X_train, y_train)
y_pred_lr_opt = best_lr_model.predict(X_test)


# C. Support Vector Machine (SVM)
svm = SVC(probability=True) 
svm.fit(X_train, y_train)
svm_search_space = [[0.01,100],[0.0001,1]] 
best_svm_ga, score_svm_ga = run_ga(fitness_svm, pop_size=10, generations=5, search_space=svm_search_space)
best_svm_gwo_params, score_svm_gwo = gwo_optimize(fitness_svm, dim=2, search_space=svm_search_space, init_pos=best_svm_ga, wolves=5, iterations=5)
best_svm_model = SVC(C=best_svm_gwo_params[0], gamma=best_svm_gwo_params[1], kernel='rbf', probability=True)
best_svm_model.fit(X_train, y_train)
y_pred_svm_opt = best_svm_model.predict(X_test)


# =========================================================================
# --- 4. EVALUASI AKHIR (OPTIMIZED MODELS) ---
# =========================================================================

print("\n\n--- 4. Evaluasi Kinerja Model Setelah Optimasi GAM-GWO ---")

# --- 4.1 Visualisasi Confusion Matrix (Optimized Models) ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
models_opt = [best_rf_model, best_lr_model, best_svm_model]
preds_opt = [y_pred_rf_opt, y_pred_lr_opt, y_pred_svm_opt]
titles_opt = ["RF (GAM-GWO)", "LR (GAM-GWO)", "SVM (GAM-GWO)"]
cmaps = ["Greens", "Blues", "Oranges"]

for i, model in enumerate(models_opt):
    cm = confusion_matrix(y_test, preds_opt[i])
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmaps[i], ax=axes[i],
                xticklabels=model.classes_, yticklabels=model.classes_)
    axes[i].set_title(f"Confusion Matrix - {titles_opt[i]}")
    axes[i].set_xlabel("Predicted Labels")
    axes[i].set_ylabel("True Labels")
plt.tight_layout()
plt.show()

# --- 4.2 Laporan Klasifikasi dan Akurasi ---
final_accuracies = {
    "Random Forest (Optimized)": accuracy_score(y_test, y_pred_rf_opt) * 100,
    "Logistic Regression (Optimized)": accuracy_score(y_test, y_pred_lr_opt) * 100,
    "SVM (Optimized)": accuracy_score(y_test, y_pred_svm_opt) * 100
}

print("\n--- Ringkasan Akurasi Model Teroptimasi ---")
for model_name, acc in final_accuracies.items():
    print(f"Akurasi {model_name}: {acc:.2f}%")
    print(classification_report(y_test, preds_opt[list(final_accuracies.keys()).index(model_name)], zero_division=0))
    
# --- 5. Perbandingan AUC/ROC (Menggunakan Model Terbaik) ---

# =============================
# Binarisasi label (Wajib untuk multi-kelas)
# =============================
classes = np.unique(y_test)
y_test_bin = label_binarize(y_test, classes=classes)

# =============================
# Fungsi hitung dan plot AUC/ROC
# =============================
def calculate_and_plot_roc(model, X_test, y_test_bin, model_name):
    """Menghitung Micro-Average AUC dan memplot ROC Curve."""
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        return 0, None

    fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.4f})')
    return roc_auc

# =============================
# Plot semua model teroptimasi dalam 1 grafik
# =============================
plt.figure(figsize=(10, 7))

auc_results = {}
auc_results["Random Forest (Optimized)"] = calculate_and_plot_roc(best_rf_model, X_test, y_test_bin, "Random Forest (Optimized)")
auc_results["Logistic Regression (Optimized)"] = calculate_and_plot_roc(best_lr_model, X_test, y_test_bin, "Logistic Regression (Optimized)")
auc_results["SVM (Optimized)"] = calculate_and_plot_roc(best_svm_model, X_test, y_test_bin, "SVM (Optimized)")

# Garis baseline
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Baseline (AUC = 0.50)')

# Label grafik
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve Comparison (Optimized Models)', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()

# Tampilkan Hasil AUC dalam Tabel
df_auc = pd.DataFrame(auc_results.items(), columns=['Model', 'AUC (Micro-Average Score)'])
df_auc = df_auc.sort_values(by='AUC (Micro-Average Score)', ascending=False).reset_index(drop=True)

print("\n--- 📈 Hasil Perbandingan AUC (Micro-Average) Model Teroptimasi ---")
print(df_auc.to_string(index=False, float_format="%.4f"))
print("-" * 60)
