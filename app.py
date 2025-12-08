import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ====================================================================
# --- ASUMSI DATA ---
# Script ini mengasumsikan variabel X_train, y_train, X_test, dan y_test 
# sudah didefinisikan dari tahap TF-IDF dan data splitting sebelumnya.
# ====================================================================


# -----------------------------
# 1️⃣ FITNESS FUNCTION LOGISTIC REGRESSION (LR)
# -----------------------------
def fitness_lr(params):
    C = params[0]
    C = max(C, 0.01)
    
    # Gunakan solver lbfgs yang stabil untuk multi-class
    model = LogisticRegression(C=C, max_iter=1000, solver='lbfgs', multi_class='auto')
    
    # Menekan FutureWarning saat cross_val_score berjalan
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        score = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy', error_score='raise')
    
    return np.mean(score)


# -----------------------------
# 2️⃣ GA FUNCTIONS (Genetic Algorithm)
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
            parents = population[np.random.choice(range(5), 2, replace=False)]
            child = ga_crossover(parents[0], parents[1])
            child = ga_mutate(child)
            child = np.clip(child, [s[0] for s in search_space], [s[1] for s in search_space])
            new_pop.append(child)
        population = np.array(new_pop)
    return population[0], fitness_func(population[0])


# -----------------------------
# 3️⃣ GWO FUNCTION (Grey Wolf Optimization)
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


# ====================================================================
# 4️⃣ EXECUTION BLOCK: OPTIMASI DAN EVALUASI
# ====================================================================

# Konfigurasi Search Space: Hanya C (regularisasi)
lr_search_space = [[0.01, 100]]

# Run GA (menghasilkan posisi awal yang baik untuk GWO)
best_lr_ga, score_lr_ga = run_ga(fitness_lr, pop_size=10, generations=5, search_space=lr_search_space)

# Run GWO (melakukan optimasi halus)
best_lr_gwo, score_lr_gwo = gwo_optimize(fitness_lr, dim=1, search_space=lr_search_space, init_pos=best_lr_ga, wolves=5, iterations=5)

print("=== Hasil GAM-GWO Logistic Regression ===")
print(f"Best Params C: {best_lr_gwo[0]:.4f}")
print(f"Best Cross-Validation Score: {score_lr_gwo:.4f}")

# -----------------------------
# 5️⃣ Training Final Model
# -----------------------------
# Menekan FutureWarning saat training model final
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    best_lr_model = LogisticRegression(C=best_lr_gwo[0], max_iter=1000, solver='lbfgs', multi_class='auto')
    best_lr_model.fit(X_train, y_train)

y_pred_lr = best_lr_model.predict(X_test)

# -----------------------------
# 6️⃣ Evaluasi dan Visualisasi
# -----------------------------
acc_lr = accuracy_score(y_test, y_pred_lr) * 100
print("\n--- Evaluasi Final ---")
print(f"Akurasi LR (GAM-GWO): {acc_lr:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred_lr, zero_division=0))


# Visualisasi Confusion Matrix
plt.figure(figsize=(6,5))
sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True, fmt="d", cmap="Blues",
            xticklabels=best_lr_model.classes_, yticklabels=best_lr_model.classes_)
plt.title("Confusion Matrix - LR (GAM-GWO)")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()
