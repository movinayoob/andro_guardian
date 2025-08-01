import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from pyswarm import pso
import numpy as np
import random
import os
import tensorflow as tf
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))
from train_dnn import train_and_evaluate_final_model

def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seed(42)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# -------------------------------
# Step 1: Load cleaned features
# -------------------------------
df = pd.read_csv("data/processed/features_cleaned.csv")
X = df.drop("label", axis=1)
y = df["label"]

# Binary classification encoding
num_classes = len(np.unique(y))
y_encoded = to_categorical(y, num_classes=num_classes)

# -------------------------------
# Step 2: Scale features
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Train/Val/Test Split
X_temp, X_test, y_temp, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, stratify=np.argmax(y_temp, axis=1), random_state=42
)
# Flatten output for binary classification
y_train = y_train[:, 1] if num_classes == 2 else y_train
y_val = y_val[:, 1] if num_classes == 2 else y_val
y_test = y_test[:, 1] if num_classes == 2 else y_test

# -------------------------------
# 🧠 Fitness function for PSO
# -------------------------------
def fitness(params):
    neurons1, neurons2, dropout, lr, batch_size = params
    neurons1 = int(neurons1)
    neurons2 = int(neurons2)
    batch_size = int(batch_size)

    # 🛡️ Clamp dropout rate to (0.001, 0.99) for safety
    dropout = max(0.001, min(dropout, 0.99))

    # ⚠️ Avoid batch_size > training samples
    if batch_size >= len(X_train):
        batch_size = len(X_train) // 2  # Safe fallback

    # Debug safety check
    if len(X_train) == 0 or len(y_train) == 0:
        print("Empty training data! Skipping this evaluation.")
        return 1.0  # High cost to penalize this particle

    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))
    model.add(Dense(neurons1, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(neurons2, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=batch_size,
        verbose=0,
        validation_data=(X_val, y_val)
    )

    val_acc_history = history.history['val_accuracy']
    best_val_acc = max(val_acc_history)
    best_epoch = np.argmax(val_acc_history) + 1

    print(f"Params: {params} → Best Val Acc: {best_val_acc:.4f} at Epoch {best_epoch}")
    return -best_val_acc  # PSO minimizes

# -------------------------------
# Custom PSO with Global, Personal, and Informant Bests
# -------------------------------
def custom_pso(fitness_fn, lb, ub, dim, num_particles=10, max_iter=5, num_informants=3):
    w = 0.5       # inertia
    c1 = 1.5      # cognitive (self)
    c2 = 1.5      # social (global)
    c3 = 1.0      # informant

    particles = np.random.uniform(low=lb, high=ub, size=(num_particles, dim))
    velocities = np.zeros((num_particles, dim))
    personal_best_positions = particles.copy()
    personal_best_scores = np.array([fitness_fn(p) for p in particles])

    global_best_index = np.argmin(personal_best_scores)
    global_best_position = personal_best_positions[global_best_index]

    # Randomly assign informants
    informants = [np.random.choice(num_particles, num_informants, replace=False) for _ in range(num_particles)]

    for iter in range(max_iter):
        for i in range(num_particles):
            # Best among informants
            informant_indices = informants[i]
            informant_best_idx = informant_indices[np.argmin(personal_best_scores[informant_indices])]
            informant_best_position = personal_best_positions[informant_best_idx]

            r1, r2, r3 = np.random.rand(3)
            velocities[i] = (
                w * velocities[i] +
                c1 * r1 * (personal_best_positions[i] - particles[i]) +
                c2 * r2 * (global_best_position - particles[i]) +
                c3 * r3 * (informant_best_position - particles[i])
            )
            particles[i] = np.clip(particles[i] + velocities[i], lb, ub)

            # Evaluate new position
            score = fitness_fn(particles[i])
            if score < personal_best_scores[i]:
                personal_best_scores[i] = score
                personal_best_positions[i] = particles[i]

        # Update global best
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_index]
        print(f"Iteration {iter+1}/{max_iter} - Best Val Acc: {-personal_best_scores[global_best_index]:.4f}")

    return global_best_position, personal_best_scores[global_best_index]

# -------------------------------
# PSO bounds
# -------------------------------
lb = [96, 45, 0.10, 0.001, 64]
ub = [160, 70, 0.22, 0.0035, 128]

# -------------------------------
# Run PSO
# -------------------------------
best_params, best_score = custom_pso(fitness, lb, ub, dim=5, num_particles=10, max_iter=5, num_informants=3)

# -------------------------------
# Final Results
# -------------------------------
print("\nBest Hyperparameters:")
print(f"Neurons in Layer 1: {int(best_params[0])}")
print(f"Neurons in Layer 2: {int(best_params[1])}")
print(f"Dropout Rate: {best_params[2]:.2f}")
print(f"Learning Rate: {best_params[3]:.5f}")
print(f"Batch Size: {int(best_params[4])}")
print(f"Best Validation Accuracy: {-best_score:.4f}")

train_and_evaluate_final_model(
    best_params
)
