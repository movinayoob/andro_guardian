import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from pyswarm import pso
from tensorflow.keras.regularizers import l2
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

# Load Data
df = pd.read_csv("data/processed/feature_mi_selected_120.csv")
X = df.drop("label", axis=1)
y = df["label"]

num_classes = len(np.unique(y))
y_encoded = to_categorical(y, num_classes=num_classes)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_temp, X_test, y_temp, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, stratify=np.argmax(y_temp, axis=1), random_state=42
)

if num_classes == 2:
    y_train = y_train[:, 1]
    y_val = y_val[:, 1]
    y_test = y_test[:, 1]

def build_optimized_dnn(input_dim, neurons1, neurons2, neurons3, dropout_rate, learning_rate):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(neurons1, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(neurons2, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(neurons3, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = AdamW(learning_rate=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def fitness(params):
    try:
        neurons1, neurons2, neurons3, dropout, lr, batch_size = params
        neurons1 = int(neurons1)
        neurons2 = int(neurons2)
        neurons3 = int(neurons3)
        lr = 10 ** lr
        batch_size = int(batch_size)
        dropout = float(dropout)

        print(f"Neurons1: {neurons1}, Neurons2: {neurons2}, Neurons3: {neurons3}, Dropout: {dropout:.4f}, Learning Rate: {lr:.5f}, Batch Size: {batch_size}")

        if batch_size >= len(X_train):
            batch_size = max(16, len(X_train) // 2)  # fallback

        if len(X_train) == 0 or len(y_train) == 0:
            print("Empty training data. Returning high loss.")
            return 1.0  # Penalize

        model = build_optimized_dnn(X_train.shape[1], neurons1, neurons2, neurons3, dropout, lr)
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=30,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=0
        )

        val_acc_list = history.history.get('val_accuracy', [])
        if not val_acc_list:
            print("Validation accuracy list is empty. Returning high loss.")
            return 1.0

        best_val_acc = max(val_acc_list)
        best_epoch = np.argmax(val_acc_list) + 1
        print(f"Params: {params} → Best Val Acc: {best_val_acc:.4f} at Epoch {best_epoch}")
        return -best_val_acc  # Negative for minimization

    except Exception as e:
        print(f"Exception occurred for params {params}: {e}")
        return 1.0  # Return poor fitness score if any error occurs

def custom_pso(fitness_fn, lb, ub, dim, num_particles, max_iter, num_informants, top_k):
    lb = np.array(lb)
    ub = np.array(ub)
    # PSO hyperparameters (initial)
    w_max, w_min = 0.9, 0.4
    c1_max, c1_min = 2.0, 1.3
    c2_max, c2_min = 2.0, 1.3
    c3 = 1.0  # informant coefficient stays constant

    # Initialize particles and velocities
    particles = np.random.uniform(low=lb, high=ub, size=(num_particles, dim))
    velocities = np.zeros((num_particles, dim))
    personal_best_positions = particles.copy()
    personal_best_scores = np.array([fitness_fn(p) for p in particles])
    global_best_index = np.argmin(personal_best_scores)
    global_best_position = personal_best_positions[global_best_index]
    informants = [np.random.choice(num_particles, num_informants, replace=False) for _ in range(num_particles)]

    # Velocity clamping max (20% of range per dimension)
    v_max = 0.2 * (ub - lb)

    # Optional: constriction factor (uncomment to use)
    # phi = c1_max + c2_max + c3
    # chi = 2 / abs(2 - phi - np.sqrt(phi**2 - 4 * phi))

    all_particles_and_scores = [(particles[i].copy(), personal_best_scores[i]) for i in range(num_particles)]

    for iter in range(max_iter):
        # Linearly decaying coefficients
        w = w_max - (w_max - w_min) * (iter / max_iter)
        c1 = c1_max - (c1_max - c1_min) * (iter / max_iter)
        c2 = c2_min + (c2_max - c2_min) * (iter / max_iter)

        for i in range(num_particles):
            informant_best_idx = informants[i][np.argmin(personal_best_scores[informants[i]])]
            r1, r2, r3 = np.random.rand(3)

            # Velocity update (with decay terms)
            v_new = (
                w * velocities[i] +
                c1 * r1 * (personal_best_positions[i] - particles[i]) +
                c2 * r2 * (global_best_position - particles[i]) +
                c3 * r3 * (personal_best_positions[informant_best_idx] - particles[i])
            )

            # Optional: apply constriction
            # v_new = chi * v_new

            # Velocity clamping
            v_new = np.clip(v_new, -v_max, v_max)
            velocities[i] = v_new

            # Position update
            particles[i] = np.clip(particles[i] + velocities[i], lb, ub)
            score = fitness_fn(particles[i])
            all_particles_and_scores.append((particles[i].copy(), score))

            # Personal best update
            if score < personal_best_scores[i]:
                personal_best_scores[i] = score
                personal_best_positions[i] = particles[i]

        # Global best update
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_index]
        print(f"Iteration {iter+1}/{max_iter} - Best Val Acc: {-personal_best_scores[global_best_index]:.4f}")

    # Tighten bounds using top-k particles
    sorted_results = sorted(all_particles_and_scores, key=lambda x: x[1])[:top_k]
    top_params = np.array([x[0] for x in sorted_results])
    tight_factor = 0.9
    center = np.mean(top_params, axis=0)
    range_half = (np.max(top_params, axis=0) - np.min(top_params, axis=0)) / 2 * tight_factor
    new_lb = np.maximum(lb, center - range_half)
    new_ub = np.minimum(ub, center + range_half)

    return global_best_position, personal_best_scores[global_best_index], new_lb, new_ub

lb = [282.18, 48.43, 24.41, 0.2043, -3.4674, 133.02]
ub = [417.45, 64.35, 32.83, 0.2665, -3.2187, 164.37]

best_params, best_score, new_lb, new_ub  = custom_pso(fitness, lb, ub, dim=6, num_particles=10, max_iter=5, num_informants=2,top_k=10)
#best_params = [339, 56, 28, 0.23, -3.33903, 145]
print("\nBest Hyperparameters:")
print(f"Neurons in Layer 1: {int(best_params[0])}")
print(f"Neurons in Layer 2: {int(best_params[1])}")
print(f"Neurons in Layer 3: {int(best_params[2])}")
print(f"Dropout Rate: {best_params[3]:.2f}")
print(f"Learning Rate: {best_params[4]:.5f}")
print(f"Batch Size: {int(best_params[5])}")
print(f"Best Validation Accuracy: {-best_score:.4f}")

print("\nRefined bounds for next PSO run:")
print("New Lower Bounds:", new_lb)
print("New Upper Bounds:", new_ub)
#best_params = [339, 56, 28, 0.23, -3.33903, 145]
train_and_evaluate_final_model(best_params)