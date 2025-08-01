import numpy as np
import pandas as pd
import os
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization,Embedding,Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

print("🚀 Script started")

def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seed(42)

print("📁 Loading data...")
df = pd.read_csv("data/processed/covearge_sequential_syscall_40k_400_107.csv")
X = df.drop("label", axis=1).values
y = df["label"].values

num_classes = len(np.unique(y))
y_encoded = to_categorical(y, num_classes=num_classes)

X_temp, X_test, y_temp, y_test = train_test_split(X, y_encoded, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, stratify=np.argmax(y_temp, axis=1), random_state=42)

if num_classes == 2:
    y_train = y_train[:, 1]
    y_val = y_val[:, 1]
    y_test = y_test[:, 1]

print("✅ Data loaded and split")

""" def fitness(params):
    try:
        print(f"🔍 Evaluating params: {params}")
        units1, units2, units3, dropout, lr, batch_size = params
        units1, units2, units3 = int(units1), int(units2), int(units3)
        dropout = float(dropout)
        lr = 10 ** lr
        batch_size = int(batch_size)

        model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Embedding(input_dim=21, output_dim=16, mask_zero=True),
        Bidirectional(LSTM(units1, return_sequences=True)),
        BatchNormalization(),
        Dropout(dropout),
        Bidirectional(LSTM(units2, return_sequences=True)),
        BatchNormalization(),
        Dropout(dropout),
        Bidirectional(LSTM(units3)),
        BatchNormalization(),
        Dropout(dropout),
        Dense(1, activation='sigmoid')
])

        model.compile(loss='binary_crossentropy', optimizer=AdamW(learning_rate=lr), metrics=['accuracy'])

        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                            epochs=100, batch_size=batch_size,
                            callbacks=[early_stop], verbose=0)

        val_acc = history.history.get('val_accuracy', [])
        best = -max(val_acc) if val_acc else 1.0
        print(f"✅ Best val_acc: {-best:.4f}")
        return best

    except Exception as e:
        print(f"❌ Error in fitness: {e}")
        return 1.0 """

def fitness(params):
    try:
        print(f"🔍 Evaluating params: {params}")
        units1, units2, units3, dropout, lr, batch_size = params
        units1, units2, units3 = int(units1), int(units2), int(units3)
        dropout = float(dropout)
        lr = 10 ** lr
        batch_size = int(batch_size)
        vocab_size = int(np.max(X_train)) + 1

        inputs = Input(shape=(X_train.shape[1],))
        x = Embedding(input_dim=vocab_size, output_dim=16, mask_zero=True)(inputs)
    
        x = Bidirectional(LSTM(units1, return_sequences=True))(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)

        x = Bidirectional(LSTM(units2, return_sequences=True))(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)

        x = Bidirectional(LSTM(units3, return_sequences=True))(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)

        # Add Attention Layer
        x = AttentionWithMasking()(x)

        outputs = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=inputs, outputs=outputs)

        model.compile(loss='binary_crossentropy', optimizer=AdamW(learning_rate=lr), metrics=['accuracy'])

        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                            epochs=100, batch_size=batch_size,
                            callbacks=[early_stop], verbose=0)

        val_acc = history.history.get('val_accuracy', [])
        best = -max(val_acc) if val_acc else 1.0
        print(f"✅ Best val_acc: {-best:.4f}")
        return best

    except Exception as e:
        print(f"❌ Error in fitness: {e}")
        return 1.0

def custom_pso(fitness_fn, lb, ub, dim, num_particles, max_iter, num_informants, top_k):
    print("⚙️ Starting PSO...")
    lb = np.array(lb)
    ub = np.array(ub)
    w_max, w_min = 0.9, 0.4
    c1_max, c1_min = 2.0, 1.3
    c2_max, c2_min = 2.0, 1.3
    c3 = 1.0

    particles = np.random.uniform(low=lb, high=ub, size=(num_particles, dim))
    velocities = np.zeros((num_particles, dim))
    personal_best_positions = particles.copy()
    personal_best_scores = np.array([fitness_fn(p) for p in particles])
    global_best_index = np.argmin(personal_best_scores)
    global_best_position = personal_best_positions[global_best_index]
    informants = [np.random.choice(num_particles, num_informants, replace=False) for _ in range(num_particles)]
    v_max = 0.2 * (ub - lb)
    all_particles_and_scores = [(particles[i].copy(), personal_best_scores[i]) for i in range(num_particles)]

    for iter in range(max_iter):
        print(f"\n🔁 Iteration {iter+1}/{max_iter}")
        w = w_max - (w_max - w_min) * (iter / max_iter)
        c1 = c1_max - (c1_max - c1_min) * (iter / max_iter)
        c2 = c2_min + (c2_max - c2_min) * (iter / max_iter)

        for i in range(num_particles):
            informant_best_idx = informants[i][np.argmin(personal_best_scores[informants[i]])]
            r1, r2, r3 = np.random.rand(3)
            v_new = (
                w * velocities[i] +
                c1 * r1 * (personal_best_positions[i] - particles[i]) +
                c2 * r2 * (global_best_position - particles[i]) +
                c3 * r3 * (personal_best_positions[informant_best_idx] - particles[i])
            )
            v_new = np.clip(v_new, -v_max, v_max)
            velocities[i] = v_new
            particles[i] = np.clip(particles[i] + velocities[i], lb, ub)
            score = fitness_fn(particles[i])
            all_particles_and_scores.append((particles[i].copy(), score))

            if score < personal_best_scores[i]:
                personal_best_scores[i] = score
                personal_best_positions[i] = particles[i]

        global_best_index = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_index]
        print(f"🔥 Best Val Acc this iter: {-personal_best_scores[global_best_index]:.4f}")

    sorted_results = sorted(all_particles_and_scores, key=lambda x: x[1])[:top_k]
    top_params = np.array([x[0] for x in sorted_results])
    tight_factor = 0.9
    center = np.mean(top_params, axis=0)
    range_half = (np.max(top_params, axis=0) - np.min(top_params, axis=0)) / 2 * tight_factor
    new_lb = np.maximum(lb, center - range_half)
    new_ub = np.minimum(ub, center + range_half)
    return global_best_position, personal_best_scores[global_best_index], new_lb, new_ub

""" def evaluate_final_lstm(best_params):
    print("🎯 Training final LSTM with best params...")
    units1, units2, units3 = int(best_params[0]), int(best_params[1]), int(best_params[2])
    dropout_rate = best_params[3]
    learning_rate = 10 ** best_params[4]
    batch_size = int(best_params[5])
    vocab_size = int(np.max(X_train)) + 1


    model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Embedding(input_dim=vocab_size, output_dim=16, mask_zero=True),
    Bidirectional(LSTM(units1, return_sequences=True)),
    BatchNormalization(),
    Dropout(dropout_rate),
    Bidirectional(LSTM(units2, return_sequences=True)),
    BatchNormalization(),
    Dropout(dropout_rate),
    Bidirectional(LSTM(units3)),
    BatchNormalization(),
    Dropout(dropout_rate),
    Dense(1, activation='sigmoid')
])
    print("🧪 Vocab size (input_dim):", vocab_size)
    print("🧪 Max token ID in train:", np.max(X_train))
    print("🧪 Class distribution in y_train:", np.unique(y_train, return_counts=True))

    model.compile(loss='binary_crossentropy', optimizer=AdamW(learning_rate=learning_rate), metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=100, batch_size=batch_size, callbacks=[early_stop], verbose=1)

    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    y_true = y_test.flatten()

    print("✅ Final Test Accuracy:", accuracy_score(y_true, y_pred))
    print("✅ Classification Report:\n", classification_report(y_true, y_pred))
    print("✅ Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show() """


from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Dropout, BatchNormalization, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# Custom Attention Layer
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def call(self, inputs):
        # inputs: [batch_size, time_steps, features]
        score = K.tanh(inputs)
        attention_weights = K.softmax(score, axis=1)
        context_vector = attention_weights * inputs
        context_vector = K.sum(context_vector, axis=1)
        return context_vector
    
class AttentionWithMasking(Layer):
    def __init__(self, **kwargs):
        super(AttentionWithMasking, self).__init__(**kwargs)

    def call(self, inputs, mask=None):
        # Compute attention scores
        score = K.tanh(inputs)
        attention_weights = K.softmax(score, axis=1)

        if mask is not None:
            attention_weights *= K.cast(mask[:, :, None], K.floatx())

        attention_weights /= K.sum(attention_weights, axis=1, keepdims=True) + K.epsilon()
        weighted_sum = K.sum(inputs * attention_weights, axis=1)
        return weighted_sum

    def compute_mask(self, inputs, mask=None):
        return None


def evaluate_final_lstm(best_params):
    print("🎯 Training final LSTM with best params...")
    units1, units2, units3 = int(best_params[0]), int(best_params[1]), int(best_params[2])
    dropout_rate = best_params[3]
    learning_rate = 10 ** best_params[4]
    batch_size = int(best_params[5])
    vocab_size = int(np.max(X_train)) + 1


    # Functional API
    inputs = Input(shape=(X_train.shape[1],))
    x = Embedding(input_dim=vocab_size, output_dim=16, mask_zero=True)(inputs)
    
    x = Bidirectional(LSTM(units1, return_sequences=True))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    x = Bidirectional(LSTM(units2, return_sequences=True))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    x = Bidirectional(LSTM(units3, return_sequences=True))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    # Add Attention Layer
    x = AttentionWithMasking()(x)

    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)

    print("🧪 Vocab size (input_dim):", vocab_size)
    print("🧪 Max token ID in train:", np.max(X_train))
    print("🧪 Class distribution in y_train:", np.unique(y_train, return_counts=True))

    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(learning_rate=learning_rate),
                  metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=100,
                        batch_size=batch_size,
                        callbacks=[early_stop],
                        verbose=1)

    # Evaluation
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    y_true = y_test.flatten()

    test_acc = accuracy_score(y_true, y_pred)

    model.save("saved_models/my_model.keras")

    print("✅ Final Test Accuracy:", test_acc)
    print("✅ Classification Report:\n", classification_report(y_true, y_pred))
    print("✅ Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

    # Plot accuracy and loss
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.axhline(test_acc, color='purple', linestyle='--', label='Test Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    cm = confusion_matrix(y_true, y_pred)
    plt.imshow(cm, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xticks([0, 1], ['Benign', 'Malware'])
    plt.yticks([0, 1], ['Benign', 'Malware'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    plt.tight_layout()
    plt.show()
# ------------------------------
# Run Main
# ------------------------------
if __name__ == "__main__":
    try:
        print("🚦 Running custom PSO...")
        lb = [32, 30, 16, 0.1, np.log10(0.0001), 64]
        ub = [128, 64, 32, 0.4, np.log10(0.01), 128]

        #lb = [32.25506836, 41.62744294, 20.34665322, 0.18653944, -4.0, 90.73930026]   # Lower bounds
        #ub = [99.15968843, 61.63485621, 26.38442986, 0.34293094, -3.54632059, 126.10082337]  # Upper bounds

        best_params, best_score, new_lb, new_ub = custom_pso(
            fitness_fn=fitness,
            lb=lb,
            ub=ub,
            dim=6,
            num_particles=3,
            max_iter=2,
            num_informants=1,
            top_k=5
        )

        print("\n🏆 Best Hyperparameters:")
        print(f"Units: {int(best_params[0])}, {int(best_params[1])}, {int(best_params[2])}")
        print(f"Dropout: {best_params[3]:.2f}, LR: {10**best_params[4]:.5f}, Batch Size: {int(best_params[5])}")
        print(f"Best Val Acc: {-best_score:.4f}")
        print("\nRefined bounds for next PSO run:")
        print("New Lower Bounds:", new_lb)
        print("New Upper Bounds:", new_ub) 


        #best_params = [61, 53, 23, 0.28, -4.0, 95]
        evaluate_final_lstm(best_params)
    except Exception as e:
        print(f"❌ Exception caught in main: {e}")