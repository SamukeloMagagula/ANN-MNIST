# =============================================================================
# ANN for Handwritten Digit Classification — MNIST Dataset
# =============================================================================

# ── IMPORTS ───────────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from PIL import Image
import os


# =============================================================================
# SECTION 2: DATASET LOADING AND EXPLORATION
# =============================================================================

print("=" * 60)
print("SECTION 2: DATASET LOADING AND EXPLORATION")
print("=" * 60)

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Print dataset shapes
print("Training data shape:", x_train.shape)   # (60000, 28, 28)
print("Training labels shape:", y_train.shape) # (60000,)
print("Test data shape:", x_test.shape)         # (10000, 28, 28)
print("Test labels shape:", y_test.shape)       # (10000,)

# Print first 20 pixel values of the first training image
print("\nFirst 20 pixel values of training image [0]:")
print(x_train[0].flatten()[:20])

# Display sample images from the dataset
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle("Sample MNIST Images", fontsize=14, fontweight='bold')
for i, ax in enumerate(axes.flatten()):
    ax.imshow(x_train[i], cmap='gray')
    ax.set_title(f"Label: {y_train[i]}")
    ax.axis('off')
plt.tight_layout()
plt.savefig("sample_images.png", dpi=150)
plt.show()
print("Sample images saved as sample_images.png")


# =============================================================================
# SECTION 3: DATA PREPROCESSING
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 3: DATA PREPROCESSING")
print("=" * 60)

# Step 1: Flatten — reshape 28x28 images to 784-element vectors
x_train = x_train.reshape(60000, 784)
x_test  = x_test.reshape(10000, 784)
print("After flattening:")
print("  x_train shape:", x_train.shape)  # (60000, 784)
print("  x_test shape: ", x_test.shape)   # (10000, 784)

# Step 2: Normalise — scale pixel values from [0, 255] to [0.0, 1.0]
x_train = x_train / 255.0
x_test  = x_test  / 255.0
print("\nAfter normalisation:")
print("  Min value:", x_train.min(), " | Max value:", x_train.max())

# Step 3: One-hot encode labels
y_train_encoded = to_categorical(y_train, 10)
y_test_encoded  = to_categorical(y_test, 10)
print("\nAfter one-hot encoding:")
print("  y_train shape:", y_train_encoded.shape)  # (60000, 10)
print("  y_test shape: ", y_test_encoded.shape)   # (10000, 10)
print("  Example — label 3 becomes:", y_train_encoded[y_train.tolist().index(3)])


# =============================================================================
# SECTION 4 & 5: BUILD AND COMPILE THE ANN MODEL
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 4 & 5: BUILDING AND COMPILING THE ANN")
print("=" * 60)

model = Sequential()

# Input + Hidden Layer 1: 128 neurons, ReLU activation
model.add(Dense(128, activation='relu', input_shape=(784,)))

# Hidden Layer 2: 64 neurons, ReLU activation
model.add(Dense(64, activation='relu'))

# Output Layer: 10 neurons (one per digit), Softmax activation
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Print model summary
model.summary()


# =============================================================================
# SECTION 6: TRAIN THE MODEL
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 6: TRAINING THE MODEL")
print("=" * 60)

history = model.fit(
    x_train, y_train_encoded,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Plot training history
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Training History", fontsize=14, fontweight='bold')

# Accuracy plot
ax1.plot(history.history['accuracy'],     label='Training Accuracy',   color='blue')
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
ax1.set_title('Accuracy over Epochs')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True)

# Loss plot
ax2.plot(history.history['loss'],     label='Training Loss',   color='blue')
ax2.plot(history.history['val_loss'], label='Validation Loss', color='orange')
ax2.set_title('Loss over Epochs')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig("training_history.png", dpi=150)
plt.show()
print("Training history saved as training_history.png")


# =============================================================================
# SECTION 7: EVALUATE THE MODEL
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 7: MODEL EVALUATION")
print("=" * 60)

test_loss, test_acc = model.evaluate(x_test, y_test_encoded, verbose=0)
print(f"Test Loss:     {test_loss:.4f}")
print(f"Test Accuracy: {test_acc * 100:.2f}%")

if test_acc >= 0.90:
    print("✓ Target accuracy of >90% achieved!")
else:
    print("✗ Target accuracy not yet met — try more epochs or adjust architecture.")


# =============================================================================
# SECTION 8: PARAMETER ANALYSIS (Experiments)
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 8: PARAMETER ANALYSIS")
print("=" * 60)

results = []

# ── Experiment 1: Different number of epochs ──────────────────────────────
print("\n[Experiment 1] Varying number of epochs: 5, 10, 20")
for epochs in [5, 10, 20]:
    m = Sequential([
        Dense(128, activation='relu', input_shape=(784,)),
        Dense(64,  activation='relu'),
        Dense(10,  activation='softmax')
    ])
    m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    m.fit(x_train, y_train_encoded, epochs=epochs, batch_size=32,
          validation_split=0.2, verbose=0)
    _, acc = m.evaluate(x_test, y_test_encoded, verbose=0)
    print(f"  Epochs={epochs:2d}  →  Test Accuracy: {acc*100:.2f}%")
    results.append({"Epochs": epochs, "LR": 0.001, "Layers": 2, "Accuracy": acc})

# ── Experiment 2: Different learning rates ────────────────────────────────
print("\n[Experiment 2] Varying learning rate: 0.0001, 0.001, 0.01")
for lr in [0.0001, 0.001, 0.01]:
    m = Sequential([
        Dense(128, activation='relu', input_shape=(784,)),
        Dense(64,  activation='relu'),
        Dense(10,  activation='softmax')
    ])
    m.compile(optimizer=Adam(learning_rate=lr),
              loss='categorical_crossentropy', metrics=['accuracy'])
    m.fit(x_train, y_train_encoded, epochs=10, batch_size=32,
          validation_split=0.2, verbose=0)
    _, acc = m.evaluate(x_test, y_test_encoded, verbose=0)
    print(f"  LR={lr}  →  Test Accuracy: {acc*100:.2f}%")
    results.append({"Epochs": 10, "LR": lr, "Layers": 2, "Accuracy": acc})

# ── Experiment 3: Different number of hidden layers ───────────────────────
print("\n[Experiment 3] Varying number of hidden layers: 1, 2, 3")

# 1 hidden layer
m1 = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dense(10,  activation='softmax')
])
m1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
m1.fit(x_train, y_train_encoded, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
_, acc1 = m1.evaluate(x_test, y_test_encoded, verbose=0)
print(f"  1 hidden layer  →  Test Accuracy: {acc1*100:.2f}%")

# 2 hidden layers (baseline model)
_, acc2 = model.evaluate(x_test, y_test_encoded, verbose=0)
print(f"  2 hidden layers →  Test Accuracy: {acc2*100:.2f}%")

# 3 hidden layers
m3 = Sequential([
    Dense(256, activation='relu', input_shape=(784,)),
    Dense(128, activation='relu'),
    Dense(64,  activation='relu'),
    Dense(10,  activation='softmax')
])
m3.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
m3.fit(x_train, y_train_encoded, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
_, acc3 = m3.evaluate(x_test, y_test_encoded, verbose=0)
print(f"  3 hidden layers →  Test Accuracy: {acc3*100:.2f}%")


# =============================================================================
# SECTION 9: CONFUSION MATRIX
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 9: CONFUSION MATRIX")
print("=" * 60)

# Generate predictions on the test set
predictions     = model.predict(x_test, verbose=0)
predicted_labels = np.argmax(predictions, axis=1)
true_labels      = np.argmax(y_test_encoded, axis=1)

# Compute confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)
print("Confusion Matrix:")
print(cm)

# Print classification report
print("\nClassification Report:")
print(classification_report(true_labels, predicted_labels,
                             target_names=[str(i) for i in range(10)]))

# Plot confusion matrix as heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(10), yticklabels=range(10))
plt.title('Confusion Matrix — ANN on MNIST Test Set', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
print("Confusion matrix saved as confusion_matrix.png")


# =============================================================================
# SECTION 10: TESTING WITH CUSTOM HANDWRITTEN IMAGES
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 10: TESTING WITH CUSTOM IMAGES")
print("=" * 60)

def preprocess_custom_image(filepath):
    """
    Load a custom digit image and preprocess it to match MNIST format.
    - Convert to grayscale
    - Resize to 28x28
    - Normalise to [0, 1]
    - Flatten to 784-element vector
    """
    img = Image.open(filepath).convert('L')   # Grayscale
    img = img.resize((28, 28), Image.LANCZOS) # Resize to 28x28
    img_array = np.array(img, dtype=np.float32)
    img_array = img_array / 255.0             # Normalise
    img_array = img_array.reshape(1, 784)     # Flatten
    return img_array


def predict_custom_digit(filepath):
    """Predict the digit in a custom image and print the result."""
    img_array  = preprocess_custom_image(filepath)
    prediction = model.predict(img_array, verbose=0)
    digit      = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    print(f"  File: {os.path.basename(filepath)}")
    print(f"  Predicted: {digit}  |  Confidence: {confidence:.1f}%")
    print(f"  All probabilities: {np.round(prediction[0], 3)}")
    return digit, confidence


# ── Usage: predict from your own image files ──────────────────────────────
# Place your images (digit_0.jpg, digit_1.jpg, etc.) in the same folder.
# Then run:
#
#   digit, conf = predict_custom_digit("digit_3.jpg")
#
# Or batch-process all your images:

custom_image_files = [
    "digit_0.jpg", "digit_1.jpg", "digit_2.jpg", "digit_3.jpg",
    "digit_4.jpg", "digit_5.jpg", "digit_6.jpg", "digit_7.jpg",
    "digit_8.jpg", "digit_9.jpg",
]

correct = 0
total   = 0

print("Custom Image Predictions:")
for i, fname in enumerate(custom_image_files):
    if os.path.exists(fname):
        predicted, conf = predict_custom_digit(fname)
        true_digit = i  # Assumes filenames match digit order (digit_0 = 0, etc.)
        if predicted == true_digit:
            correct += 1
        total += 1
        print()
    else:
        print(f"  {fname} not found — skipping")

if total > 0:
    print(f"Custom Image Accuracy: {correct}/{total} = {correct/total*100:.1f}%")
else:
    print("No custom images found. Add digit_0.jpg to digit_9.jpg to this folder.")


# =============================================================================
# SECTION 11: RESULTS SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 11: FINAL RESULTS SUMMARY")
print("=" * 60)

print(f"MNIST Test Accuracy:     {test_acc*100:.2f}%")
print(f"MNIST Test Loss:         {test_loss:.4f}")
if total > 0:
    print(f"Custom Image Accuracy:   {correct/total*100:.1f}%  ({correct}/{total})")
print(f"Target (>90%) achieved:  {'YES ✓' if test_acc >= 0.90 else 'NO ✗'}")
print("=" * 60)
