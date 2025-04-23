import os
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('esc50_labeled_with_loudness.csv')

# Enhanced feature extraction
def extract_features(file_path, segment_length=3, hop_length=1, n_mels=128):
    y, sr = librosa.load(file_path, sr=None)
    segments = []
    segment_samples = segment_length * sr
    
    for start in range(0, len(y) - segment_samples, hop_length * sr):
        segment = y[start:start + segment_samples]
        
        # Extract multiple features
        mel_spec = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=n_mels, fmax=8000)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Add delta features
        delta = librosa.feature.delta(mel_spec_db)
        delta2 = librosa.feature.delta(mel_spec_db, order=2)
        
        # Stack features
        features = np.stack([mel_spec_db, delta, delta2], axis=-1)
        segments.append(features)
    
    return segments

# Process data
df['features'] = df['file_path'].apply(extract_features)
df = df[df['features'].notnull()]

X = np.concatenate(df['features'].values)
y = np.repeat(df['label'].values, [len(x) for x in df['features']])

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Optimized CNN Model
def build_high_accuracy_cnn(input_shape, num_classes):
    model = models.Sequential([
        # Input block
        layers.Conv2D(64, (3,3), activation='relu', 
                      kernel_regularizer=regularizers.l2(0.001),
                      input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.3),
        
        # Middle blocks
        layers.Conv2D(128, (3,3), activation='relu',
                     kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.4),
        
        layers.Conv2D(256, (3,3), activation='relu',
                     kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.5),
        
        # Classifier
        layers.Flatten(),
        layers.Dense(512, activation='relu',
                    kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    
    return model

# Build and train
input_shape = X_train.shape[1:]
num_classes = len(np.unique(y))

model = build_high_accuracy_cnn(input_shape, num_classes)

# Add callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
]

history = model.fit(X_train, y_train,
                   epochs=50,
                   batch_size=32,
                   validation_data=(X_test, y_test),
                   callbacks=callbacks,
                   verbose=1)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc*100:.2f}%")

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy Over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Over Epochs')
plt.legend()

plt.tight_layout()
plt.show()
# After model training, evaluate and print accuracy
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n=== Final Model Accuracy ===")
print(f"Test Accuracy: {test_acc*100:.2f}%")

# Get predictions for more detailed analysis
y_pred = np.argmax(model.predict(X_test), axis=1)

# Print classification metrics
from sklearn.metrics import classification_report
print("\n=== Detailed Classification Report ===")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Print confusion matrix (visual)
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', 
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_,
            cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
# Save model
model.save("best_cnn_loudness_classifier.h5")