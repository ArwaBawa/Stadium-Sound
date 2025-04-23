import os
import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub

# Load pretrained YAMNet model
yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
yamnet_model = hub.load(yamnet_model_handle)

# Load class map (YAMNet labels)
class_map_path = tf.keras.utils.get_file(
    'yamnet_class_map.csv',
    'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv'
)
class_names = pd.read_csv(class_map_path)['display_name'].tolist()

# Folder containing ESC-50 audio
audio_folder = 'audio'  # Change this if needed

# Loudness label function
def loudness_to_label(db):
    if db <= -40:
        return "Low"
    elif db <= -20:
        return "Medium"
    else:
        return "High"

# Analyze one file
def analyze_audio(file_path):
    waveform, sr = librosa.load(file_path, sr=16000)  # YAMNet requires 16kHz
    # Predict with YAMNet
    scores, embeddings, spectrogram = yamnet_model(waveform)
    mean_scores = tf.reduce_mean(scores, axis=0)
    top_class = tf.argmax(mean_scores)
    label = class_names[top_class.numpy()]
    confidence = mean_scores[top_class].numpy()

    # Loudness
    rms = librosa.feature.rms(y=waveform)
    db = librosa.amplitude_to_db(rms, ref=np.max)
    loudness_db = np.mean(db)
    loudness_label = loudness_to_label(loudness_db)

    return label, confidence, loudness_db, loudness_label

# Scan all files and collect results
data = []

for filename in os.listdir(audio_folder):
    if filename.endswith(".wav"):
        file_path = os.path.join(audio_folder, filename)
        try:
            label, conf, loudness_db, loud_label = analyze_audio(file_path)
            data.append({
                'file_path': file_path,  # Include exact file path here
                'label': loud_label
            })
            print(f"Processed: {filename} → {label} ({round(conf,2)}), Loudness: {loud_label}")
        except Exception as e:
            print(f"Failed to process {filename}: {e}")

# Save to CSV
df = pd.DataFrame(data)
df.to_csv('esc50_labeled_with_loudness.csv', index=False)
print("✅ CSV saved as 'esc50_labeled_with_loudness.csv'")
