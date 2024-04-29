import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch

# class EmotionDataset(Dataset):
#     def __init__(self, audio_paths, text_data, max_length):
#         self.audio_paths = audio_paths
#         self.text_data = text_data
#         self.max_length = max_length
#         self.emotions = np.repeat(np.arange(6), len(audio_paths) // 6)  # Adjust based on your actual distribution

#     def __len__(self):
#         return len(self.audio_paths)
    
#     def __getitem__(self, idx):
#         spectrogram_path = self.audio_paths[idx]
#         spectrogram = np.load(spectrogram_path)
#         spectrogram = torch.tensor(spectrogram, dtype=torch.float32)

#         # Ensure all spectrograms are padded to the global maximum length
#         if spectrogram.shape[-1] < self.max_length:
#             padding_size = self.max_length - spectrogram.shape[-1]
#             spectrogram = torch.nn.functional.pad(spectrogram, (0, padding_size), "constant", 0)  # Pad on the last dimension

#         emotion = torch.tensor(self.emotions[idx], dtype=torch.long)
#         text_input = torch.tensor(self.text_data[idx], dtype=torch.long) if self.text_data else None

#         return spectrogram, emotion, text_input

# def get_train_loader(audio_paths, text_data, max_length, batch_size=8):
#     dataset = EmotionDataset(audio_paths, text_data, max_length)
#     return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# # Example usage
# df = pd.read_excel('spect_output/spectrogram_files.xlsx')
# audio_paths = df['Filename'].tolist()
# # Calculate the maximum length only once across all data
# max_length = max([np.load(path).shape[1] for path in audio_paths])

# train_loader = get_train_loader(audio_paths, np.arange(len(df)), max_length)

# # Iterating through batches
# for spectrograms, emotions, text_indices in train_loader:
#     print(f"Spectrogram batch shape: {spectrograms.shape}")
#     print(f"Emotion labels: {emotions.shape}")
#     if text_indices is not None:
#         print(f"Text indices shape: {text_indices.shape}")
#     break  # Remove break to iterate through all batches
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import whisper
import torch.nn.functional as F
# Assuming whisper model is already loaded
whisper_model = whisper.load_model("base")

class EmotionDataset(Dataset):
    def __init__(self, audio_paths, emotion_labels, max_length, device, whisper_frame_size=3000):
        self.audio_paths = audio_paths
        self.emotion_labels = emotion_labels
        self.max_length = max_length
        self.device = device
        self.whisper_frame_size = whisper_frame_size  # Define the constant frame size for Whisper model input

    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        spectrogram_path = self.audio_paths[idx]
        spectrogram = np.load(spectrogram_path)
        spectrogram = torch.tensor(spectrogram, dtype=torch.float32).unsqueeze(0)  # Ensure it has a channel dimension

        # Pad the spectrogram to ensure uniform size
        if spectrogram.shape[-1] < self.max_length:
            padding_size = self.max_length - spectrogram.shape[-1]
            spectrogram = torch.nn.functional.pad(spectrogram, (0, padding_size), 'constant', 0)

        spectrogram = spectrogram.to(self.device)
        spectrogram = torch.squeeze(spectrogram)
        padded_output = F.pad(spectrogram.unsqueeze(0), (0, 3000-spectrogram.shape[-1]))
        
        # Generate features using Whisper
        features = whisper_model.encoder(padded_output)
        features = torch.flatten(features, start_dim=1, end_dim=2)
        features = torch.squeeze(features)

        emotion = torch.tensor(self.emotion_labels[idx], dtype=torch.long).to(self.device)
        return spectrogram, features, emotion

def get_train_loader(audio_paths, emotion_labels, max_length, device, batch_size=8, shuffle=True):
    dataset = EmotionDataset(audio_paths, emotion_labels, max_length, device, whisper_frame_size=3000)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# Example usage
df = pd.read_excel('spect_output/spectrogram_files.xlsx')
audio_paths = df['Filename'].tolist()
emotion_labels = np.random.randint(0, 6, size=len(audio_paths))  # Example: Random labels, replace with actual data
max_length = max([np.load(path).shape[-1] for path in audio_paths])  # Ensure dimension indexing is correct
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader = get_train_loader(audio_paths, emotion_labels, max_length, device)

# Iterating through the DataLoader
for spectrograms, features, emotions in train_loader:
    print(f"Spectrogram batch shape: {spectrograms.shape}")  # Should be [batch_size, 1, 80, max_length]
    print(f"Features batch shape: {features.shape}")
    print(f"Emotion labels shape: {emotions.shape}")
    break  # Test only the first batch

# # Example usage
# df = pd.read_excel('spect_output/spectrogram_files.xlsx')
# audio_paths = df['Filename'].tolist()
# emotion_labels = np.random.randint(0, 6, size=len(audio_paths))  # Example: Random labels, replace with actual data
# max_length = max([np.load(path).shape[-1] for path in audio_paths])  # Ensure dimension indexing is correct

# train_loader = get_train_loader(audio_paths, emotion_labels, max_length)

# # Iterating through the DataLoader
# for spectrograms, emotions in train_loader:
#     print(f"Spectrogram batch shape: {spectrograms.shape}")  # Should be [batch_size, 80, max_length]
#     print(f"Emotion labels shape: {emotions.shape}")
#     # break  # Test only the first batch
