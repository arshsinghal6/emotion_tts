import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset

class EmotionDataset(Dataset):
    def __init__(self, audio_paths, text_data, max_length):
        self.pairs = [(path, emotion) for path in audio_paths for emotion in range(6)]  # 6 emotions
        self.text_data = np.repeat(text_data, 6)  # Repeat text data for each emotion
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        spectrogram_path, emotion = self.pairs[idx]
        spectrogram = torch.load(spectrogram_path)
        
        # we pad the spectrogram if we need to
        if spectrogram.shape[1] < self.max_length:
            spectrogram = torch.cat([spectrogram, torch.zeros((80, self.max_length - spectrogram.shape[1]))], dim=1)

        text_input = self.text_data[idx]
        return spectrogram, emotion, text_input

def get_train_loader(audio_paths, text_data, max_length):
    dataset = EmotionDataset(audio_paths, text_data, max_length)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=False)  # change batch_size if needed
    return train_loader

df = pd.read_excel('output_text.xlsx')
text_indices = np.arange(len(df)) # attempt to get text length (not working)
audio_paths = df['Spectrogram Path'].tolist()
max_length = max(torch.load(path).shape[1] for path in audio_paths)
train_loader = get_train_loader(audio_paths, text_indices, max_length)

for batch in train_loader:
    spectrograms, emotions, text_indices = batch
    print(f"Spectrogram batch shape: {spectrograms.shape}")
    print(f"Emotion labels: {emotions}")
    print(f"Text indices: {text_indices}")
    print("----------")
