
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset

class EmotionDataset(Dataset):
    def __init__(self, audio_paths, text_data, max_length):
        self.pairs = [(path, emotion) for path in audio_paths for emotion in range(6)]  # 6 emotions
        self.text_data = text_data
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        spectrogram_path, emotion = self.pairs[idx]
        spectrogram = torch.load(spectrogram_path)
        
        # here we pad spectrogram if needed
        if spectrogram.shape[1] < self.max_length:
            spectrogram = torch.cat([spectrogram, torch.zeros((80, self.max_length - spectrogram.shape[1]))], dim=1)

        text_input = self.text_data[idx // 6]
        return spectrogram, emotion, text_input

def get_train_loader(audio_paths, text_data, max_length):
    dataset = EmotionDataset(audio_paths, text_data, max_length)
    train_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    return train_loader

df = pd.read_excel('output_text.xlsx')
text_indices = np.arange(len(df))
audio_paths = df['Spectrogram Path'].tolist()
max_length = max(torch.load(path).shape[1] for path in audio_paths)
train_loader = get_train_loader(audio_paths, text_indices, max_length)

for spectrogram, emotion, text_index in train_loader:
    print(f"Spectrogram shape: {spectrogram.shape}, Emotion label: {emotion.item()}, Text index: {text_index.item()}")
    print("----------")