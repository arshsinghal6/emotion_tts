import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import whisper
import torch.nn.functional as F
whisper_model = whisper.load_model("base")

class EmotionDataset(Dataset):
    def __init__(self, audio_paths, emotion_labels, max_length, device, whisper_frame_size=3000):
        self.audio_paths = audio_paths
        self.emotion_labels = emotion_labels
        self.max_length = max_length
        self.device = device
        self.whisper_frame_size = whisper_frame_size 

    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        spectrogram_path = self.audio_paths[idx]
        spectrogram = np.load(spectrogram_path)
        spectrogram = torch.tensor(spectrogram, dtype=torch.float32).unsqueeze(0) 


        if spectrogram.shape[-1] < self.max_length:
            padding_size = self.max_length - spectrogram.shape[-1]
            spectrogram = torch.nn.functional.pad(spectrogram, (0, padding_size), 'constant', 0)

        spectrogram = spectrogram.to(self.device)
        spectrogram = torch.squeeze(spectrogram)
        padded_output = F.pad(spectrogram.unsqueeze(0), (0, 3000-spectrogram.shape[-1]))
        
        features = whisper_model.encoder(padded_output)
        features = torch.flatten(features, start_dim=1, end_dim=2)
        features = torch.squeeze(features)

        emotion = torch.tensor(self.emotion_labels[idx], dtype=torch.long).to(self.device)
        return spectrogram, features, emotion

def get_train_loader(audio_paths, emotion_labels, max_length, device, batch_size=8, shuffle=True):
    dataset = EmotionDataset(audio_paths, emotion_labels, max_length, device, whisper_frame_size=3000)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


df = pd.read_excel('spect_output/spectrogram_files.xlsx')
audio_paths = df['Filename'].tolist()
emotion_labels = np.random.randint(0, 6, size=len(audio_paths)) 
max_length = max([np.load(path).shape[-1] for path in audio_paths]) 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader = get_train_loader(audio_paths, emotion_labels, max_length, device)

# Test
for spectrograms, features, emotions in train_loader:
    print(f"Spectrogram batch shape: {spectrograms.shape}") 
    print(f"Features batch shape: {features.shape}")
    print(f"Emotion labels shape: {emotions.shape}")
    break
