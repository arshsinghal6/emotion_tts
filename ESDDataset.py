import os

import torch
from torch.utils.data import Dataset

import pandas as pd

class ESDDataset(Dataset):
    def __init__(self, dataset_dir: str):
        self.df = pd.DataFrame(columns=['audio_file_path', 'transcription', 'emotion_class'])
        self.dataset_dir = dataset_dir

        self.extract_data()

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        return self.df.loc[index]

    def extract_data(self):
        for root, dirs, files in os.walk(self.dataset_dir):
            for file in files:
                if file.endswith(".txt"):
                    with open(os.path.join(root, file)) as f:
                        for line in f.readlines():
                            audio_file_path, transcription, emotion_class = line.split('\t')
                            audio_file_path = os.path.join(root, audio_file_path) + '.wav'

                            row = [audio_file_path, transcription, emotion_class.strip()]
                            self.df.loc[len(self.df)] = row