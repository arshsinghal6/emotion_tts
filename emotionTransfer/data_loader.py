import torch
from torch.utils.data import DataLoader, Dataset

class EmotionDataset(Dataset):
    def __init__(self, audio_data, emotion_labels, text_data, text_labels):
        self.audio_data = audio_data
        self.emotion_labels = emotion_labels
        self.text_data = text_data
        self.text_labels = text_labels
        
    def __len__(self):
        return len(self.audio_data)
    
    def __getitem__(self, idx):
        audio_input = self.audio_data[idx]
        emotion_input = self.emotion_labels[idx]
        text_input = self.text_data[idx]
        emotion_label = self.emotion_labels[idx]
        text_label = self.text_labels[idx]
        return audio_input, emotion_input, text_input, emotion_label, text_label

def get_train_loader():
    # Assuming you have audio data, emotion labels, text data, and text labels
    audio_data = ...  # Tensor of shape (num_samples, 1, 80, 300)
    emotion_labels = ...  # Tensor of emotion labels (e.g., 0, 1, 2, ...)
    text_data = ...  # Tensor of text data (e.g., word indices)
    text_labels = ...  # Tensor of text labels (e.g., word indices)

    # Create an instance of the custom dataset
    dataset = EmotionDataset(audio_data, emotion_labels, text_data, text_labels)

    # Define batch size
    batch_size = 32

    # Create the DataLoader
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
