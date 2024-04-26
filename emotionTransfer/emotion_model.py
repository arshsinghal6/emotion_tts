import torch
import torch.nn as nn
import torch.nn.functional as F
from data_loader import get_train_loader

class AudioTextEmotionModel(nn.Module):
    def __init__(self, num_emotions, embedding_dim, num_words, word_embedding_dim):
        super(AudioTextEmotionModel, self).__init__()
        
        # Convolutional layers for mel spectrogram input
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=(1, 1))
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1))
        
        # Emotion embedding layer
        self.emotion_embedding = nn.Embedding(num_emotions, embedding_dim)
        
        # Text embedding layer
        self.text_embedding = nn.Embedding(num_words, word_embedding_dim)
        
        # Fully connected layers for prediction
        self.fc1 = nn.Linear(embedding_dim + word_embedding_dim + 64*38*12, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_emotions)
        
    def forward(self, audio_input, emotion_input, text_input):
        # Process mel spectrogram input through convolutional layers
        x = F.relu(self.conv1(audio_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64*38*12)  # Flatten
        
        # Emotion embedding
        emotion_embedding = self.emotion_embedding(emotion_input)
        
        # Text embedding
        text_embedding = self.text_embedding(text_input)
        
        # Concatenate audio, emotion, and text embeddings
        x = torch.cat((x, emotion_embedding, text_embedding), dim=1)
        
        # Fully connected layers for prediction
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = self.fc3(x)
        
        return output

# Define model hyperparameters
num_emotions = 6
embedding_dim = 32
num_words = 10000  # Number of unique words in vocabulary
word_embedding_dim = 64

# Initialize the model
model = AudioTextEmotionModel(num_emotions, embedding_dim, num_words, word_embedding_dim)

# Define the loss functions (example)
emotion_criterion = nn.CrossEntropyLoss()
text_criterion = nn.CrossEntropyLoss()


# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Define number of epochs
num_epochs = 10

train_loader = get_train_loader()

# Training loop
for epoch in range(num_epochs):
    # Initialize total loss for the epoch
    total_loss = 0
    
    # Iterate over the dataset (assuming you have a DataLoader named 'train_loader')
    for audio_input, emotion_input, text_input, emotion_label, text_label in train_loader:
        # Clear gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(audio_input, emotion_input, text_input)
        
        # Calculate loss
        emotion_loss = emotion_criterion(output, emotion_label)
        text_loss = text_criterion(output, text_label)
        loss = emotion_loss + text_loss
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        # Accumulate total loss for the epoch
        total_loss += loss.item()
    
    # Calculate average loss for the epoch
    avg_loss = total_loss / len(train_loader)
    
    # Print training progress
    print(f"Epoch [{epoch+1}/{num_epochs}], Avg. Loss: {avg_loss:.4f}")