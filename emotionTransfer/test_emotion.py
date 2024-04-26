from transformers import AutoModelForAudioClassification
import librosa, torch

#load model
model = AutoModelForAudioClassification.from_pretrained("3loi/SER-Odyssey-Baseline-WavLM-Categorical-Attributes", trust_remote_code=True)

#get mean/std
mean = model.config.mean
std = model.config.std


#load an audio file
audio_path = "IEMOCAP_happy.wav"
raw_wav, _ = librosa.load(audio_path, sr=model.config.sampling_rate)

#normalize the audio by mean/std
norm_wav = (raw_wav - mean) / (std+0.000001)

#generate the mask
mask = torch.ones(1, len(norm_wav))

#batch it (add dim)
wavs = torch.tensor(norm_wav).unsqueeze(0)


#predict
with torch.no_grad():
    pred = model(wavs, mask)

print(model.config.id2label)  
print(pred)
#{0: 'Angry', 1: 'Sad', 2: 'Happy', 3: 'Surprise', 4: 'Fear', 5: 'Disgust', 6: 'Contempt', 7: 'Neutral'}
#tensor([[0.0015, 0.3651, 0.0593, 0.0315, 0.0600, 0.0125, 0.0319, 0.4382]])

#convert logits to probability
probabilities = torch.nn.functional.softmax(pred, dim=1)
print(probabilities)
#[[0.0015, 0.3651, 0.0593, 0.0315, 0.0600, 0.0125, 0.0319, 0.4382]]