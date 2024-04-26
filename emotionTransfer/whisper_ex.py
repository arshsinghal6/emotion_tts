import whisper
import matplotlib.pyplot as plt

model = whisper.load_model("base")
audio = whisper.load_audio("output.wav")
audio = whisper.pad_or_trim(audio)
mel = whisper.log_mel_spectrogram(audio)

plt.imshow(mel)
plt.show()

options = whisper.DecodingOptions()
result = whisper.decode(model, mel, options)
print(result.text)
# print(result["text"])