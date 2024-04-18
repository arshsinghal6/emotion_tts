import whisper

model = whisper.load_model("base")
audio = whisper.load_audio("output_2.wav")
audio = whisper.pad_or_trim(audio)
mel = whisper.log_mel_spectrogram(audio)
print(mel.shape)

options = whisper.DecodingOptions()
result = whisper.decode(model, mel, options)
print(result.text)
# print(result["text"])