import librosa

my_audio_as_np_array, my_sample_rate= librosa.load("output.wav")

# step2 - converting audio np array to spectrogram
spec = librosa.feature.melspectrogram(y=my_audio_as_np_array,
                                        sr=my_sample_rate, 
                                            n_fft=2048, 
                                            hop_length=512, 
                                            win_length=None, 
                                            window='hann', 
                                            center=True, 
                                            pad_mode='reflect', 
                                            power=2.0,
                                     n_mels=80)
print(spec.shape)
# step3 converting mel-spectrogrma back to wav file
res = librosa.feature.inverse.mel_to_audio(spec, 
                                           sr=my_sample_rate, 
                                           n_fft=2048, 
                                           hop_length=512, 
                                           win_length=None, 
                                           window='hann', 
                                           center=True, 
                                           pad_mode='reflect', 
                                           power=2.0, 
                                           n_iter=32)

# step4 - save it as a wav file
import soundfile as sf
sf.write("test1.wav", res, samplerate=my_sample_rate)

print(res.shape)

raw_wav, _ = librosa.load("test1.wav", sr=my_sample_rate)
print(raw_wav.shape)
