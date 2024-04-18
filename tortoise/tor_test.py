from TTS.api import TTS
import wave
import numpy as np
# tts = TTS("tts_models/en/multi-dataset/tortoise-v2")

# print(tts)

# cloning `lj` voice from `TTS/tts/utils/assets/tortoise/voices/lj`
# # with custom inference settings overriding defaults.
# tts.tts_to_file(text="Hello, my name is Manmay , how are you?",
#                 file_path="output.wav",
#                 voice_dir="path/to/tortoise/voices/dir/",
#                 speaker="lj",
#                 num_autoregressive_samples=1,
#                 diffusion_iterations=10)

# # Using presets with the same voice
# tts.tts_to_file(text="Hello, my name is Manmay , how are you?",
#                 file_path="output.wav",
#                 voice_dir="path/to/tortoise/voices/dir/",
#                 speaker="lj",
#                 preset="ultra_fast")

# # Random voice generation
# tts.tts_to_file(text="Hello, my name is Manmay , how are you?",
#                 file_path="output.wav")

# from TTS.api import TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)

# generate speech by cloning a voice using default settings
tts.tts_to_file(text="It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent. Oh wait, I am going to be silent!",
                file_path="output_2.wav",
                speaker="Ana Florence",
                language="en",
                split_sentences=True
                )

# Open the WAV file
# with wave.open('output.wav', 'rb') as wav_file:
#     # Extract audio frames
#     frames = wav_file.readframes(wav_file.getnframes())
    
#     # Convert audio bytes to a NumPy array
#     # Note: np.int16 is used for 16-bit WAV files. This may vary depending on the file.
#     audio_array = np.frombuffer(frames, dtype=np.int16)

#     # Normalize the array to range between -1 and 1 if necessary
#     max_val = np.max(np.abs(audio_array))
#     normalized_array = audio_array / max_val

# # Display the shape of the array
# print(audio_array.shape)

# # You might want to reshape the array based on the number of channels
# num_channels = wav_file.getnchannels()
# if num_channels > 1:
#     # Reshape the array into two columns, each column representing one channel
#     reshaped_array = np.reshape(audio_array, (-1, num_channels))
#     print(reshaped_array.shape)
# else:
#     print("This is a mono file.")
