import pyaudio
import wave
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal
import time
from speechbrain.pretrained import EncoderClassifier

# Initialize the emotion classifier from speechbrain
classifier = EncoderClassifier.from_hparams(source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP", savedir="tmpdir")

# Define the mapping of class indices to emotion labels
emotion_labels = {
    0: 'female_angry',
    1: 'female_calm',
    2: 'female_fearful',
    3: 'female_happy',
    4: 'female_sad',
    5: 'male_angry',
    6: 'male_calm',
    7: 'male_fearful',
    8: 'male_happy',
    9: 'male_sad'
}

# Audio recording parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # Sample rate expected by most speech emotion models
CHUNK = 1024  # Buffer size
THRESHOLD = 500  # Adjust this threshold based on your environment
SILENCE_LIMIT = 1  # Time in seconds to wait before stopping the recording after silence is detected
WAVE_OUTPUT_FILENAME = "temp_audio.wav"

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Start the audio stream for monitoring
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

print("Monitoring for voice activity...")

emotion_count = {}  # Dictionary to keep track of detected emotions

def rms(data):
    """Calculate RMS of audio chunk."""
    array = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    return np.sqrt(np.mean(np.square(array)))

def normalize(audio_signal):
    """Normalize the audio signal to the range [-1, 1]."""
    return audio_signal / np.max(np.abs(audio_signal))

def apply_high_pass_filter(audio_signal, sr, cutoff=80):
    """Apply a high-pass filter to remove low-frequency noise."""
    return librosa.effects.preemphasis(audio_signal, coef=cutoff/sr)

def basic_noise_reduction(audio_signal, sr):
    """Apply a basic noise reduction using a high-pass filter."""
    b, a = scipy_signal.butter(1, 100/(sr/2), btype='highpass')  # High-pass filter at 100 Hz
    return scipy_signal.filtfilt(b, a, audio_signal)

try:
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        rms_value = rms(data)
        
        if rms_value > THRESHOLD:
            print("Voice detected! Starting recording...")
            frames = []
            silence_start_time = None
            
            while True:
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
                rms_value = rms(data)
                
                if rms_value > THRESHOLD:
                    silence_start_time = None  # Reset silence timer when voice is detected
                elif silence_start_time is None:
                    silence_start_time = time.time()  # Start silence timer
                
                if silence_start_time is not None:
                    silence_duration = time.time() - silence_start_time
                    if silence_duration > SILENCE_LIMIT:
                        print("Silence detected. Stopping recording.")
                        break
            
            # Save the recorded audio to a temporary file
            wave_file = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
            wave_file.setnchannels(CHANNELS)
            wave_file.setsampwidth(audio.get_sample_size(FORMAT))
            wave_file.setframerate(RATE)
            wave_file.writeframes(b''.join(frames))
            wave_file.close()
            
            # Load the audio file
            audio_signal, sr = librosa.load(WAVE_OUTPUT_FILENAME, sr=RATE)
            
            # Apply high-pass filtering, normalize and reduce noise
            audio_signal = apply_high_pass_filter(audio_signal, sr)
            audio_signal = normalize(audio_signal)
            audio_signal = basic_noise_reduction(audio_signal, sr)
            
            # Plot the processed audio signal
            plt.figure(figsize=(10, 4))
            plt.plot(audio_signal)
            plt.title('Normalized and Noise Reduced Audio Signal')
            plt.ylim([-1.1, 1.1])
            plt.show()

            # Pass the processed audio signal to the emotion recognition model
            prediction = classifier.classify_file(WAVE_OUTPUT_FILENAME)
            emotion_index = prediction[3].item()  # The class index is the 4th item in the returned tuple
            emotion_label = emotion_labels[emotion_index]

            # Output the detected emotion label
            print(f"Detected emotion: {emotion_label}")

            # Update the emotion count in the dictionary
            if emotion_label in emotion_count:
                emotion_count[emotion_label] += 1
            else:
                emotion_count[emotion_label] = 1

            # Print the current emotion count
            print(f"Emotion counts: {emotion_count}")

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    # Stop and close the audio stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Write final emotion counts to a file
    with open("emotion_counts.txt", "w") as file:
        file.write("Final emotion counts:\n")
        for emotion, count in emotion_count.items():
            file.write(f"{emotion}: {count}\n")

    print("Emotion counts written to 'emotion_counts.txt'")
    print("Recording stopped.")
