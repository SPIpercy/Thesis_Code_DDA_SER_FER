from multiprocessing import Pipe
import pyaudio
import wave
import librosa
import numpy as np
import time
import torch
from transformers import Wav2Vec2FeatureExtractor, AutoModelForAudioClassification

class RealTimeSpeechEmotionRecognizer:
    def __init__(self):
        # Load the Wav2Vec2 model and feature extractor
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
        )
        self.model = AutoModelForAudioClassification.from_pretrained(
            "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
        )
        self.emotion_labels = self.model.config.id2label

        # Audio recording parameters
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.CHUNK = 512
        self.THRESHOLD = 500
        self.SILENCE_LIMIT = 1.5
        self.WAVE_OUTPUT_FILENAME = "SER_System\\temp_audio.wav"
        self.emotion_count = {}

        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK
        )

    def rms(self, data):
        """Calculate RMS of audio chunk."""
        array = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        return np.sqrt(np.mean(np.square(array)))

    def record_audio(self):
        """Record audio until silence is detected."""
        frames = []
        silence_start_time = None
        print("Voice detected! Starting recording...")
        
        while True:
            data = self.stream.read(self.CHUNK, exception_on_overflow=False)
            frames.append(data)
            rms_value = self.rms(data)
            
            if rms_value > self.THRESHOLD:
                silence_start_time = None
            elif silence_start_time is None:
                silence_start_time = time.time()
            
            if silence_start_time is not None:
                silence_duration = time.time() - silence_start_time
                if silence_duration > self.SILENCE_LIMIT:
                    print("Silence detected. Stopping recording.")
                    break

        # Save the recorded audio to a file
        wave_file = wave.open(self.WAVE_OUTPUT_FILENAME, 'wb')
        wave_file.setnchannels(self.CHANNELS)
        wave_file.setsampwidth(self.audio.get_sample_size(self.FORMAT))
        wave_file.setframerate(self.RATE)
        wave_file.writeframes(b''.join(frames))
        wave_file.close()

    def predict_emotion(self):
        """Predict emotion from the recorded audio."""
        audio_signal, sr = librosa.load(self.WAVE_OUTPUT_FILENAME, sr=self.RATE)
        inputs = self.feature_extractor(audio_signal, sampling_rate=sr, return_tensors="pt")
        
        with torch.no_grad():
            logits = self.model(**inputs).logits
        
        predicted_class_id = logits.argmax(-1).item()
        predicted_label = self.emotion_labels[predicted_class_id]
        return predicted_label

    def update_emotion_count(self, emotion):
        """Update the emotion count dictionary."""
        if emotion in self.emotion_count:
            self.emotion_count[emotion] += 1
        else:
            self.emotion_count[emotion] = 1

    def save_emotion_counts(self):
        """Save the final emotion counts to a file."""
        with open("SER_System\\total_emotion_counts.txt", "w") as file:
            file.write("Final emotion counts:\n")
            for emotion, count in self.emotion_count.items():
                file.write(f"{emotion}: {count}\n")
        print("Emotion counts written to 'SER_System\\total_emotion_counts.txt'")

    def start_recognition(self,conn=None):
        """Start the real-time speech emotion recognition."""
        print("Monitoring for voice activity...")
        try:
            while True:
                data = self.stream.read(self.CHUNK, exception_on_overflow=False)
                rms_value = self.rms(data)
                
                if rms_value > self.THRESHOLD:
                    self.record_audio()
                    predicted_emotion = self.predict_emotion()

                    print(f"Detected emotion: {predicted_emotion}")
                    self.update_emotion_count(predicted_emotion)
                    print(f"Emotion counts: {self.emotion_count}")

                    if conn:
                        conn.send(predicted_emotion)
            
        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            self.stop_recognition()

    def stop_recognition(self):
        """Stop the audio stream and save emotion counts."""
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()
        self.save_emotion_counts()
        print("Recording stopped.")

    def list_emotions(self):
        """List all available emotions."""
        print("Available emotions:")
        for idx, emotion in self.emotion_labels.items():
            print(f"{idx}: {emotion}")

if __name__ == "__main__":
    
    parent_conn, child_conn = Pipe()
    recognizer = RealTimeSpeechEmotionRecognizer()
    recognizer.list_emotions()  # List all available emotions
    recognizer.start_recognition(child_conn)
