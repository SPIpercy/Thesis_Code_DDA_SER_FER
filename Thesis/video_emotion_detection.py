from collections import Counter
import cv2
from deepface import DeepFace
import time
from multiprocessing import Pipe

class RealTimeFaicalEmotionDetector:
    def __init__(self):
        # Load face cascade classifier
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        self.emotion_count = {}
        # Track time for emotion detection
        self.last_detection_time = time.time()
        self.last_five_second_check = time.time()

        self.five_second_emotions = {}

    def capture_frame(self):
        """Captures a frame from the video feed."""
        ret, frame = self.cap.read()
        return ret, frame

    def preprocess_frame(self, frame):
        """Converts the frame to grayscale and then to RGB."""
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)
        return gray_frame, rgb_frame

    def detect_faces(self, gray_frame):
        """Detects faces in the grayscale frame."""
        faces = self.face_cascade.detectMultiScale(
            gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        return faces

    def analyze_emotion(self, face_roi):
        """Analyzes the emotion in the face ROI using DeepFace."""
        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
        dominant_emotion = result[0]['dominant_emotion']
        return dominant_emotion

    def update_emotion_count(self, emotion, emotion_dict):
        """Updates the specified emotion count dictionary."""
        if emotion in emotion_dict:
            emotion_dict[emotion] += 1
        else:
            emotion_dict[emotion] = 1

    def draw_face_rectangle(self, frame, x, y, w, h, emotion):
        """Draws a rectangle around the detected face and labels it with the predicted emotion."""
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    def process_frame(self, frame):
        """Processes a single frame: detects faces, analyzes emotion, and updates the frame with results."""
        gray_frame, rgb_frame = self.preprocess_frame(frame)
        faces = self.detect_faces(gray_frame)

        for (x, y, w, h) in faces:
            face_roi = rgb_frame[y:y + h, x:x + w]
            emotion = self.analyze_emotion(face_roi)
            self.draw_face_rectangle(frame, x, y, w, h, emotion)
            self.update_emotion_count(emotion, self.emotion_count)
            self.update_emotion_count(emotion, self.five_second_emotions)

        return frame

    def save_emotion_counts(self):
        """Save the final emotion counts to a file."""
        with open("FER_System\\total_emotion_counts.txt", "w") as file:
            file.write("Final emotion counts:\n")
            for emotion, count in self.emotion_count.items():
                file.write(f"{emotion}: {count}\n")
        print("Emotion counts written to 'FER_System\\total_emotion_counts.txt'")

    def most_common_emotion(self, conn=None):
        """Determines the most common emotion in the last 5 seconds."""
        if len(self.five_second_emotions) > 0:
            most_common_emotion = Counter(self.five_second_emotions).most_common(1)[0][0]

            if conn:
                conn.send(most_common_emotion)

            self.five_second_emotions.clear()

    def start_detection(self, child_conn):
        """Starts the real-time emotion detection."""
        try:
            while True:
                ret, frame = self.capture_frame()

                if not ret:
                    print("Failed to capture frame")
                    break

                current_time = time.time()
                
                if current_time - self.last_detection_time >= 1:
                    frame = self.process_frame(frame)
                    self.last_detection_time = current_time

                cv2.imshow('Real-time Emotion Detection', frame)

                if current_time - self.last_five_second_check >= 5:
                    self.most_common_emotion(child_conn)
                    self.last_five_second_check = current_time

                if cv2.waitKey(1) & 0xFF == ord('m'):
                    break

        except KeyboardInterrupt:
            print("Interrupted by user")

        finally:
            self.stop_recognition()

    def stop_recognition(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.save_emotion_counts()

if __name__ == "__main__":
    parent_conn, child_conn = Pipe()
    detector = RealTimeFaicalEmotionDetector()
    detector.start_detection(child_conn)
