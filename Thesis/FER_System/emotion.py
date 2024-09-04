import cv2
from deepface import DeepFace

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start capturing video
cap = cv2.VideoCapture(0)

frame_count = 0  # Initialize frame counter
emotion_count = {}  # Initialize dictionary to keep track of detected emotions

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert grayscale frame to RGB format
    rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Increment frame count
    frame_count += 1

    # Capture emotion every 60 frames
    if frame_count % 60 == 0:
        for (x, y, w, h) in faces:
            # Extract the face ROI (Region of Interest)
            face_roi = rgb_frame[y:y + h, x:x + w]

            # Perform emotion analysis on the face ROI
            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

            # Determine the dominant emotion
            emotion = result[0]['dominant_emotion']

            # Output the detected emotion to the terminal
            print(f"Detected emotion: {emotion}")

            # Draw rectangle around face and label with predicted emotion
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # Update the emotion count in the dictionary
            if emotion in emotion_count:
                emotion_count[emotion] += 1
            else:
                emotion_count[emotion] = 1

            # Print the current emotion count
            print(f"Emotion counts: {emotion_count}")

    # Display the resulting frame
    cv2.imshow('Real-time Emotion Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()

# Write final emotion counts to a file
with open("emotion_counts.txt", "w") as file:
    file.write("Final emotion counts:\n")
    for emotion, count in emotion_count.items():
        file.write(f"{emotion}: {count}\n")

print("Emotion counts written to 'emotion_counts.txt'")
