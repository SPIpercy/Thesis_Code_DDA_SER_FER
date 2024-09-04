# run_concurrently.py
import multiprocessing
import time
from audio_emotion_detection import RealTimeSpeechEmotionRecognizer
from video_emotion_detection import RealTimeFaicalEmotionDetector

def run_audio_detection(conn):
    recognizer = RealTimeSpeechEmotionRecognizer()
    recognizer.start_recognition(conn)

def run_video_detection(conn):
    detector = RealTimeFaicalEmotionDetector()
    detector.start_detection(conn)

def save_results_to_txt(emotion, output_path):
    try:
        with open(output_path, "w") as file:
            if emotion:
                file.write(emotion)
                
    except Exception as e:
        print(f"Failed to write to file {output_path}: {e}")

if __name__ == "__main__":
    # Create pipes for communication
    audio_conn_parent, audio_conn_child = multiprocessing.Pipe()
    video_conn_parent, video_conn_child = multiprocessing.Pipe()

    # Create processes for audio and video detection
    audio_process = multiprocessing.Process(target=run_audio_detection, args=(audio_conn_child,))
    video_process = multiprocessing.Process(target=run_video_detection, args=(video_conn_child,))

    # Start the processes
    audio_process.start()
    video_process.start()

    output_path = "D:\\Steam\\steamapps\\common\\skyrim\\emotion_results.txt"


    try:
        
        while True:
            audio_emotion = ""
            video_emotion = ""
            # Check for new data from the audio process
            if audio_conn_parent.poll():
                audio_emotion = audio_conn_parent.recv()
                print(f"Audio detected emotion: {audio_emotion}")
            
            # Check for new data from the video process
            if video_conn_parent.poll():
                video_emotion = video_conn_parent.recv()
                print(f"Video detected emotions: {video_emotion}")

            # Determine the result emotion based on available data
            if audio_emotion and video_emotion:
                # If both have results, prioritize audio
                result_emotion = audio_emotion
            elif video_emotion:
                # Only video has a result
                result_emotion = video_emotion
            elif audio_emotion:
                # Only audio has a result
                result_emotion = audio_emotion
            else:
                # Neither has a result, default to "neutral"
                result_emotion = "neutral"

            # Save the results to a text file
            save_results_to_txt(result_emotion, output_path)

            time.sleep(5)
            
    except KeyboardInterrupt:
        print("Main process interrupted.")

    finally:
        # Terminate processes when done
        audio_process.terminate()
        video_process.terminate()
        audio_process.join()
