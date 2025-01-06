import cv2
import numpy as np
from keras.models import model_from_json
import time
import matplotlib.pyplot as plt

# Define labels for emotions and cheat status
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised", 7: "Cheat"}
cheat_label = {0: "Not Cheating", 1: "Cheating"}

# Load the model architecture
try:
    json_file = open(r"C:\Users\Jeeval Bhatia\Downloads\Emotion-Detector--CNN--main\Emotion-Detector--CNN--main\model\emotion_model.json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    emotion_model = model_from_json(loaded_model_json)

    # Load the model weights
    emotion_model.load_weights(r"C:\Users\Jeeval Bhatia\Downloads\Emotion-Detector--CNN--main\Emotion-Detector--CNN--main\model\emotion_model.h5")
    print("Model successfully loaded!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Start the webcam feed
cap = cv2.VideoCapture(0)

# Haarcascade path for face detection
haarcascade_path = r"C:\Users\Jeeval Bhatia\Downloads\Emotion-Detector--CNN--main\Emotion-Detector--CNN--main\haarcascades\haarcascade_frontalface_default.xml"
face_detector = cv2.CascadeClassifier(haarcascade_path)

# Variables to store data for graphing
times = []
emotion_data = []
cheat_data = []

# Begin video processing
start_time = time.time()  # Start the timer
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture video frame. Exiting.")
        break

    # Resize frame for consistency
    frame = cv2.resize(frame, (1280, 720))

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # Process each face detected
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (0, 255, 0), 4)

        # Extract and preprocess the region of interest (ROI)
        roi_gray = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)

        try:
            # Predict using the model for emotion
            emotion_prediction = emotion_model.predict(cropped_img)

            # Cheat detection (Assume 8th index is 'cheat')
            if len(emotion_prediction[0]) == 8:  # Ensure model supports cheat detection
                cheat_probability = emotion_prediction[0][7]
                cheat_threshold = 0.6  # Set this threshold for "Cheating"
                is_cheating = int(cheat_probability > cheat_threshold)  # Cheating or Not
                cheat_status = cheat_label[is_cheating]
                emotion_prediction[0][7] = 0  # Ignore "cheat" for emotion prediction
            else:
                cheat_status = "Not Cheating"  # Default to "Not Cheating"

            # Emotion detection (ignore cheat category)
            maxindex = int(np.argmax(emotion_prediction[0][:7]))  # Ignore 8th index (cheat)
            emotion = emotion_dict.get(maxindex, "Unknown")

            # Add data to list every 1 second
            current_time = time.time()
            if current_time - start_time >= 1:  # Collect data every second
                start_time = current_time  # Reset the timer
                times.append(current_time)
                emotion_data.append(emotion)
                cheat_data.append(cheat_status)

            # Display the emotion and cheat status on the frame
            cv2.putText(frame, f"Emotion: {emotion}", (x + 5, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Status: {cheat_status}", (x + 5, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

        except Exception as e:
            print(f"Error during prediction: {e}")
            cv2.putText(frame, "Error", (x + 5, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Show the video feed with both outputs (Emotion and Cheat Status)
    cv2.imshow("Emotion and Cheat Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Plot the bar graph once 'q' is pressed
if times:
    plt.figure(figsize=(12, 6))

    # Count the frequency of each emotion detected
    emotion_counts = {emotion: emotion_data.count(emotion) for emotion in emotion_dict.values() if emotion != "Cheat"}
    
    # Create the bar graph for emotions
    plt.bar(emotion_counts.keys(), emotion_counts.values(), color='blue')
    plt.title("Emotion Detection Over Time")
    plt.xlabel("Emotion")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)

    # Plot Cheat status as a bar chart (use a binary value for "Not Cheating" and "Cheating")
    cheat_binary = [1 if status == "Cheating" else 0 for status in cheat_data]
    cheat_count = { "Not Cheating": cheat_binary.count(0), "Cheating": cheat_binary.count(1) }
    
    plt.figure(figsize=(6, 6))
    plt.bar(cheat_count.keys(), cheat_count.values(), color='orange')
    plt.title("Cheat Detection Frequency")
    plt.xlabel("Cheat Status")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()
