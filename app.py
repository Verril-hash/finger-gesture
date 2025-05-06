from flask import Flask, Response, render_template
import cv2
import mediapipe as mp
import numpy as np
import threading
import time
import os

app = Flask(__name__)

# Global variables
output_frame = None
lock = threading.Lock()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Draw hand landmarks
mp_draw = mp.solutions.drawing_utils

# Finger tip landmark indices
finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
thumb_tip = 4

def detect_fingers():
    global output_frame, lock
    
    # Start webcam - note: this won't work in a web environment
    # We're keeping the structure for local testing
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("No camera available - generating test frames")
        # If no camera is available (like on deployment platforms)
        # we'll generate test frames with a message
        while True:
            # Create a blank frame with a message
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "No camera access on server", (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, "This app requires client-side camera", (50, 280), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Encode the frame for streaming
            with lock:
                output_frame = frame.copy()
            time.sleep(0.033)  # ~30 FPS
    else:
        # If camera is available (local testing)
        while True:
            success, image = cap.read()
            if not success:
                break

            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            finger_count = 0

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    landmarks = hand_landmarks.landmark
                    h, w, _ = image.shape
                    finger_states = []

                    # Thumb
                    if landmarks[thumb_tip].x < landmarks[thumb_tip - 1].x:
                        finger_states.append(1)
                    else:
                        finger_states.append(0)

                    # Other 4 fingers
                    for tip in finger_tips:
                        if landmarks[tip].y < landmarks[tip - 2].y:
                            finger_states.append(1)
                        else:
                            finger_states.append(0)

                    finger_count = sum(finger_states)

                    cv2.putText(image, f'Fingers: {finger_count}', (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

            with lock:
                output_frame = image.copy()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()

def generate():
    global output_frame, lock
    while True:
        with lock:
            if output_frame is None:
                continue
            (flag, encoded_image) = cv2.imencode(".jpg", output_frame)
            if not flag:
                continue
        
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
              bytearray(encoded_image) + b'\r\n')
        time.sleep(0.033)  # ~30 FPS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')
                   
@app.route('/client')
def client_side():
    return render_template('client_side.html')

if __name__ == '__main__':
    # Start the finger detection in a separate thread
    t = threading.Thread(target=detect_fingers)
    t.daemon = True
    t.start()
    
    # Start the Flask app
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)