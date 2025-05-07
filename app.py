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
processing = False

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

def gesture_worker():
    global processing
    import cv2
    import mediapipe as mp

    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands.Hands()

    while processing:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame here (gesture detection)
        results = mp_hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # You can log or store results somewhere here

    cap.release()

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

@app.route('/start')
def start_processing():
    global processing
    if not processing:
        processing = True
        thread = threading.Thread(target=gesture_worker)
        thread.daemon = True
        thread.start()
    return "Gesture processing started."

@app.route('/stop')
def stop_processing():
    global processing
    processing = False
    return "Gesture processing stopped."

@app.route('/static/js/client_side_implementation.js')
def client_side_js():
    return Response("""
    // JavaScript implementation for client-side processing
    let videoElement, canvasElement, ctx, model;
    let isProcessing = false;
    let fingerCount = 0;

    document.addEventListener('DOMContentLoaded', function() {
        videoElement = document.getElementById('webcam');
        canvasElement = document.getElementById('output-canvas');
        ctx = canvasElement.getContext('2d');
        
        document.getElementById('start-button').addEventListener('click', startDetection);
        
        // Initialize camera
        initCamera();
    });

    async function initCamera() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480 }
            });
            videoElement.srcObject = stream;
            videoElement.style.visibility = 'visible';
            document.getElementById('status').innerHTML = 'Camera initialized. Click Start Detection.';
        } catch (error) {
            document.getElementById('status').innerHTML = 'Error accessing camera: ' + error.message;
        }
    }

    async function startDetection() {
        if (isProcessing) return;
        
        document.getElementById('status').innerHTML = 'Loading hand detection model...';
        
        try {
            // Load the MediaPipe Hands model
            const model = await handPoseDetection.createDetector(
                handPoseDetection.SupportedModels.MediaPipeHands,
                {
                    runtime: 'mediapipe',
                    modelType: 'full',
                    maxHands: 1
                }
            );
            
            document.getElementById('status').innerHTML = 'Model loaded. Processing...';
            isProcessing = true;
            
            // Start detection loop
            detectHands(model);
        } catch (error) {
            document.getElementById('status').innerHTML = 'Error loading model: ' + error.message;
        }
    }

    async function detectHands(model) {
        if (!isProcessing) return;
        
        // Draw video frame to canvas
        ctx.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);
        
        try {
            // Detect hands
            const hands = await model.estimateHands(canvasElement);
            
            if (hands.length > 0) {
                const hand = hands[0];
                
                // Draw hand landmarks
                drawLandmarks(hand.keypoints);
                
                // Count extended fingers
                fingerCount = countFingers(hand.keypoints);
                document.getElementById('finger-count').innerHTML = 'Fingers: ' + fingerCount;
            } else {
                document.getElementById('finger-count').innerHTML = 'No hand detected';
            }
        } catch (error) {
            console.error('Detection error:', error);
        }
        
        // Continue detection loop
        requestAnimationFrame(() => detectHands(model));
    }

    function drawLandmarks(keypoints) {
        // Draw connections
        const connections = [
            [0, 1], [1, 2], [2, 3], [3, 4], // thumb
            [0, 5], [5, 6], [6, 7], [7, 8], // index finger
            [0, 9], [9, 10], [10, 11], [11, 12], // middle finger
            [0, 13], [13, 14], [14, 15], [15, 16], // ring finger
            [0, 17], [17, 18], [18, 19], [19, 20] // pinky
        ];
        
        // Draw points
        ctx.fillStyle = 'lime';
        keypoints.forEach(point => {
            ctx.beginPath();
            ctx.arc(point.x, point.y, 5, 0, 2 * Math.PI);
            ctx.fill();
        });
        
        // Draw connections
        ctx.strokeStyle = 'yellow';
        ctx.lineWidth = 2;
        connections.forEach(([i, j]) => {
            const start = keypoints[i];
            const end = keypoints[j];
            if (start && end) {
                ctx.beginPath();
                ctx.moveTo(start.x, start.y);
                ctx.lineTo(end.x, end.y);
                ctx.stroke();
            }
        });
    }

    function countFingers(keypoints) {
        if (!keypoints || keypoints.length < 21) return 0;
        
        // MediaPipe hand landmark indices
        const fingertips = [4, 8, 12, 16, 20]; // thumb, index, middle, ring, pinky
        const mcp = [1, 5, 9, 13, 17]; // metacarpophalangeal joints
        
        let extendedFingers = 0;
        
        // Check thumb (special case)
        const thumbTip = keypoints[fingertips[0]];
        const thumbMcp = keypoints[mcp[0]];
        const wrist = keypoints[0];
        
        // Thumb is extended if its tip is to the side of the MCP joint
        if (thumbTip.x < thumbMcp.x) {
            extendedFingers++;
        }
        
        // Check other 4 fingers
        for (let i = 1; i < 5; i++) {
            const tip = keypoints[fingertips[i]];
            const pip = keypoints[fingertips[i] - 2]; // Proximal Interphalangeal joint
            
            // A finger is extended if its tip is higher than its PIP joint
            if (tip.y < pip.y) {
                extendedFingers++;
            }
        }
        
        return extendedFingers;
    }
    """, mimetype='application/javascript')

if __name__ == '__main__':
    # Check if the environment is not production
    if os.environ.get('ENVIRONMENT') != 'production':
        # Start the finger detection in a separate thread only in non-production environments
        t = threading.Thread(target=detect_fingers)
        t.daemon = True
        t.start()
    
    # Start the Flask app
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)), debug=False)