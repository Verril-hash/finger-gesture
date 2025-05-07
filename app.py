from flask import Flask, render_template, Response
import os

app = Flask(__name__)

@app.route('/')
def index():
    # Redirect to client-side implementation by default
    return render_template('client_side.html')

@app.route('/static/js/client_side_implementation.js')
def client_side_js():
    # Return the JavaScript for client-side processing
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
    # Start the Flask app
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)), debug=False)