let videoElement, canvasElement, ctx;
let isProcessing = false;
let fingerCount = 0;
let model = null;

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
            video: { 
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: 'user'
            }
        });
        videoElement.srcObject = stream;
        videoElement.style.visibility = 'visible';
        document.getElementById('status').innerHTML = 'Camera initialized. Click Start Detection.';
    } catch (error) {
        document.getElementById('status').innerHTML = 'Error accessing camera: ' + error.message;
        console.error('Camera error:', error);
    }
}

async function startDetection() {
    if (isProcessing) return;
    
    document.getElementById('status').innerHTML = 'Loading hand detection model...';
    
    try {
        // Clear any prior model
        model = null;
        
        // Make sure required libraries are loaded
        if (!window.handPoseDetection) {
            document.getElementById('status').innerHTML = 'Error: Hand pose detection library not loaded';
            return;
        }
        
        // Use MediaPipe Hands model with proper configuration
        model = await handPoseDetection.createDetector(
            handPoseDetection.SupportedModels.MediaPipeHands,
            {
                runtime: 'mediapipe',
                solutionPath: 'https://cdn.jsdelivr.net/npm/@mediapipe/hands',
                modelType: 'lite'
            }
        );
        
        document.getElementById('status').innerHTML = 'Model loaded. Processing...';
        isProcessing = true;
        
        // Start detection loop
        detectHands();
    } catch (error) {
        document.getElementById('status').innerHTML = 'Error loading model: ' + error.message;
        console.error('Model loading error:', error);
    }
}

async function detectHands() {
    if (!isProcessing || !model) return;
    
    try {
        // Draw video frame to canvas
        ctx.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);
        
        // Detect hands
        const hands = await model.estimateHands(videoElement);
        
        if (hands && hands.length > 0) {
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
    requestAnimationFrame(detectHands);
}

function drawLandmarks(keypoints) {
    if (!keypoints || keypoints.length === 0) return;
    
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
    
    // Thumb is extended if its tip is to the side of the MCP joint
    if (thumbTip && thumbMcp && thumbTip.x < thumbMcp.x) {
        extendedFingers++;
    }
    
    // Check other 4 fingers
    for (let i = 1; i < 5; i++) {
        const tip = keypoints[fingertips[i]];
        const pip = keypoints[fingertips[i] - 2]; // Proximal Interphalangeal joint
        
        // A finger is extended if its tip is higher than its PIP joint
        if (tip && pip && tip.y < pip.y) {
            extendedFingers++;
        }
    }
    
    return extendedFingers;
}