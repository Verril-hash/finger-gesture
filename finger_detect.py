import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Draw hand landmarks
mp_draw = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)

# Finger tip landmark indices
finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
thumb_tip = 4

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

    cv2.imshow("Finger Gesture Detection", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
