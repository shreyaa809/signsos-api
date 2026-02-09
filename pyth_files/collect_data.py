import cv2
import mediapipe as mp
import csv

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
cap = cv2.VideoCapture(0)

label = "TWO_PEOPLE_TRAPPED"   # 🔴 CHANGE THIS LATER

with open("gesture_data.csv", "a", newline="") as f:
    writer = csv.writer(f)

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                row = []
                for lm in hand.landmark:
                    row.extend([lm.x, lm.y])
                row.append(label)
                writer.writerow(row)

        cv2.putText(frame, f"Recording: {label}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow("Data Collection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
