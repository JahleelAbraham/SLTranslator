import cv2
import mediapipe as mp

mphands = mp.solutions.hands
hands = mphands.Hands()
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

_, frame = cap.read()

h, w, c = frame.shape


def render(triggerModel):
    _, frame = cap.read()

    cropped = frame.copy()

    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)
    hand_landmarks = result.multi_hand_landmarks
    rect = None

    x_max = 0
    y_max = 0
    x_min = w
    y_min = h
    if hand_landmarks:
        for handLMs in hand_landmarks:

            for lm in handLMs.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y
            rect = cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            mp_drawing.draw_landmarks(frame, handLMs, mphands.HAND_CONNECTIONS)
    else:
        rect = None

    frame = cv2.flip(frame, 1)

    padding = 35

    if x_min - padding <= 0 or y_min - padding <= 0 or x_max + padding >= w or y_max + padding >= h:
        cv2.putText(frame, "Please keep your hands in frame!", (25, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 0.9,
                    (0, 255, 0), 2)
    else:
        if rect is not None:
            print(f"({x_min - padding}, {y_min - padding}) to ({x_max + padding}, {y_max + padding})")
            cropped = cropped[y_min - padding:y_max + padding, x_min - padding:x_max + padding]
            cv2.imshow("Cropped",
                       cv2.flip(cropped, 1))

            gray = cv2.cvtColor(cv2.flip(cropped, 1), cv2.COLOR_BGR2GRAY)
            resize = cv2.resize(gray, (28, 28))
            cv2.putText(frame, triggerModel(resize), (25, 100), cv2.FONT_HERSHEY_COMPLEX, 0.9, (255, 255, 255), 2)
            cv2.imshow("Smol", resize)

    cv2.rectangle(frame, (0 + padding, 0 + padding), (w - padding, h - padding), (255, 255, 255), 2)
    cv2.imshow("Frame", frame)

    cv2.waitKey(1)

    # return resize
