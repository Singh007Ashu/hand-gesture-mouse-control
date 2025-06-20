import cv2
import mediapipe as mp
import math

class HandTracking:
    def __init__(self, min_detection_confidence=0.7):
        print("Initializing HandTracking")
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.results = None
        self.landmarks = {}

    def find_hand(self, image):
        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.results = self.hands.process(image_rgb)
            self.landmarks = {}
            if self.results.multi_hand_landmarks:
                for hand_id, hand_lms in enumerate(self.results.multi_hand_landmarks):
                    self.landmarks[str(hand_id)] = hand_lms.landmark
        except Exception as e:
            print(f"Error in find_hand: {e}")

    def find_finger_tips(
        self, image, finger_list=None, show_connected=True,
        show_landmarks=True, draw_tips=False, hand_id_list=None
    ):
        try:
            if self.results.multi_hand_landmarks:
                for hand_id, hand_lms in enumerate(self.results.multi_hand_landmarks):
                    if hand_id_list is None or str(hand_id) in [str(i) for i in hand_id_list]:
                        if show_connected:
                            self.mp_draw.draw_landmarks(image, hand_lms, self.mp_hands.HAND_CONNECTIONS)
                        if show_landmarks or draw_tips:
                            h, w, c = image.shape
                            for idx, lm in enumerate(hand_lms.landmark):
                                cx, cy = int(lm.x * w), int(lm.y * h)
                                if draw_tips and idx in [4, 8, 12, 16, 20]:
                                    cv2.circle(image, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                                if show_landmarks:
                                    cv2.circle(image, (cx, cy), 3, (0, 255, 0), cv2.FILLED)
        except Exception as e:
            print(f"Error in find_finger_tips: {e}")

    def is_finger_up(self, image, hand_id_list=None):
        finger_up_dict = {}
        try:
            if self.results.multi_hand_landmarks:
                for hand_id, hand_lms in enumerate(self.results.multi_hand_landmarks):
                    if hand_id_list is None or str(hand_id) in [str(i) for i in hand_id_list]:
                        finger_up = [0] * 5  # [Thumb, Index, Middle, Ring, Pinky]
                        landmarks = hand_lms.landmark
                        if landmarks[4].x < landmarks[3].x:
                            finger_up[0] = 1
                        for i in range(1, 5):
                            tip_id = 4 + i * 4  # 8, 12, 16, 20
                            pip_id = tip_id - 2  # 6, 10, 14, 18
                            if landmarks[tip_id].y < landmarks[pip_id].y:
                                finger_up[i] = 1
                        finger_up_dict[str(hand_id)] = {
                            '0': finger_up,
                            'lms': {i: [int(lm.x * image.shape[1]), int(lm.y * image.shape[0]), lm.z]
                                    for i, lm in enumerate(landmarks)}
                        }
        except Exception as e:
            print(f"Error in is_finger_up: {e}")
        return finger_up_dict

    @staticmethod
    def calculate_distance(x1, y1, x2, y2):
        try:
            distance = math.hypot(x2 - x1, y2 - y1)
            print(f"Calculated distance: {distance}")
            return distance
        except Exception as e:
            print(f"Error in calculate_distance: {e}")
            return float('inf')