import cv2
import time
import pyautogui
import numpy as np
import hand_tracking as ht
import math

def main(show_fps=True, video_src=0):
    print("Starting main function")
    cap = cv2.VideoCapture(video_src)
    if not cap.isOpened():
        print(f"Error: Could not open webcam with source {video_src}")
        return

    previous_time = 0
    try:
        track = ht.HandTracking(min_detection_confidence=0.7)
    except Exception as e:
        print(f"Error initializing HandTracking: {e}")
        cap.release()
        return

    screen_width, screen_height = pyautogui.size()
    print(f"Screen resolution: {screen_width}x{screen_height}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    w_box, h_box = 800, 600  # Control box for better up-down range
    width_offset = 50
    height_offset = 30
    pyautogui.FAILSAFE = False
    smooth_fact = 3  # Fast cursor response
    pre_loc_x, pre_loc_y = 0, 0
    click_cooldown = 0  # Prevent rapid clicks

    while True:
        key = cv2.waitKey(1)
        if key in [ord('q'), ord('Q')]:
            print("Exiting on 'q' keypress")
            break

        success, img = cap.read()
        if not success:
            print("Warning: Failed to read frame from webcam")
            continue

        flip_image = cv2.flip(img, 1)
        h, w, c = flip_image.shape

        track.find_hand(flip_image)
        track.find_finger_tips(
            flip_image,
            show_connected=True,
            show_landmarks=True,
            draw_tips=True,
            hand_id_list=[0]
        )

        mode = ""  # Default to empty string, no "Idle" text
        finger_up_dict = track.is_finger_up(flip_image, hand_id_list=[0])
        finger_up = finger_up_dict.get('0', {}).get('0', [])
        pt1_x, pt1_y = (w - width_offset, height_offset)
        pt2_x, pt2_y = (w - (w_box + width_offset), h_box + height_offset)

        if len(finger_up):
            landmarks = finger_up_dict.get('0', {}).get('lms', {})
            print(f"Fingers up: {finger_up}")  # Debug: Show which fingers are detected

            # Move Mode (Index only)
            if finger_up[1] and sum(finger_up) == 1:
                finger_pos = landmarks.get(8, [0, 0])[:2]
                cv2.rectangle(flip_image, (pt1_x, pt1_y), (pt2_x, pt2_y), (255, 0, 255), 2)

                try:
                    abs_x = np.interp(finger_pos[0] - pt2_x, [0, w_box], [0, screen_width])
                    abs_y = np.interp(finger_pos[1] - pt1_y, [0, h_box], [0, screen_height])

                    cur_loc_x = pre_loc_x + (abs_x - pre_loc_x) / smooth_fact
                    cur_loc_y = pre_loc_y + (abs_y - pre_loc_y) / smooth_fact

                    pyautogui.moveTo(cur_loc_x, cur_loc_y)
                    pre_loc_x, pre_loc_y = cur_loc_x, cur_loc_y
                    mode = "Move"
                    print(f"Moving cursor to: ({cur_loc_x}, {cur_loc_y})")
                except Exception as e:
                    print(f"Error in mouse movement: {e}")

            # Click Mode (Index + Middle)
            elif finger_up[1] and finger_up[2] and sum(finger_up) == 2:
                index_pos = landmarks.get(8, [0, 0])[:2]
                middle_pos = landmarks.get(12, [0, 0])[:2]
                print(f"Index pos: {index_pos}, Middle pos: {middle_pos}")  # Debug
                try:
                    distance = ht.calculate_distance(index_pos[0], index_pos[1], middle_pos[0], middle_pos[1])
                except AttributeError:
                    print("Warning: calculate_distance not found, using fallback")
                    distance = math.hypot(middle_pos[0] - index_pos[0], middle_pos[1] - index_pos[1])
                    print(f"Fallback calculated distance: {distance}")
                if distance < 50 and click_cooldown <= 0:
                    try:
                        pyautogui.click()
                        mode = "Click"
                        click_cooldown = 10  # Cooldown for 10 frames (~0.3s)
                        print("Click triggered")
                    except Exception as e:
                        print(f"Error in click: {e}")
                cv2.rectangle(flip_image, (pt1_x, pt1_y), (pt2_x, pt2_y), (0, 255, 0), 2)

        if click_cooldown > 0:
            click_cooldown -= 1

        # Only display mode text if it's "Move" or "Click"
        if mode in ["Move", "Click"]:
            cv2.putText(flip_image, mode, (pt2_x + 10, pt2_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        if show_fps:
            current_time = time.time()
            fps = 1 / (current_time - previous_time) if previous_time != 0 else 0
            previous_time = current_time
            cv2.putText(flip_image, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

        try:
            cv2.imshow("Output", flip_image)
        except Exception as e:
            print(f"Error displaying frame: {e}")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Script terminated")

if __name__ == "__main__":
    main(show_fps=True)