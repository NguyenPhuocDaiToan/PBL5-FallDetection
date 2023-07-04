import cv2
import mediapipe as mp
import numpy as np
import threading
import tensorflow as tf
import time
import datetime
from common import upload_video_to_firebase

# Initialize pose estimate
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

# Load model lstm for FALL detect
model = tf.keras.models.load_model("/home/yuu/Documents/PBL5/modelLSTM/model_lstm_30frames_v1.h5", compile=False)


label = "NORMAL"
number_frames_of_observation = 30
lm_list = []

frames = []
state_previous = 'Normal'
video_filename = "Fall_video.mp4"
video_writer = None

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def make_landmark_timestep(results):
    c_lm = []
    for id, lm in enumerate(results.pose_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm

def draw_landmark_on_image(mpDraw, results, img):
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    for id, lm in enumerate(results.pose_landmarks.landmark):
        h, w, c = img.shape
        # print(id, lm)
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
    return img

def draw_class_on_image(label, fps, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 30)
    fpsLocation = (220, 30)
    timeLocation = (img.shape[1] - 190, 30)  # Right corner
    fontScale = 1
    fontScaleSmall = 0.5
    thickness = 2
    lineType = 2

    if label == 'NORMAL':
        color = (0, 255, 0)
    else:
        color = (0, 0, 255)


    cv2.putText(img, label, bottomLeftCornerOfText, font, fontScale, color, thickness, lineType)

    rounded_fps = round(float(fps))
    fps_text = f"fps: {rounded_fps}"
    cv2.putText(img, fps_text, fpsLocation, font, fontScale, color, thickness, lineType)

    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(img, current_time, timeLocation, font, fontScaleSmall, color, thickness, lineType)

    return img

def detect(lm_list):
    global label

    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)

    # predict fall
    results = model.predict(lm_list)

    detect.counter = getattr(detect, 'counter', 0)
    if results[0][0] > 0.5:
        detect.counter += 1
        if detect.counter >= 2:
            label = "FALL"
            detect.counter = 0
    else:
        detect.counter = 0
        label = "NORMAL"
    print("Label: ", label)
    return label

video_writer = cv2.VideoWriter("video_demo.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 17, (640, 480))

count = 0

while True:
    start_time = time.time()
    success, img = cap.read()
    if not success:
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)

    if results.pose_landmarks:
        # draw human pose
        img = draw_landmark_on_image(mpDraw, results, img)


        c_lm = make_landmark_timestep(results)

        lm_list.append(c_lm)
        if len(lm_list) == number_frames_of_observation:
            # predict fall
            label = detect(lm_list)
            if label == 'FALL':
                if state_previous == 'NORMAL':
                    count = count + 1
                    i = 0
                    while i < 20:
                        success, img = cap.read()
                        if not success:
                            break 
                        img = draw_class_on_image(label, fps, img)
                        frames.append(img)
                        cv2.imshow("Fall detection", img)
                        if cv2.waitKey(1) == ord('q'):
                            break
                        i = i + 1
                        
                    # write video and send to firebase
                    thread_send_video = threading.Thread(target=upload_video_to_firebase, args=(frames, count, ))
                    thread_send_video.start()
                
                frames = []
                lm_list = []
                state_previous = 'FALL'
            else:
                state_previous = 'NORMAL'
            lm_list = lm_list[10:]
            frames = frames[10:]

        fps = str(1.0 / (time.time() - start_time))
        img = draw_class_on_image(label, fps, img)
        frames.append(img)

    cv2.imshow("Fall detection", img)
    video_writer.write(cv2.resize(img, (640, 480)))
    if cv2.waitKey(1) == ord('q'):
        break

    # # Write the annotated image to the video file
    # if video_writer is not None:
    #     video_writer.write(cv2.resize(img, (640, 480)))

# Release the video writer and upload the last video if applicable
if video_writer is not None:
    video_writer.release()

cap.release()
cv2.destroyAllWindows()