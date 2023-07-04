import time
import numpy as np
from ultralytics import YOLO
import cv2
import tensorflow as tf

# sử đường dẫn để load 2 model
modelYOLO = YOLO('models\\best.pt')
model = tf.keras.models.load_model(f'models/35frames_top_down.h5', compile=False)


def get_feature(mask, s):
    h, w = mask.shape

    # Compute ratio
    features = [0, 0, 0, 0, 0]

    coor = (w // 2, h // 2)
    tan55 = 1.428  # tan(55)

    for i in range(h):
        i = i - coor[1]
        for j in range(w):
            j = j - coor[0]
            if (j <= 0) and (j - i > 0):
                features[0] += mask[i, j]
            elif (j > 0) and (j + i <= 0):
                features[1] += mask[i + coor[1], j + coor[0]]
            elif (j - i <= 0) and (i + tan55 * j <= 0):
                features[2] += mask[i + coor[1], j + coor[0]]
            elif (j + i > 0) and (i - tan55 * j > 0):
                features[3] += mask[i + coor[1], j + coor[0]]
            else:
                features[4] += mask[i + coor[1], j + coor[0]]
    return [f / s for f in features]


def get_bounding_box(frame):
    # Run YOLOv8 inference on the frame
    results = modelYOLO(frame, verbose=False)
    # If not exist person
    if results[0].masks is None:
        return None

    # get box object
    box = results[0].boxes[0].xyxy[0]
    box = box.numpy().astype(int)
    x1, y1, x2, y2 = box

    # s = abs(x1 - x2) * abs(y1 - y2)
    # background subtraction
    mask = (results[0].masks.data[0].numpy() * 255).astype('uint8')
    # resize size mask equal size original frame
    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
    # get bounding box person
    mask = mask[y1:y2, x1:x2]
    return mask

# link video dùng để test
video_path = 'D:\pythonProject\\video\\test_dat_1.mp4'

cap = cv2.VideoCapture(video_path)
X = []
batchSize = 1
numberFall = 0

text = "NORMAL"
color = (0, 255, 0)

# kết quả dự đoán lưu vào video
result = cv2.VideoWriter('result.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (720, 480))

while cap.isOpened():
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break
    mask = get_bounding_box(frame)
    if mask is None:
        cv2.putText(frame, text, (250, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_4)
        cv2.imshow('Result', frame)
        result.write(cv2.resize(frame, (720, 480)))

        if cv2.waitKey(1) == 27:  # ESC key
            break
        continue

    # Tinh dien tich khung hinh
    h, w, c = frame.shape
    s = h * w
    if batchSize <= 5:
        features = get_feature(mask, s)
        X.append(features)
        batchSize = batchSize + 1
    else:
        batchSize = 1
        if len(X) > 35:
            X = X[5:]
            x = np.expand_dims(np.array(X), axis=0)
            p = model.predict(x)[0]

            if p > 0.5:
                numberFall = numberFall + 1
                if numberFall == 3:
                    text = 'FALL'
                    color = (0, 0, 255)
            else:
                numberFall = 0
                text = "NORMAL"
                color = (0, 255, 0)

    time_process = time.time() - start_time
    fps = str(int(1.0 / time_process))
    cv2.putText(frame, text, (250, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_4)
    cv2.putText(frame, f"FPS: {fps}", (450, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_4)

    cv2.imshow('Result', frame)
    result.write(cv2.resize(frame, (720, 480)))

    if cv2.waitKey(1) == 27:  # ESC key
        break

cap.release()
result.release()
cv2.destroyAllWindows()
