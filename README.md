# 💻Tech Stack
![Python](https://img.shields.io/badge/python-3670A0?style=plastic&logo=python&logoColor=ffdd54) ![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=plastic&logo=Keras&logoColor=white) ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=plastic&logo=numpy&logoColor=white) ![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=plastic&logo=TensorFlow&logoColor=white)

# 📊 Thư viện và các bước thực hiện Phương pháp Top-down:

* Chương trình chạy trên laptop do thư viện ultralytics (yêu cầu python >= 3.8 nguồn: Link: [Tham khảo:](https://docs.ultralytics.com/tasks/segment/)) trong khi Jetson Nano Developer Kit B01 hỗ trợ python 3.6.

* Các thư viện nhóm đã cài đặt để chạy PP Top-down trên python 3.9 (chỉ mang tính chất tham khảo):
    -	numpy: 1.24.3
    -	cv2: 4.7.0
    -	ultralytics: 8.0.117 (để import sử dụng YOLOv8 segment: Document sử dụng yolov8)
    -	tensorflow: 2.8.0

* Môi trường huấn luyện model: sử dụng GPU Google colab.

* Các bước thực hiện:
    -	Thu thập dataset fall detection: 
        + https://data.mendeley.com/datasets/7w7fccy7ky/4
        + https://imvia.u-bourgogne.fr/en/database/fall-detection-dataset-2.html
        + https://github.com/YifeiYang210/Fall_Detection_dataset
        + https://rose1.ntu.edu.sg
    -	Custom train yolov8 segment: dataset tự tạo trên roboflow vì nhóm sử dụng model mặc định nhiều trường hợp không segment được. Link tham khảo: [Train lại model yolov8 segment](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/train-yolov8-instance-segmentation-on-custom-dataset.ipynb)
    -	Thực hiện training model để predict fall
 
* Chạy chương trình: chạy file demo_top-down.py (sửa đường dẫn video, path model để chạy)

# 📊 Thư viện và các bước thực hiện Phương pháp Bottom-up:

* Cài thư viện phụ thuộc: https://github.com/nt-myduyen/fall-dection-system-on-jetsonnano/blob/main/Config_JetsonNano.md

* Các bước thực hiện tương tự phương pháp Top-down

* Chạy chương trình: chạy file demo_bottom-up.py

# 📊 Link demo: https://www.youtube.com/watch?v=QQgUqKkMC-E

