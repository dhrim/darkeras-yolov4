# Darkeras Wrapper

LA Tran님의 https://github.com/tranleanh/darkeras-yolov4 을 가져와서
사용하기 쉽게 가공한 것.

다음의 코드로 사용 가능
```
import yolov3_wrapper
model = yolov3_wrapper.YoloV3Wrapper("yolov3.weights") # darknet 포멧의 모델 파일 로딩
image = ... # load image into numpy arrary
bboxes = model.predict(image)
```

다음의 코드로 keras포멧으로 저장 가능
```
model.save("keras_model.h5")
```

<br>


# Darkeras: Execute YOLOv3/YOLOv4 Object Detection on Keras with Darknet Pre-trained Weights

This project is to execute YOLOv3/YOLOv4 object detector on Keras framework with Darknet pre-trained weights.

Medium article: [Darkeras: Execute YOLOv3/YOLOv4 Object Detection on Keras with Darknet Pre-trained Weights](https://towardsdatascience.com/darkeras-execute-yolov3-yolov4-object-detection-on-keras-with-darknet-pre-trained-weights-5e8428b959e2)

### Dependencies
- OpenCV
- Tensorflow 2.3.0
- Keras 2.4.0
- Easydict
- Matplotlib

You can simply run:
```bashrc
$ pip install -r requirements.txt
```

### Supported Models
- YOLOv3
- YOLOv4

### How to Use

#### 1. Run with Notebook
- Put pre-trained weights from official Darknet [website](https://pjreddie.com/darknet/yolo/) or your trained weights into "weights" folder (If you use your model trained on your customed dataset, please change NUM_CLASS and ANCHORS in the notebooks)
- Run YOLOv3: darkeras-yolov3.ipynb
- Run YOLOv4: darkeras-yolov4.ipynb

#### 2. Run with Python Scripts
- Convert Darknet (format: weight_file.weights) to Keras (format: weight_file.h5): to be updated
- Run YOLOv3: to be updated
```bashrc
$ python detect_yolov3.py
```
- Run YOLOv4: to be updated
```bashrc
$ python detect_yolov4.py
```

### Detection Result
- Input
<img src="docs/kite.jpg" width="800">

- YOLOv3
<img src="docs/detections_yolov3.jpg" width="800">

- YOLOv4
<img src="docs/detections.jpg" width="800">




Have fun!

LA Tran

Korea, August 2021
