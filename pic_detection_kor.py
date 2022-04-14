from cProfile import label
import cv2
from matplotlib import image
from matplotlib.pyplot import draw
# print(cv2.__version__)#4.5.5
import numpy as np

# Load Yolo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
#클래스 이름을 따로 저장해준다. 이 형식은 클래스가 한글이름 일 때 불러오는 방식이다.
with open("coco.names", "r", encoding='UTF8') as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames() # 네트워크의 모든 레이어 이름을 가져옵니다.
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()] 
# 네트워크의 출력 레이어 이름을 가져옵니다.
'''
It's may caused by the different versions of cv2. 
The version of cv2 module with CUDA support will give you a 2-D array 
when calling network.getUnconnectedOutLayers(). 
However, the version without CUDA support will give a 1-D array.
'''
colors = np.random.uniform(0, 255, size=(len(classes), 4))

# 이미지 가져오기
img = cv2.imread("wife.jpg")
img = cv2.resize(img, None, fx=0.4, fy=0.4)
height, width, channels = img.shape




# Detecting objects
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
#네트워크에 넣어줄 이미지를 blob로 만들어준다.
net.setInput(blob) # 전처리된 이미지를 네트워크에 넣어준다.
outs = net.forward(output_layers) # 결과값을 가져온다.

# 정보를 화면에 표시
class_ids = []
confidences = []
boxes = []

for out in outs: # 출력을 각각 확인합니다.
    for detection in out: # detection = out[i] = [x, y, w, h, obj_score, class_id]
        scores = detection[5:] # [5:] 는 가장 앞의 5개를 버리고 나머지를 가져옵니다.
        class_id = np.argmax(scores) # 가장 높은 점수를 가진 클래스 아이디를 가져옵니다.
        confidence = scores[class_id]
        if confidence > 0.5: # 확률이 0.5 이상인 것만 가져옵니다.
            # Object detected
            # 탐지된 객체의 너비, 높이 및 중앙 좌표값 찾기
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            #print(center_x,center_y)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            #print(w,h)
            # 객체의 사각형 테두리 중 좌상단 좌표값 찾기
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)
            
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
# Non Maximum Suppression (겹쳐있는 박스 중 confidence 가 가장 높은 박스를 선택)
# 같은 index 중 확률이 가장 높은것

from PIL import ImageFont, ImageDraw, Image

font = cv2.FONT_HERSHEY_PLAIN

labels = []
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i] # 객체의 사각형 테두리 중 좌상단 좌표값 찾기
        label = str(classes[class_ids[i]]) # 클래스 이름을 가져옵니다.
        color = colors[i] # 색상을 가져옵니다.
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2) # 사각형 테두리 그리기
        img = np.array(img) # 이미지를 numpy array로 변환
        # draw.text(img,  label, (x, y + 30), font, 3, color, 3)
        # cv2.putText(img, label, (x, y + 30), font, 3, color, 3) # 텍스트 그리기
        labels.append(label) # 클래스 이름을 리스트에 추가
        
        
        b,g,r,a = int(color[0]), int(color[1]), int(color[2]), int(color[3])
        fontpath = "./malgun.ttf" 
        font = ImageFont.truetype(fontpath, 32)
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)

        draw.text((x , y-50),  label, font = font, fill = (b, g, r, a))
        img = np.array(img_pil)

print(labels) 

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
