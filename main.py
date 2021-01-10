import cv2
import numpy as np

#Init files:
namesFile = "coco.names"
configurationFile = "yolov4.cfg"
#A weight fajl valoszinuleg hianyzik a git repobol, mert az internet feltoltesi sebessegem lassu
#link a sulyhoz: https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
weightFile = "yolov4.weights"
imageFile = "car2.jpg"

#Init variables:
confidenceThreshold = 0.5
nMaximumSuppresssionThreshold = 0.4
inputWidth = 608
inputheight = 608

#Load YOLO with files:
net = cv2.dnn.readNetFromDarknet(configurationFile,weightFile)
classes = []
with open(namesFile, "r") as f:
    classes = [line.strip() for line in f.readlines()]

#Load image:
my_img = cv2.imread(imageFile)
my_img = cv2.resize(my_img,(1280,720))
bh, bw , channels = my_img.shape

#Converting into Blob:
blob = cv2.dnn.blobFromImage(my_img,1/255,(inputWidth,inputheight),(0,0,0),swapRB= True,crop=False)
net.setInput(blob)
last_layer = net.getUnconnectedOutLayersNames()
layer_out = net.forward(last_layer)

#Get data from layers_out
boxes = []
confidences = []
class_ids = []
for output in layer_out:
    for detection in output:
        score = detection[5:]
        class_id = np.argmax(score)
        confidence = score[class_id]
        if confidence > confidenceThreshold:
            center_x = int(detection[0] * bw)
            center_y = int(detection[1] * bh)
            w = int(detection[2]*bw)
            h = int(detection[3]*bh)
            x = int(center_x - w/2)
            y = int(center_y - h/2)
            boxes.append([x,y,w,h])
            confidences.append((float(confidence)))
            class_ids.append((class_id))

#Non-maximum suppression:
indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidenceThreshold, nMaximumSuppresssionThreshold)

#visualization:
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0,255,size = (len(boxes),3))
for i in indexes.flatten():
    x,y,w,h = boxes[i]
    label = str(classes[class_ids[i]])
    confidence = str(round(confidences[i],2))
    color=colors[i]
    cv2.rectangle(my_img,(x,y),(x+w,y+h),color,2)
    cv2.putText(my_img,label + " " + confidence, (x,y+20),font,1,(255,255,0),2)

cv2.imshow('img',my_img)
cv2.waitKey(0)
cv2.destroyAllWindows()