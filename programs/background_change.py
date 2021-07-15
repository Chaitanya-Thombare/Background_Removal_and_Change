import numpy as np
import cv2
import time

confThreshold = 0.5
nmsThreshold = 0.2
inpWidth = 416
inpHeight = 416

def get_coods(img, outs):
    frameHeight, frameWidth = img.shape[0], img.shape[1]

    classIds, confidences, boxes = [], [], []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    return[boxes[i[0]] for i in indices]

start = time.time()

cfg_path = "../yolo/yolov3-tiny.cfg"
weights_path = "../yolo/yolov3-tiny.weights"

net = cv2.dnn.readNetFromDarknet(cfg_path , weights_path)

classes = []
with open("../yolo/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()

output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

frame = cv2.imread('../media/view3.jpeg')
background = cv2.imread('../media/s.jpg')

img = cv2.resize(frame, (504, 378))

bgdModel = np.zeros((1,65), np.float64)
fgdModel = np.zeros((1,65), np.float64)

blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

rects = get_coods(img, outs)

for rect in rects:
    mask = np.zeros(img.shape[:2],np.uint8)
    
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img*mask2[:,:,np.newaxis]

    img = cv2.resize(img, (4032, 3024))

    non_black_pixels_mask = np.all(img != [0, 0, 0], axis=-1)
    background[non_black_pixels_mask] = frame[non_black_pixels_mask]

    cv2.imwrite("output3c.jpg", background)
    end = time.time()

print(end-start)