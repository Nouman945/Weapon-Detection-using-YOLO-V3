import cv2 as cv
import numpy as np
import winsound
import os

#Write down conf, nms thresholds,inp width/height
confThreshold = 0.25
nmsThreshold = 0.20
inpWidth = 416
inpHeight = 416


#Settings for Beep Sound
frequency = 2500  # Set Frequency To 2500 Hertz
duration = 1000  # Set Duration To 1000 ms == 1 second

#Load names of classes and turn that into a list
classesFile = "model/coco.names"
classes = None

with open(classesFile,'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

#Model configuration
modelConf = 'model/yolov3.cfg'
modelWeights = 'model/yolov3.weights'

def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIDs = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            
            scores = detection [5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > confThreshold:
                centerX = int(detection[0] * frameWidth)
                centerY = int(detection[1] * frameHeight)

                width = int(detection[2]* frameWidth)
                height = int(detection[3]*frameHeight )

                left = int(centerX - width/2)
                top = int(centerY - height/2)

                classIDs.append(classID)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    indices = cv.dnn.NMSBoxes (boxes,confidences, confThreshold, nmsThreshold )

    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        
        drawPred(classIDs[i], confidences[i], left, top, left + width, top + height)


def drawPred(classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)

    label = '%.2f' % conf

    # Get the label for the class name and its confidence
    if classes:
        assert (classId < len(classes))
        label = '%s:%s' % (classes[classId], label)
        name , label1 = (classes[classId], label)
        print(name)
        if name == 'cell phone' :
            winsound.Beep(frequency, duration)

    cv.putText(frame, label, (left,top), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)


 

def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
   
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


#Set up the net

net = cv.dnn.readNetFromDarknet(modelConf, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)


#Process inputs
winName = 'DL OD with OpenCV'
cv.namedWindow(winName, cv.WINDOW_NORMAL)
cv.resizeWindow(winName, 1000,2000 )





cap = cv.VideoCapture(0)
# cap = cv.VideoCapture('test.mp4')

fourcc = cv.VideoWriter_fourcc(*'XVID') 
out = cv.VideoWriter('output/output.avi', fourcc, 20.0, (640, 480))

while cv.waitKey(1) < 0:

    #get frame from video
    hasFrame, frame = cap.read()

    #Create a 4D blob from a frame
    
    blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop = False)

    #Set the input the the net
    net.setInput(blob)
    outs = net.forward (getOutputsNames(net))

    out.write(frame)  

    postprocess (frame, outs)

    #show the image
    cv.imshow(winName, frame)

