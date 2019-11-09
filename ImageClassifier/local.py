#!/usr/bin/env python3
import time
import datetime
import numpy as np
import cv2
import boto3
#from jetbot import Camera
from dlr import DLRModel
#import greengrasssdk
current_milli_time = lambda: int(round(time.time() * 1000))
#mqtt_client = greengrasssdk.client('iot-data')
model_resource_path =  ('/home/ggc_user/artifact1/model')
#model_resource_path = '/home/jetbot/projects/dino/dino_model'
dlr_model = DLRModel(model_resource_path, 'gpu')

#cloudwatch = boto3.client('cloudwatch')
prev_class = -1

classes_list = ['person','bicycle','car','motorbike','aeroplane','bus','train','truck', 'boat', 'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog']


def predict(re, change):
    #send_mqtt_message("enter predict class")
    img = change#['new']
    if re is True:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.asarray(img)
        img = np.rollaxis(img, axis=2, start=0)[np.newaxis,:]
        flattened_data = img.astype(np.float32)
        t1 = current_milli_time()
        prediction_scores = dlr_model.run({'data' : flattened_data})
        t2 = current_milli_time()
        print('done m.run(), time (ms): {}'.format(t2 - t1))
        print("priediction_scores:", prediction_scores)
        max_score_id = np.argmax(prediction_scores)
        max_score = np.max(prediction_scores)
        print(max_score_id)
    #send_mqtt_message("get the max score: {0}".format(str(max_score_id)))
        return max_score, max_score_id
    else:
        pass

def predict1(re, change):
    #send_mqtt_message("enter predict class")
    img = change#['new']
    if re is True:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
       pass
    img = np.asarray(img)
    img = np.rollaxis(img, axis=2, start=0)[np.newaxis,:]
    flattened_data = img.astype(np.float32)

    prediction_scores = dlr_model.run({'data' : flattened_data})
    max_score_id = np.argmax(prediction_scores)
    max_score = np.max(prediction_scores)
    print(max_score_id)
    #send_mqtt_message("get the max score: {0}".format(str(max_score_id)))
    return max_score, max_score_id


def send_mqtt_message(message):
    mqtt_client.publish(topic='hello/nvidianano',
                        payload=message)

# replace it with any local image to avoid the warm up period
def warmup_model():
    img = cv2.imread("/home/sarita/t1.jpeg")
    img = cv2.resize(img, (224, 224,))
    re = True
    probs, classes = predict(re,img)
    print("Done warming up resource ...")


warmup_model()

gst_str = 'nvarguscamerasrc ! video/x-raw(memory:NVMM), width=%d, height=%d, format=(string)NV12, framerate=(fraction)%d/1 ! nvvidconv ! video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! videoconvert ! appsink' % (1920, 1080, 1, 600, 400)
cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

print("Started capturing ....")

iteration_total_time = time.time()

for i in range(10):
    #img = cv2.imread("/home/sarita/t1.jpeg")
    #re = True
    re, img = cap.read()
    #img = cv2.resize(img, (224, 224,))
    #img = img[200:350, 210:360]
    img = cv2.resize(img, (224, 224,))
    #print(img.shape)
    if (i == 1):
        cv2.imwrite("/home/sarita/cap_t1.jpeg", img)
    #print(re, img)
    probs, classes = predict(re,img) 
    msg = "Start..."
    print("starts the process....!")
    s3url = ""
    if probs > 0.5 and prev_class != classes:
        prev_class = classes
        msg = '{"Object":"' + classes_list[classes] + '"' + ',"confidence":"' + str(probs) +'"}'
        print(msg)
        #send_mqtt_message(msg)
        #push_to_cloudwatch(dino_names[classes], round(probs.item(), 2))
        #break
    # cv2.imshow('image', img)
    # cv2.waitKey(0)

print("Total time taken: ", time.time() - iteration_total_time)

# The lambda to be invoked in Greengrass
def handler(event, context):
    pass
