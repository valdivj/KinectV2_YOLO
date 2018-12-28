import cv2
from darkflow.net.build import TFNet
from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
import numpy as np
import time
import tensorflow as tf


config = tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    options = {
            'model': 'cfg/yolov2-tiny-voc.cfg',
            'load': 'bin/yolov2-tiny-voc.weights',
            'threshold': 0.2,
            'gpu': 0.3
                    }
    tfnet = TFNet(options)

colors = [tuple(255 * np.random.rand(3)) for _ in range(10)]
kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)
kinectD = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth)


while True:
    stime = time.time()
    if kinect.has_new_color_frame():
        frame = kinect.get_last_color_frame()
        frame = np.reshape(frame, (1080, 1920, 4))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        results = tfnet.return_predict(frame)
        for color, result in zip(colors, results):
            tl = (result['topleft']['x'], result['topleft']['y'])
            #cnt = (result['x'], result['y'])
            br = (result['bottomright']['x'], result['bottomright']['y'])
            label = result['label']
            confidence = result['confidence']
            text = '{}: {:.0f}%'.format(label, confidence * 100)
            frame = cv2.rectangle(frame, tl, br, color, 5)
            frame = cv2.putText(
                frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
        cv2.imshow('frame', frame)
        frame = None
        #print('FPS {:.1f}'.format(1 / (time.time() - stime)))
        print(tl)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#frame.release()
cv2.destroyAllWindows()
