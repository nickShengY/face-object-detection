# from training import *
import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model

cap = cv2.VideoCapture(0)


def get_frame(cap):
    _, frame = cap.read()
    return frame[50:500, 50:500, :]

def process_frame(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = tf.image.resize(rgb, (120, 120))
    return resized/255

def draw_rectangle(frame, sample_coords, color, thickness):
    cv2.rectangle(frame, 
                  tuple(np.multiply(sample_coords[:2], [450,450]).astype(int)),
                  tuple(np.multiply(sample_coords[2:], [450,450]).astype(int)), 
                  color, thickness)

def draw_label(frame, sample_coords):
    cv2.rectangle(frame, 
                  tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int), [0,-30])),
                  tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int), [80,0])), 
                  (255,0,0), -1)
    cv2.putText(frame, 'face', 
                tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int), [0,-5])),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

def live_detection():
    facetracker = load_model('facetracker.h5')
    while cap.isOpened():
        frame = get_frame(cap)
        preprocessed_frame = process_frame(frame)
        
        yhat = facetracker.predict(np.expand_dims(preprocessed_frame,0))
        sample_coords = yhat[1][0]
        
        if yhat[0] > 0.5: 
            draw_rectangle(frame, sample_coords, (0,255,0), 2)
            draw_label(frame, sample_coords)
        
        cv2.imshow('Facetracker', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

