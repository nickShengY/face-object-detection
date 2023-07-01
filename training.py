import numpy as np
from config import EPOCHS
from model import model, facetracker
from model import FaceTracker
from data_augmentation import train, test, val
import tensorflow as tf
from matplotlib import pyplot as plt
import cv2

def train_model():
    logdir='logs'
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    hist = model.fit(train, epochs=EPOCHS, validation_data=val, callbacks=[tensorboard_callback])

    hist.history

    fig, ax = plt.subplots(ncols=3, figsize=(20,5))

    ax[0].plot(hist.history['total_loss'], color='teal', label='loss')
    ax[0].plot(hist.history['val_total_loss'], color='orange', label='val loss')
    ax[0].title.set_text('Loss')
    ax[0].legend()

    ax[1].plot(hist.history['class_loss'], color='teal', label='class loss')
    ax[1].plot(hist.history['val_class_loss'], color='orange', label='val class loss')
    ax[1].title.set_text('Classification Loss')
    ax[1].legend()

    ax[2].plot(hist.history['regress_loss'], color='teal', label='regress loss')
    ax[2].plot(hist.history['val_regress_loss'], color='orange', label='val regress loss')
    ax[2].title.set_text('Regression Loss')
    ax[2].legend()

    plt.show()

    print("validating with holdout validation set:")

    test_data = test.as_numpy_iterator()

    test_sample = test_data.next()

    yhat = facetracker.predict(test_sample[0])

    fig, ax = plt.subplots(ncols=4, figsize=(20,20))
    for idx in range(4): 
        sample_image = test_sample[0][idx]
        sample_coords = yhat[1][idx]
        
        if yhat[0][idx] > 0.9:
            cv2.rectangle(sample_image, 
                        tuple(np.multiply(sample_coords[:2], [120,120]).astype(int)),
                        tuple(np.multiply(sample_coords[2:], [120,120]).astype(int)), 
                                (255,0,0), 2)
        
        ax[idx].imshow(sample_image)
        
        
    from tensorflow.keras.models import load_model

    facetracker.save('facetracker.h5')

    facetracker = load_model('facetracker.h5')
    return facetracker

if __name__ == "__main__":
    pass
    #train_model()
    



