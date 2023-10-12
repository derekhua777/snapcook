import matplotlib.pyplot as plt
import os
from PIL import Image
import cv2

import tensorflow as tf
import numpy as np
from googlesearch import search

NUM_CLASSES = 15


classes = ['Capsicum',
 'Radish',
 'Pumpkin',
 'Cauliflower',
 'Potato',
 'Bean',
 'Cucumber',
 'Brinjal',
 'Cabbage',
 'Broccoli',
 'Bitter_Gourd',
 'Papaya',
 'Bottle_Gourd',
 'Tomato',
 'Carrot']

model = tf.keras.models.load_model('models/resnet50/')

def predict(img):
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(np.array(img), axis=0)
    prediction = classes[np.argmax(model.predict(img)[0])]
    print(prediction)
    return prediction


if __name__ == '__main__':
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    ingredients = ""
    while rval:
        cv2.imshow("preview", frame)
        rval, frame = vc.read()
        key = cv2.waitKey(20)
        if key == 32: # exit on ESC
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ingredients += predict(img) + " "
        if key == 27:
            links = search(ingredients + "recipe", num=5, stop=5, pause=2)
            print("Here are the 5 top recipes with your ingredients")
            print("----------")
            for link in links:
                print(link)
            print("----------")

    vc.release()
    cv2.destroyWindow("preview")