from PIL import Image
import cv2

import tensorflow as tf
import numpy as np

from googlesearch import search
import google.generativeai as palm

palm.configure(api_key='YOUR_API_KEY_HERE')
models = [m for m in palm.list_models() if 'generateText' in m.supported_generation_methods]
palm_model = models[0].name


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

model = tf.keras.models.load_model('models/resnet50')

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
        if key == 32: # SPACE
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ingredients += predict(img) + ", "
        if key == 27: # ESCAPE
            #links = search(ingredients + "recipe", num=5, stop=5, pause=2)
            print("Here's a recipe with " + ingredients[:-2])
            print("----------")
            completion = palm.generate_text(
                model=palm_model,
                prompt="Give me a recipe with these ingredients:" + ingredients[:-2] + ". Also tell me the name of the dish.",
                temperature=0,
                # The maximum length of the response
                max_output_tokens=1200,
            )
            print(completion.result)
            print("----------")

    vc.release()
    cv2.destroyWindow("preview")