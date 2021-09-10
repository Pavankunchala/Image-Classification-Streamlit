import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.applications.inception_v3 import *
from tensorflow.keras.preprocessing.image import *

import streamlit as st

import streamlit.components.v1 as stc
from PIL import Image
import cv2

DEMO_IMAGE = 'pug.jpg'

HTML_BANNER = """
<div style="background-color:Orange;padding:10px;border-radius:10px">
<h1 style="color:Black;text-align:center;">Image Classifier</h1>
</div>
"""


stc.html(HTML_BANNER)

img_file_buffer = st.sidebar.file_uploader("Upload an image", type=[ "jpg", "jpeg"])

if img_file_buffer is not None:

    image = np.array(Image.open(img_file_buffer))

else:

    demo_image = DEMO_IMAGE

    image = np.array(Image.open(demo_image))


kp2_text = st.markdown('')


model = InceptionV3(weights='imagenet')
original = image

image = cv2.resize(image,(299,299))

#image = load_img('dog.jpg', target_size=(299, 299))

image = img_to_array(image)
image = np.expand_dims(image, axis=0)

image = preprocess_input(image)

predictions = model.predict(image)
prediction_matrix = (imagenet_utils
                     .decode_predictions(predictions))

for i in range(3):
    imagenet_id, label, probability = prediction_matrix[0][i]
    print(f'{i + 1}. {label}: {probability * 100:.3f}%')

_, label, _ = prediction_matrix[0][0]

st.image(original, use_column_width = True )
kp2_text.write(f"<h1 style='color:gold;text-align:center;'>Label is {label}</h1>",unsafe_allow_html=True)
#plt.figure()
#plt.title(f'Label: {label}.')
#original = load_img('dog.jpg')
#original = img_to_array(original)
#plt.imshow(original / 255.0)
#plt.show()