import argparse
import numpy as np
import tensorflow as tf
from research import CLASSES, load_image

p=argparse.ArgumentParser()
p.add_argument('--model',required=True)
p.add_argument('--image',required=True)
a=p.parse_args()
m=tf.keras.models.load_model(a.model,compile=False)
scores=m(load_image(a.image)[None],training=False).numpy()[0]
print('Predicted category:',CLASSES[int(np.argmax(scores))])
for c,s in sorted(zip(CLASSES,scores),key=lambda x:-x[1]):print(f'{c}: {s:.4f}')
print('Scores are uncalibrated; unknown objects also receive a category.')
