#test
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import numpy as np

new_model = tf.keras.models.load_model("best.h5")
path = 'traintest\spec2\opencv_frame_0.png'
img = image.load_img(path, target_size=(300, 300))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x/= 255.
images = np.vstack([x])
predict = new_model.predict(images)
#print(fn)
print(predict[0][0])
if predict[0][0]>predict[0][1]:
    print("caategory 1")
else:print("category 2")