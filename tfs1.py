spec1dir='F:\SURYA_PROJ\Tensorproj\traintest\spec1'
spec2dir='F:\SURYA_PROJ\Tensorproj\traintest\spec2'

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 300x300 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fifth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')
    tf.keras.layers.Dense(2, activation='sigmoid')
])

from tensorflow.keras.optimizers import RMSprop

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=1e-4),
              metrics=['accuracy'])

img_width = 300
img_height = 300

datagen = ImageDataGenerator(rescale=1/255.0, validation_split=0.2)
# 1) TRAINING SET
train_generator = datagen.flow_from_directory(directory=r'F:\SURYA_PROJ\Tensorproj\traintest',
                                                   target_size=(img_width, img_height),
                                                   batch_size=16,
                                                   subset='training'
                                                   )
# 2)CROSS VALIDATION SET
validation_generator = datagen.flow_from_directory(directory=r'F:\SURYA_PROJ\Tensorproj\traintest',
                                                    target_size=(img_width,img_height),
                                                    batch_size=16,
                                                    subset='validation'
                                                   )
    
history = model.fit(
      train_generator,
      steps_per_epoch=8,  
      epochs=1,
      verbose=1,
      validation_data = validation_generator,
      validation_steps=8)
model.save("best.h5")
