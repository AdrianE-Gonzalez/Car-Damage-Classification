from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras
import functions as f
print("TensorFlow version is ", tf.__version__)
import matplotlib.pyplot as plt
import build_model as build
import pandas as pd

base_dir = f.get_base_dir()
print(f.get_base_dir())
train_dir = f.get_train_dir()
print(f.get_train_dir())
val_dir = f.get_val_dir()
print(f.get_val_dir())

#get subdirectories
train_subdir = f.get_subdirectories(train_dir)
val_subdir = f.get_subdirectories(val_dir)
n = train_subdir.__len__()

#create image data generator with image augmentation
image_size = 160 # All images will be resized to 160x160
batch_size = 32

IMG_SHAPE = (image_size, image_size, 3)

classes = []
classes.append('damage')
classes.append('whole')

print('{} vs {}'.format(classes[0], classes[1]))
# Rescale all images by 1./255 and apply image augmentation
train_datagen = keras.preprocessing.image.ImageDataGenerator(
                rescale=1./255)

validation_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
                train_dir,  # Source directory for the training images
                classes={classes[0]: 0,
                         classes[1]: 1},
                target_size=(image_size, image_size),
                batch_size=batch_size,
                # Since we use binary_crossentropy loss, we need binary labels
                class_mode='binary')

# Flow validation images in batches of 20 using test_datagen generator
validation_generator = validation_datagen.flow_from_directory(
                val_dir, # Source directory for the validation images
                classes={classes[0]:0,
                         classes[1]:1},
                target_size=(image_size, image_size),
                batch_size=batch_size,
                class_mode='binary')

# Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

base_model.trainable = False

# classification head
model = build.simple_binary_model(base_model)

# compilation of the model
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Training
epochs = 20
steps_per_epoch = train_generator.n // batch_size
validation_steps = validation_generator.n // batch_size

history = model.fit_generator(train_generator,
                              steps_per_epoch=steps_per_epoch,
                              epochs=epochs,
                              workers=4,
                              validation_data=validation_generator,
                              validation_steps=validation_steps,
                              verbose=1)

# accuracy and loss graph
acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

# fine tuning
# unfreeze the top layers of the model
base_model.trainable = True

# Layers in the base model
#print("Number of layers in the base model: ", len(base_model.layers))

# Fine tune from this layer onwards
fine_tune_at = 100

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# recompiling the model
model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(lr=2e-5),
              metrics=['accuracy'])

#model.summary()

#print('# of variables: {}'.format(len(model.trainable_variables)))

# training
history_fine = model.fit_generator(train_generator,
                                   steps_per_epoch=steps_per_epoch,
                                   epochs=epochs,
                                   workers=4,
                                   validation_data=validation_generator,
                                   validation_steps=validation_steps,
                                   verbose=1)

# learning curves
acc += history_fine.history['acc']
val_acc += history_fine.history['val_acc']

loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([.3, 1])
plt.plot([epochs - 1, epochs - 1], plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1])
plt.plot([epochs - 1, epochs - 1], plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
filename = 'out/' + classes[0] + '-vs-' + classes[1] + '_binary_test.png'
plt.savefig(filename)

pd.DataFrame(history.history['val_acc']).to_csv('accuracies/'+ classes[0] +'-vs-'+ classes[1] +'binary_test.csv')

model_path = 'models/' + classes[0] + 'v' + classes[1]
model.save_weights(model_path + 'model_weights.h5')

with open(model_path + 'model_architecture.json', 'w') as f:
    f.write(model.to_json())