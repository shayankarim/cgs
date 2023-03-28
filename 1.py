
# import required libraries
import numpy as np
import pandas as pd
import keras
from keras_preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# load HAM10000 dataset
data = pd.read_csv('HAM10000_metadata.csv')
labels = data['dx']
image_path = data['image_id'].apply(lambda x: f'HAM10000_images_part_1/{x}.jpg')
images = np.asarray([keras.preprocessing.image.load_img(img, target_size=(224,224)) for img in image_path])
images = np.asarray([keras.preprocessing.image.img_to_array(img) for img in images])

# preprocess data
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

train_datagen = ImageDataGenerator(
        rotation_range=90,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

validation_datagen = ImageDataGenerator()

train_datagen.fit(x_train)

train_generator = train_datagen.flow(
        x_train,
        y_train,
        batch_size=32)

validation_generator = validation_datagen.flow(
        x_test,
        y_test,
        batch_size=32)

# preprocess labels
y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)

# load pre-trained model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# build new model on top of base model
model = Sequential()
model.add(base_model)
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# train model
history = model.fit_generator(
          train_generator,
          epochs=15,
          steps_per_epoch=len(x_train) // 32,
          validation_data=validation_generator,
          validation_steps= len(x_test) // 32)

# evaluate model
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f'Loss on Test Data: {loss:.2f}')
print(f'Accuracy on Test Data: {accuracy*100:.2f}%')

# Classification report and confusion matrix
y_pred = np.argmax(model.predict(x_test), axis=-1)
print(classification_report(y_test.values.argmax(axis=-1), y_pred))

cm= confusion_matrix(y_test.values.argmax(axis=-1), y_pred)
sns.heatmap(cm,annot=True)
plt.show()

# plot metrics
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
