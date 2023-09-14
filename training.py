from google.colab import drive
drive.mount('/content/drive')

!pip install keras

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import cv2

labels = os.listdir("/content/drive/MyDrive/dataset/train")


import matplotlib.pyplot as plt
plt.imshow(plt.imread("/content/drive/MyDrive/dataset/train/Closed/_10.jpg"))

def face_for_yawn(direc="/content/drive/MyDrive/dataset/train", face_cas_path="/content/drive/MyDrive/dataset/haarcascade_frontalface_alt.xml"):
    yaw_no = []
    IMG_SIZE = 145
    categories = ["yawn", "no_yawn"]
    for category in categories:
        path_link = os.path.join(direc, category)
        class_num1 = categories.index(category)
        print(class_num1)
        for image in os.listdir(path_link):
            image_array = cv2.imread(os.path.join(path_link, image), cv2.IMREAD_COLOR)
            face_cascade = cv2.CascadeClassifier(face_cas_path)
            faces = face_cascade.detectMultiScale(image_array, 1.3, 5)
            for (x, y, w, h) in faces:
                img = cv2.rectangle(image_array, (x, y), (x+w, y+h), (0, 255, 0), 2)
                roi_color = img[y:y+h, x:x+w]
                resized_array = cv2.resize(roi_color, (IMG_SIZE, IMG_SIZE))
                yaw_no.append([resized_array, class_num1])
    return yaw_no
yawn_no_yawn = face_for_yawn()

def get_data(dir_path="/content/drive/MyDrive/dataset/train", face_cas="/content/haarcascade_frontalface_default.xml", eye_cas="/content/drive/MyDrive/dataset/haarcascade_eye.xml"):
    labels = ['Closed', 'Open']
    IMG_SIZE = 145
    data = []
    for label in labels:
        path = os.path.join(dir_path, label)
        class_num = labels.index(label)
        class_num +=2
        print(class_num)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
                resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                data.append([resized_array, class_num])
            except Exception as e:
                print(e)
    return data

data_train = get_data()



def append_data():
#     total_data = []
    yaw_no = face_for_yawn()
    data = get_data()
    yaw_no.extend(data)
    return np.array(yaw_no)

#new variable to store

new_data = append_data()

#separate label and features

X = []
y = []
for feature, label in new_data:
    X.append(feature)
    y.append(label)

X = np.array(X)
X = X.reshape(-1, 145, 145, 3)

#LabelBinarizer

from sklearn.preprocessing import LabelBinarizer
label_bin = LabelBinarizer()
y = label_bin.fit_transform(y)

#label array

y = np.array(y)

#train test split

from sklearn.model_selection import train_test_split
seed = 42
test_size = 0.30
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, test_size=test_size)

print(len(X_test))
len(X_train)

#Import necessary modules

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

#Data Augmentation

train_generator = ImageDataGenerator(rescale=1/255, zoom_range=0.2, horizontal_flip=True, rotation_range=30)
test_generator = ImageDataGenerator(rescale=1/255)

train_generator = train_generator.flow(np.array(X_train), y_train, shuffle=False)
test_generator = test_generator.flow(np.array(X_test), y_test, shuffle=False)

X_train = np.array(X_train)

#Model

# !pip install tensorflow==2.3.1
# !pip install keras==2.4.3

model = Sequential()

model.add(Conv2D(256, (3, 3), activation="relu", input_shape=X_train.shape[1:]))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(32, (3, 3), activation="relu"))
model.add(MaxPooling2D(2, 2))

model.add(Flatten())
model.add(Dropout(0.5))

model.add(Dense(64, activation="relu"))
model.add(Dense(4, activation="softmax"))

model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer="adam")

model.summary()

history = model.fit(train_generator, epochs=60, validation_data=test_generator, shuffle=True, validation_steps=len(test_generator))

loss, accuracy = model.evaluate(test_generator)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

#**Alexnet**

from tensorflow.keras.optimizers import Adam

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(accuracy))

plt.plot(epochs, accuracy, "b", label="trainning accuracy")
plt.plot(epochs, val_accuracy, "r", label="validation accuracy")
plt.legend()
plt.show()

plt.plot(epochs, loss, "b", label="trainning loss")
plt.plot(epochs, val_loss, "r", label="validation loss")
plt.legend()
plt.show()

model.save("/content/drive/MyDrive/dataset/drowsiness_new6.h5")

model.save("/content/drive/MyDrive/dataset/drowsiness_new6.model")

print(model.input_shape)

X_test=np.array(X_test)
probas = model.predict(X_test)   # obtain predicted probabilities for each class
prediction = np.argmax(probas, axis=1)  
 # select class with highest probability

labels_new = ["yawn", "no_yawn", "Closed", "Open"]


#prediction

prediction=prediction.reshape(-1,1)


print(type(prediction))

import numpy as np

n_values =int(np.max(prediction) + 1)
print(n_values)
pred = np.eye(n_values)[prediction.astype(int)]


print(y_test.shape)
print(pred.shape)
pred=np.reshape(pred, (535, 4))


from sklearn.metrics import confusion_matrix,accuracy_score

#Generate the confusion matrix
cf_matrix=confusion_matrix(y_test.argmax(axis=1), pred.argmax(axis=1))
print(cf_matrix)
print(accuracy_score(y_test.argmax(axis=1), pred.argmax(axis=1)))

import seaborn as sns

ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')
ax.xaxis.set_ticklabels(['yawn','no_yawn','Closed','Open'])
ax.yaxis.set_ticklabels(['yawn','no_yawn','Closed','Open'])

from sklearn.metrics import classification_report
print(classification_report(np.argmax(y_test, axis=1), prediction, target_names=labels_new))

labels_new = ["yawn", "no_yawn", "Closed", "Open"]
IMG_SIZE = 145
def prepare(filepath, face_cas="/content/drive/MyDrive/dataset/haarcascade_frontalface_alt.xml"):
    img_array = cv2.imread(filepath, cv2.IMREAD_COLOR)
    img_array = img_array / 255
    resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return resized_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)

model = tf.keras.models.load_model("/content/drive/MyDrive/dataset/drowsiness_new6.model")

import matplotlib.pyplot as plt

#Prediction
#0-yawn, 1-no_yawn, 2-Closed, 3-Open

prediction = model.predict([prepare("/content/drive/MyDrive/dataset/train/no_yawn/6.jpg")])
a=prediction
a=np.argmax(prediction)

plt.rcParams['figure.figsize'] = [4, 4]
plt.imshow(plt.imread("/content/drive/MyDrive/dataset/train/no_yawn/6.jpg"))
print(labels_new[a])

prediction = model.predict([prepare("/content/drive/MyDrive/dataset/train/Closed/_107.jpg")])
a=np.argmax(prediction)
print(labels_new[a])
plt.imshow(plt.imread("/content/drive/MyDrive/dataset/train/Closed/_103.jpg"))

prediction = model.predict([prepare("/content/drive/MyDrive/dataset/train/Open/_10.jpg")])
a=np.argmax(prediction)
print(labels_new[a])
plt.imshow(plt.imread("/content/drive/MyDrive/dataset/train/Open/_10.jpg"))

prediction = model.predict([prepare("/content/drive/MyDrive/dataset/train/yawn/16.jpg")])
np.argmax(prediction)
