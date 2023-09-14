import numpy as np2 
import pandas as pds
import os 
import cv2 as cvv \\for capturing
labels = os.listdir("/content/drive/MyDrive/dataset/train") #triaining data path
import matplotlib.pyplot as plot
plot.imshow(plot.imread("/content/drive/MyDrive/Mydataset/training/Closed/_10.jpg"))
def defaces_for_yawning(directory="/content/drive/MyDrive2/Mydataset/training",
faces_cascade_paths="/content/drive/MyDrive/Mydataset/haarcascade_frontalface_alt.xml"):
 yawning_no = []
 imgsize = 145
 categories2 = ["yawn", "no_yawn"]
 for category1 in categories1:
 path_links = os.path.join(directory, category1)
 class_number = categories1.index(category1)
 print(class_number)
 for image2 in os.listdir(path_links):
 img_array = cvv.imread(os.path.join(path_links, image2), cvv.IMREAD_COLOR)
 faces_cascader = cvv.CascadeClassifier(faces_cascade_paths)
 faces = faces_cascader.detectMultiScale(img_array, 1.3, 5)
 for (p, q, r, s) in faces:
 img = cvv.rectangle(image_array, (p, q), (p+r, q+s), (0, 255, 0), 2)
 roicolor = img[q:q+s, p:p+r]
 resize_array = cvv.resize(roicolor, (imgsize, imgsize))
 yawning_no.append([resize_array, class_number])
 return yawning_no
yawn_no_yawn = faces_for_yawning()
def getting_data(directory_paths="/content/drive/MyDrive/Mydataset/training", 
faces_cascad="/content/haarcascade_frontalface_default.xml", 
eye_cascad="/content/drive/MyDrive/Mydataset/haarcascade_eye.xml"):
 labelnames = ['Closed', 'Open']
 imgsize = 145
 data2 = []
 for la in labelnames:
 paths = os.path.join(directory_paths, la)
 class_number = labelnames.index(label)
 class_number +=2
 print(class_number)
 for img2 in os.listdir(paths):
 try:
 imgarray = cvv.imread(os.path.join(paths, imgs), cvv.IMREAD_COLOR)
 resize_array = cvv.resize(imgarray, (imgsize,imgsize))
 data.append([resize_array, class_number])
 except Exception as e:
 print(e)
 return data
data_training = getting_data()
def appending_data():
 yawning_no = faces_for_yawning()
 data = getting_data()
 yawning_no.extend(data)
 return npp.array(yaw_no)
#new variable to store
new_data2 = appending_data()
X = Y= []
for feature2, label2 in new_data2:
 X.append(feature2)
 y.append(label2)
X = npp.array(X)
X = X.reshape(-1, 145, 145, 3)
from sklearn.preprocessing import LabelBinarizer as lb
label_binary = lb()
y = label_binary.fit_transform(y)
#label array
y = npp.array(y)
from sklearn.model_selection import train_test_split as training
seed2 = 43
testsize = 0.20
x_trainset, x_testset, y_trainset, y_testset = training(X, y, random_state=seed2, test_size=testsize)
print(len(x_testset))
len(x_trainset)
#Import necessary modules
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, Conv2D, MaxPooling2D, Dropout #for model purpose
from tensorflow.keras.models import Model //models
from tensorflow.keras.models import Sequential 
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf2
#Data Augmentation
traingenerator = ImageDataGenerator(rescale=1/255, zoom_range=0.2, horizontal_flip=True, rotation_range=30)
testgenerator = ImageDataGenerator(rescale=1/255)
traingenerator = traingenerator.flow(npp.array(x_trainset), y_trainset, shuffle=False)
testgenerator = testgenerator.flow(npp.array(x_testset), y_testset, shuffle=False)
x_trainset = np.array(x_trainset)
model = Sequential()
#with 256 filters
function=”relu”
model.add(Conv2D(256, (3, 3), activation=function, input_shape=x_trainset.shape[1:]))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(MaxPooling2D(2, 2))
#con layer with 64 filters
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(2, 2))
#con layer with 32 filters
model.add(Conv2D(32, (3, 3), activation="relu"))
model.add(MaxPooling2D(2, 2))
model.add(Flatten())
model.add(Dropout(0.5))
#final dense layer with relu activation function
model.add(Dense(64, activation="relu"))
model.add(Dense(4, activation="softmax"))
model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer="adam")
model.summary()
history = model.fit(train_generator, epochs=60, validation_data=testgenerator, shuffle=True, 
validation_steps=len(testgenerator))
loss, accuracy = model.evaluate(testgenerator)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)
from tensorflow.keras.optimizers import Adam
##Deployment code##
import numpy
from pygame import mixer
import cv2 as cvv
from tkinter import *\\for gui purpose
import tkinter.messagebox
import numpy as npp
import dlib
import os
import imutils
from PIL import ImageTk, Image
import tensorflow as tf
from keras.models import load_model
from imutils import face_utils
from scipy.spatial import distance
from playsound import playsound 
from keras.models import load_model
import math
main=Tk()
main.geometry('500x500')
frame2 = Frame(main, relief=RIDGE, borderwidth=3)
frame2.pack(fill=BOTH,expand=1)
main.title("Drowsiness Detection System")
frame2.config(background='white')
label = Label(frame, text="Drowsiness Detection System",bg='white',font=('Times 20 bold'))
label.place(x=5,y=120)
label.pack(side=TOP)
def mouth_aspect_ratio2(mouth2):
A = distance.euclidean(mouth[3], mouth[9])
B = distance.euclidean(mouth[2], mouth[10])
C = distance.euclidean(mouth[4], mouth[8])
L = (A+B+C)/3
D = distance.euclidean(mouth[0], mouth[6])
mar2=L/D
return mar2
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
mouthThresh = 0.60
detect = dlib.get_frontal_face_detector()
frame_check_mouth = 5
flag_mouth=0
mouthThresh = 0.60
# frame to check
frame_check_mouth = 10
(mSt, mEn) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]
labels = ['yawn','no_yawn','Closed', 'Open']
face_cascader= cvv.CascadeClassifier("haarcascade_frontalface_default.xml")
left_eye_cascader = cvv.CascadeClassifier("haarcascade_lefteye_2splits.xml")
right_eye_cascader = cvv.CascadeClassifier("haarcascade_righteye_2splits.xml")
model = load_model("drowiness_new7.h5")
count = count2=0
status1 = 0
status2 = 0
def exit():
 main.destroy()
def start():
 stat1 = 0
 stat2 = 0
 flagmouth=0
 cap =cvv.VideoCapture(0)
 while True:
 _, frame = cap.read()
 frame2 = imutils.resize(frame, height = 800, width=1000)
 gray2 = cvv.cvtColor(frame2, cvv.COLOR_BGR2GRAY)
 subjects = detect(gray2, 0)
 for subject in subjects:
 shape = predict(gray2, subject)
 shape = face_utils.shape_to_np(shape)
 mar = mouth_aspect_ratio(mouth)
 mouthHull = cvv.convexHull(mouth)
 height = frame.shape[0]
 gray = cvv.cvtColor(frame, cvv.COLOR_BGR2GRAY)
 faces = face_cascader.detectMultiScale(gray, 1.3, 5)
 for (p2, q2, r2, s2) in faces:
 img = cvv.rectangle(frame, (p2, q2), (p2+r2, q2+s2), (0, 255, 0), 2)
 roi_color = img[q2:q2+s2, p2:p2+r2]
 feye1= cvv.resize(roi_color, (145, 145))
 feye1 = eye1.astype('float') / 255.0
 feye1 = np.array(eye1)
 feye1 = np.expand_dims(feye1, axis=0)
 predict1 = model.predict(feye1)
 stat3=np.argmax(predict1)
 print(stat3)
 for (p3, q3, r3, s3) in faces:
 cvv.rectangle(frame, (p3, q3), (p3+r3, q3+s3), (0, 255, 0), 1)
 roigray = gray[q3:q3+s3, p3:p3+r3]
 roicolor = frame[q3:q3+s3, p3:p3+r3]
 left_eye2= left_eye_cascader.detectMultiScale(roigray)
 right_eye2 = right_eye_cascader.detectMultiScale(roigray)
 for (r1, s1, t1, u1) in left_eye2:
 cvv.rectangle(roi_color, (r1, s1), (r1 + t1, s1 + u1), (0, 255, 0), 1)
 leye1 = roi_color[s1:s1+u1, r1:r1+t1]
 leye1 = cvv.resize(leye1, (145, 145))
 leye1 = leye1.astype('float') / 255.0
 leye1 = np.array(leye1)
 leye1 = np.expand_dims(leye1, axis=0)
 predict1 = model.predict(leye1)
 stat1=np.argmax(pred1)
 print(stat1)
 #status1 = classes[predict1.argmax(axis=-1)[0]]
 break
 
 for (r2, s2, t2, u2) in right_eye2:
 cvv.rectangle(roi_color, (r2, s2), (r2 + t2, s2 + u2), (0, 255, 0), 1)
 reye2 = roi_color[y2:y2 + h2, x2:x2 + w2]
 reye2 = cvv.resize(reye2, (145, 145))
 reye2 = reye2.astype('float') / 255.0
 reye2 = np.array(reye2)
 reye2 = np.expand_dims(reye2, axis=0)
 predict2 = model.predict(reye2)
 stat2=np.argmax(predict2)
 break
 if mar > mouthThresh:
 flagmouth += 1
 if flagmouth >= frame_check_mouth:
 cvv.putText(frame, "***** YAWNING ******", (10, 370),cvv.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
 alarm2()
 else:
 flagmouth=0
 if stat1 == 2 and stat2 == 2: 
 count += 1
 cvv.putText(frame, "Closed Eyes No.of Frames" + str(count), (10, 30), cvv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
 if count >= 4:
 cvv.putText(frame, "*** YOU ARE SLEEPING ***", (10,400),cvv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 cvv.putText(frame, "Drowsiness Alert!!!!!", (100, height-20), cvv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
 alarm2()
 else:
 cvv.putText(frame, "Eyes are Opened", (10, 30), cvv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
 count = 0
 cvv.imshow("Drowsiness Detector", frame)
 if cvv.waitKey(1) & 0xFF == ord('q'):
 break
 main.update()
 cap.release()
 cvv.destroyAllWindows() 
def alarm2():
 mixer.init()
 alarm2=mixer.Sound('beep.wav')
 alarm2.play()
 time.sleep(0.1)
 alarm2.play() 
def stop():
 cap.release()
 cvv.destroyAllWindows()
button2=Button(frame2,padx=5,pady=5,width=40,bg='light blue',fg='white', command=start,text='Start detection')
buttton2.place(x=5,y=176)
button5=Button(frame2,padx=6,pady=6,width=40,bg='light blue',fg='white',command=stop,text='Stop Detection')
button5.place(x=5,y=250)
button5=Button(frame2, padx=6,pady=6,width=6,bg='white',fg='black',text='EXIT',command=exit)
button5.place(x=210,y=320)
main.mainloop()
