from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime

#stream = cv2.VideoCapture(2)

stream = cv2.VideoCapture("D:/Videos/2024-10-26 19-27-42.mkv")


#

# initialize face_detector model based on Haar Algorithm
faceDetector = cv2.CascadeClassifier('DATA/haarcascade_frontalface_default.xml')

with open('DATA/names.pkl', 'rb') as f:
    LABLES = pickle.load(f)

with open('DATA/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABLES)

# Initialize coloum for the attendance
COL_NAMES = ['NAME', "TIME"]

while True:
    ret, frame = stream.read()  # reed the frame as image from the stream
    if not ret:
        print("Failed to capture image")
        break
    # transfer the colored(BGR) frame to the gray scale for the detector
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detectMultiScale method will output position (x,y,w,h) of the detected face in the frame_gray
    faces = faceDetector.detectMultiScale(frame_gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Crop the imag to get only the deteceted face and put 100 samples in a list
        crop_img = frame[y:y+h, x:x+w, :]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)

        # The output will contain the recognized face, and it's attached label by the ML Algo
        output = knn.predict(resized_img)

        # get a time object and determine day-month-year and put them on date
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")


        pathExist = os.path.isfile('Attendance/Attendance_' + date + '.csv')

        # print the counter of the faces samples taken on the frame
        # Note: output[0] contains the attached Label for the recognised face
        cv2.putText(frame, str(output[0]), (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        # draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)

        attendance = [str(output[0]), str(timestamp)]

    # Show the frame in the stream window
    cv2.imshow('frame_gray', frame)
    # Exit program when press Esc or when you take 100 samples of the face
    key = cv2.waitKey(1)

    # When user press o take the attendance
    if key == ord('o'):
        #time.sleep(5)
        if pathExist:
            with open('Attendance/Attendance_' + date + '.csv', '+a') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(attendance)
            csvfile.close()
        else:
            with open('Attendance/Attendance_' + date + '.csv', '+a') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(COL_NAMES)
                writer.writerow(attendance)
            csvfile.close()

    if key == 27 :
        break

stream.release()
cv2.destroyAllWindows()



