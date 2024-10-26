import cv2
import pickle
# Pickle is a Python module used for serializing and deserializing objects.
# Serialization refers to converting an object into a byte stream (a format that can be stored in a file or transmitted over a network),
# while deserialization is the reverse process, converting a byte stream back into a Python object.
import numpy as np
import os

stream = cv2.VideoCapture(2)
if not stream.isOpened():
    print("Error: Could not open webcam.")
    exit()

# initialize face_detector model based on Haar Algorithm
faceDetector = cv2.CascadeClassifier('DATA/haarcascade_frontalface_default.xml')

# This list will contain 100 samples of the detected face to pass it to ML model
faces_data = []

i=0

name = input("Enter your name")

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
        resized_img = cv2.resize(crop_img, (50, 50))
        if len(faces_data) <= 100 and i%10==0:      # i%10  to take a sample every 10 frames bcz fps is so large
            faces_data.append(resized_img)
        i=i+1
        # print the counter of the faces samples taken on the frame
        cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

        # draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # Show the frame in the stream window
    cv2.imshow('frame_gray', frame)
    # Exit program when press Esc or when you take 100 samples of the face
    key = cv2.waitKey(1)
    if key == 27 or len(faces_data) == 100:
        break

stream.release()
cv2.destroyAllWindows()

# list as numpy array
faces_data = np.array(faces_data)
#  This line reshaping faces_data into a new 2D array with 100 rows, and each row contains the flattened pixel values of a single face image.
#  This is a common preprocessing step before feeding image data into machine learning models, where the input often needs to be in a 2D array format.
faces_data = faces_data.reshape(100, -1)

# Check if the pickle file created or not,
# if not, put name as a list and serialize it on the file
if 'names.pkl' not in os.listdir('DATA/'):
    names=[name]*100
    with open('DATA/names.pkl', 'wb') as f:
        pickle.dump(names, f)
else:
    if os.path.getsize('DATA/names.pkl') > 0:  # Check if the file is not empty
        with open('DATA/names.pkl', 'rb') as f:
            names = pickle.load(f)
    else:
        names = []
    names=names+[name]*100
    with open('DATA/names.pkl', 'wb') as f:
        pickle.dump(names, f)


# same thing with faces we will crate serealized data file for it
if 'faces_data.pkl' not in os.listdir('DATA/'):
    with open('DATA/faces_data.pkl', 'wb') as f:
        pickle.dump(faces_data, f)
else:
    with open('DATA/faces_data.pkl', 'rb') as f:
        faces=pickle.load(f)
    faces=np.append(faces, faces_data, axis=0)
    with open('DATA/faces_data.pkl', 'wb') as f:
        pickle.dump(faces, f)