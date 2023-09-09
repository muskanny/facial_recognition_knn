import cv2
import numpy as np
import os
#click 20 pictures of the person
# Read images
cam = cv2.VideoCapture(0)
fileName = input("Enter the name of the person")
dataset_path = "./data"
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)
model = cv2.CascadeClassifier("C:\\Users\\Muskan Singh\\AppData\\Local\\Packages\\PythonSoftwareFoundation.Python.3.10_qbz5n2kfra8p0\\LocalCache\\local-packages\\Python310\\site-packages\\cv2\\data\\haarcascade_frontalface_alt.xml")
#create a list to save face data
faceData=[]
skip=0
while True:
    success, img = cam.read()
    if not success:
        print("Reading camera failed!")

    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = model.detectMultiScale(img, 1.3, 5)
    # Pick the face with the largest bounding box
    faces = sorted(faces, key=lambda f: f[2] * f[3])
    # Pick the largest face
    if len(faces)>0:
        f = faces[-1]

        x, y, w, h = f
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Crop and save the largest face
        cropped_face = img[y:y + h, x:x + w]
        cropped_face=cv2.resize(cropped_face,(100,100))
        skip+=1
        if skip %10==0:
            faceData.append(cropped_face)
            print("saved so far" + str(len(faceData)))

    cv2.imshow("image window", img)
    #cv2.imshow("cropped face", cropped_face)
    key = cv2.waitKey(1)  # Pause here for 1 ms before you read the next image
    if key == ord("q"):
        break
        
#write the face data on the disk
faceData=np.asarray(faceData)
m=faceData.shape[0]
faceData=faceData.reshape((m,-1))


print(faceData.shape)

#save on the disk as np array
filepath = os.path.join(dataset_path, fileName + ".npy")

np.save(filepath, faceData)
print("data saved successfully"+filepath)

