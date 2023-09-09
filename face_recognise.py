import cv2
import numpy as np
import os

#data prep
dataset_path="./data/"
faceData=[]
labels=[]
nameMap={}

classId=0

for f in os.listdir(dataset_path):
    if f.endswith(".npy"):
        nameMap[classId]=f[:-4]
        #x value
        dataItem=np.load(dataset_path+f)
        m=dataItem.shape[0]#number of images
        #print(dataItem.shape)
        faceData.append(dataItem)

        #yvalues
        target=classId*np.ones((m,))
        classId+=1
        labels.append(target)

print(faceData)
print(labels)
XT=np.concatenate(faceData,axis=0)
yT=np.concatenate(labels, axis=0).reshape((-1,1))

print(XT.shape)
print(yT.shape)
print(nameMap)



#algorithm
def dist(p,q):
    return np.sqrt(np.sum((p-q)**2))
def knn(X,y,xt,k=5):
    m=X.shape[0]
    dlist=[]

    for i in range(m):
        d=dist(X[i],xt)
        dlist.append((d,y[i]))

    dlist=sorted(dlist)
    dlist = np.array([item[1] for item in dlist[:k]])
    #labels=dlist[:,1]

    labels,cnts=np.unique(dlist, return_counts=True)
    idx=cnts.argmax()
    pred=labels[idx]

    return int(pred)

#Predictions

#create a camera object

cam=cv2.VideoCapture(0)

model = cv2.CascadeClassifier("C:\\Users\\Muskan Singh\\AppData\\Local\\Packages\\PythonSoftwareFoundation.Python.3.10_qbz5n2kfra8p0\\LocalCache\\local-packages\\Python310\\site-packages\\cv2\\data\\haarcascade_frontalface_alt.xml")
while True:
    success, img = cam.read()
    if not success:
        print("Reading camera failed!")

    #grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = model.detectMultiScale(img, 1.3, 5)
    # Pick the face with the largest bounding box
    #faces = sorted(faces, key=lambda f: f[2] * f[3])
    # Pick the largest face
    #render a box around each face and predict its name
    for f in faces:
      

        x, y, w, h = f
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Crop and save the largest face
        cropped_face = img[y:y + h, x:x + w]
        cropped_face=cv2.resize(cropped_face,(100,100))
        #skip+=1
        #if skip %10==0:
        #   faceData.append(cropped_face)
        #   print("saved so far" + str(len(faceData)))

        #cv2.imshow("image window", img)
        #predict hte name using knn

        classPredicted=knn(XT,yT,cropped_face.flatten())
        #name
        namePredicted=nameMap[classPredicted]
        #display the name and the box
        cv2.putText(img, namePredicted,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        cv2.rectangle(img,(x,y), (x+w,y+h), (0,255,0),2)
        cv2.imshow("cropped face", cropped_face)
    cv2.imshow("prediction window", img)

    
    key = cv2.waitKey(1)  # Pause here for 1 ms before you read the next image
    if key == ord("q"):
        break
cam.release()
cv2.destroyAllWindows()
