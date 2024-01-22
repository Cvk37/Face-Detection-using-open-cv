import cv2
import os

dataset = "dataset"
name = "vk"

path = os.path.join(dataset, name)

if not os.path.exists(dataset):
    os.mkdir(dataset)
if not os.path.exists(path):
    os.mkdir(path)

(width, height) = (130, 100)
# Path to the Haar Cascade classifier XML file
alg = "haarcascade_frontalface_default.xml"

# Create a CascadeClassifier object using the Haar Cascade XML file
haar_cascade = cv2.CascadeClassifier(alg)

# Open a connection to the default camera (camera index 0)
cam = cv2.VideoCapture(0)

count = 1
while count < 31:
    # Read a frame from the camera
    ret, img = cam.read()
    

    # Convert the frame to grayscale
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Perform face detection using the Haar Cascade classifier
    faces = haar_cascade.detectMultiScale(grayImg, scaleFactor=1.3, minNeighbors=4)

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        faceOnly = grayImg[y:y + h, x:x + w]
        resizeImg = cv2.resize(faceOnly, (width, height))
        cv2.imwrite("%s/%s.jpg" % (path, count), resizeImg)
        count += 1
        print("Count:", count)

    if len(faces) > 0:
        print("Person detected")
    else:
        print("No Person Detected")

    # Display the image with rectangles drawn around detected faces
    cv2.imshow("FaceDetection", img)

    # Check for the 'Esc' key to exit the loop
    key = cv2.waitKey(10)
    if key == 27:
        break

# Release the camera and close all OpenCV windows outside the loop
cam.release()
cv2.destroyAllWindows()
        

