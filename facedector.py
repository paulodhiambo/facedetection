from random import randrange

import cv2

#  Load the front face images dataset
trained_face_data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
img = cv2.imread("elon.jpg")

# capture image from a webcam
webcam = cv2.VideoCapture(0)

# make images greyscale
greyscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# detect faces
face_coordinates = trained_face_data.detectMultiScale(greyscale_img)
print(face_coordinates)
# draw rectangle around the face
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x + w, y + h), (randrange(256), randrange(256), randrange(256)), 2)

cv2.imshow("Codepro face detector", img)
cv2.waitKey()
print("completed")
