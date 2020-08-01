from random import randrange

import cv2

#  Load the front face images dataset
trained_face_data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# img = cv2.imread("images/elon.jpg")

# capture image from a webcam
webcam = cv2.VideoCapture(0)
# key = cv2.waitKey(1)
while True:
    # read successful frames
    successful_frame_read, frame = webcam.read()
    # make images greyscale
    greyscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect faces
    face_coordinates = trained_face_data.detectMultiScale(greyscale_img)
    # draw rectangle around the face
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (randrange(256), randrange(256), randrange(256)), 2)

    cv2.imshow("CodePro face detector", frame)
    key = cv2.waitKey(1)

    # Quit if Q is pressed
    if key == 81 or key == 113:
        break
# Release the video object
webcam.release()
