import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # loading files
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
cap = cv2.VideoCapture(0)
while True:
    ret, image = cap.read(0)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(image, "Face", (x, y+h+25), font, 2, (255, 0, 0), 1)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = image[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
            font1 = cv2.FONT_HERSHEY_PLAIN
            cv2.putText(roi_color, "Eye", (ex, ey + eh + 20), font1, 1, (255, 0, 0), 1)
    cv2.imshow("image", image)

    k = cv2.waitKey(1)
    if (k == ord('q')):
        break

cap.release()
cv2.destroyAllWindows()


