import cv2
import os

def main():
    def begin():
        option = int(input("Would you like to get facial detection from your camera or an image?\n"
                           "1. Use Camera    2. Use image\n >> "))
        return option

    def upload():
        while True:
            path = input("Enter the file path to the photo that you would like to use\n >> ")
            image = cv2.imread(f'{path}')
            cv2.imshow('Original', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return image, path

    def cameraDetection():
        fCascPath = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_default.xml"
        eCascPath = os.path.dirname(cv2.__file__) + "/data/haarcascade_eye.xml"

        faceCascade = cv2.CascadeClassifier(fCascPath)
        eyeCascade = cv2.CascadeClassifier(eCascPath)

        video_capture = cv2.VideoCapture(0)
        while True:
            ret, frames = video_capture.read()
            gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=21, minSize=(30, 30),
                                                 flags=cv2.CASCADE_SCALE_IMAGE)
            eyes = eyeCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=20, minSize=(30, 30),
                                               flags=cv2.CASCADE_SCALE_IMAGE)
            for (x, y, w, h) in faces:
                cv2.rectangle(frames, (x, y), (x + w, y + h), (0, 255, 0), 2)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(frames, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)

            font = cv2.FONT_HERSHEY_TRIPLEX
            x, y, w, h = 0, 0, 400, 50
            cv2.rectangle(frames, (x, x), (x + w, y + h), (0, 0, 0), -1)
            cv2.putText(frames, "Found {0} faces".format(len(faces)) + " and {0} eyes".format(len(eyes)),
                        (x + int(w / 10), y + int(h / 1.5)), font, 0.7, (255, 255, 255), 2, 5)

            cv2.imshow('Faces and Eyes Detected', frames)
            if cv2.waitKey(1) & 0xFF == ord(' '):
                cv2.destroyAllWindows()
                break
        video_capture.release()
        cv2.destroyAllWindows()
        print("Found {0} faces".format(len(faces)) + " and {0} eyes".format(len(eyes))+"/n")

    def imageDetection():
        image, path = upload()
        image = cv2.imread(path, 1)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        fCascPath = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_default.xml"
        eCascPath = os.path.dirname(cv2.__file__) + "/data/haarcascade_eye.xml"

        faceCascade = cv2.CascadeClassifier(fCascPath)
        eyeCascade = cv2.CascadeClassifier(eCascPath)

        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=21, minSize=(30, 30),
                                             flags=cv2.CASCADE_SCALE_IMAGE)
        eyes = eyeCascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=20, minSize=(20, 20),
                                           flags=cv2.CASCADE_SCALE_IMAGE)
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (2, 255, 2), 15)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(image, (ex, ey), (ex + ew, ey + eh), (255, 2, 2), 15)

        font = cv2.FONT_HERSHEY_DUPLEX
        x, y, w, h = 0, 0, 400, 50
        cv2.rectangle(image, (x, x), (x + w, y + h), (0, 0, 0), -1)
        cv2.putText(image, "Found {0} faces".format(len(faces)) + " and {0} eyes".format(len(eyes)),
                    (x + int(w / 10), y + int(h / 2)),
                    font, 0.6, (255, 255, 255), 2, 5)

        cv2.imshow("Faces and Eyes Detected", image)
        cv2.waitKey(1)
        cv2.destroyAllWindows()

    option = begin()
    if option == 1:
        cameraDetection()
    if option == 2:
        imageDetection()


if __name__ == '__main__':
    main()
    while True:
        ending = int(input("Would you like to restart this program or finish?\n"
                           "1. Restart Program    2. Finish Program\n"
                           ">> "))
        if ending == 1:
            main()
        if ending == 2:
            print("All Done!\n"
                  "Created by: Jordyn Wright")
            break
