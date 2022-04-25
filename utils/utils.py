import cv2
import keras.models
import numpy as np


def process_video(model: keras.models.Model, vcap: cv2.VideoCapture, video_writer: cv2.VideoWriter,
                  threshold: float = 0.5) -> None:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, im = vcap.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        if not ret:
            break
        else:
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                face_img = im[y:y + h, x:x + w]
                rect_resized = cv2.resize(face_img, (224, 224))
                normalized = rect_resized / 255.0
                reshaped = np.reshape(normalized, (1, 224, 224, 3))
                reshaped = np.vstack([reshaped])
                result = model.predict(reshaped)[0][0]
                print(result)

                if result > threshold:
                    cv2.putText(im, "No mask!", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 2)
                if result <= threshold:
                    cv2.putText(im, "Mask" + str(), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 128, 0), 2)
                    cv2.rectangle(im, (x, y), (x + w, y + h), (0, 128, 0), 2)

        video_writer.write(im)
        cv2.imshow('frame', im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vcap.release()
    video_writer.release()

    cv2.destroyAllWindows()
