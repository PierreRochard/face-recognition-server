import logging
import os
import sys

import cv2
import numpy as np

from config import MODEL_FILE
from models import Label, Image


def detect_faces(image):
    cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_alt.xml")
    grayscale_image = to_grayscale(image)
    rectangles = cascade.detectMultiScale(grayscale_image,
                                          scaleFactor=1.3,
                                          minNeighbors=4,
                                          minSize=(30, 30),
                                          flags=cv2.CASCADE_SCALE_IMAGE)
    if len(rectangles) == 0:
        return []
    return rectangles


def to_grayscale(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    except TypeError:
        print image
        raise
    gray = cv2.equalizeHist(gray)
    return gray


def crop_faces(image, faces):
    for face in faces:
        x, y, h, w = [result for result in face]
        return image[y:y + h, x:x + w]


def load_images(path):
    images, labels = [], []
    c = 0
    print "test " + path
    for dirname, dirnames, filenames in os.walk(path):
        print "test"
        for subdirname in dirnames:
            subjectPath = os.path.join(dirname, subdirname)
            for filename in os.listdir(subjectPath):
                try:
                    img = cv2.imread(os.path.join(subjectPath, filename),
                                     cv2.IMREAD_GRAYSCALE)
                    images.append(np.asarray(img, dtype=np.uint8))
                    labels.append(c)
                except IOError, (errno, strerror):
                    print "IOError({0}): {1}".format(errno, strerror)
                except:
                    print "Unexpected error:", sys.exc_info()[0]
                    raise
            c += 1
        return images, labels


def load_images_to_db(path):
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            label = Label.get_or_create(name=subdirname)[0]
            label.save()
            for filename in os.listdir(subject_path):
                path = os.path.abspath(os.path.join(subject_path, filename))
                # logging.info('saving path %s' % path)
                image = Image.get_or_create(path=path, label=label)[0]
                image.save()


def load_images_from_db():
    images, labels = [], []
    for label in Label.select():
        for image in label.image_set:
            try:
                cv_image = cv2.imread(image.path, cv2.IMREAD_GRAYSCALE)
                if cv_image is not None:
                    cv_image = cv2.resize(cv_image, (100, 100))
                    images.append(np.asarray(cv_image, dtype=np.uint8))
                    labels.append(label.id)
            except IOError, (errno, strerror):
                print "IOError({0}): {1}".format(errno, strerror)
    return images, np.asarray(labels)


def train():
    images, labels = load_images_from_db()
    model = cv2.createFisherFaceRecognizer()
    # model = cv2.createEigenFaceRecognizer()
    model.train(images, labels)
    model.save(MODEL_FILE)


def predict(cv_image):
    faces = detect_faces(cv_image)
    result = None
    if len(faces) > 0:
        cropped = to_grayscale(crop_faces(cv_image, faces))
        resized = cv2.resize(cropped, (100, 100))

        model = cv2.createFisherFaceRecognizer()
        # model = cv2.createEigenFaceRecognizer()
        model.load(MODEL_FILE)
        prediction = model.predict(resized)
        result = {
            'face': {
                'name': Label.get(Label.id == prediction[0]).name,
                'distance': prediction[1],
                'coords': {
                    'x': str(faces[0][0]),
                    'y': str(faces[0][1]),
                    'width': str(faces[0][2]),
                    'height': str(faces[0][3])
                }
            }
        }
    return result


if __name__ == "__main__":
    load_images_to_db("data/images")
    # train()

    print 'done'
    # predict()
    # train()
