import logging
import os
import shutil

import cv2
from peewee import SqliteDatabase, Model, CharField, ForeignKeyField

from config import IMAGE_DIR

db = SqliteDatabase("data/images.db")


class BaseModel(Model):
    class Meta:
        database = db


class Label(BaseModel):
    name = CharField()

    def persist(self):
        path = os.path.join(IMAGE_DIR, self.name)
        if os.path.exists(path) and len(os.listdir(path)) >= 10:
            shutil.rmtree(path)

        if not os.path.exists(path):
            logging.info("Created directory: %s" % self.name)
            os.makedirs(path)

        Label.get_or_create(name=self.name)


class Image(BaseModel):
    path = CharField()
    label = ForeignKeyField(Label)

    def persist(self, cv_image):

        from image_functions import detect_faces, to_grayscale, crop_faces
        path = os.path.join(IMAGE_DIR, self.label.name)
        nr_of_images = len(os.listdir(path))
        if nr_of_images >= 10:
            return 'Done'
        faces = detect_faces(cv_image)
        if len(faces) > 0 and nr_of_images < 10:
            path += "/%s.jpg" % nr_of_images
            path = os.path.abspath(path)
            logging.info("Saving %s" % path)
            cropped = to_grayscale(crop_faces(cv_image, faces))
            cv2.imwrite(path, cropped)
            self.path = path
            self.save()
