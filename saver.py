import os

import cv2


class Saver:

    def __init__(self):
        None

    def create_folder(self, file_directory):
        try:
            if not os.path.exists(file_directory):
                os.makedirs(file_directory)
        except OSError:
            print('can\'t make directory: ' + file_directory)

    def save_imgs(self, file_directory, file_name, img):
        self.create_folder(file_directory)
        file_location = file_directory + file_name
        cv2.imwrite(file_location, img)
