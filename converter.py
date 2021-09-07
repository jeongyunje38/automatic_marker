import cv2


class Converter:

    def __init__(self, img):
        self.img = img

    def convert_img(self, border_size=10):
        img = self.img
        img = cv2.resize(img, (28, 28))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, img = cv2.threshold(img, 225, 0, cv2.THRESH_TOZERO_INV)
        ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
        img = cv2.copyMakeBorder(img, top=border_size, bottom=border_size,
                                 left=border_size, right=border_size, borderType=cv2.BORDER_CONSTANT)
        img = cv2.resize(img, (28, 28))

        return img
