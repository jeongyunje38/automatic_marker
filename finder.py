import copy

import cv2


class Finder:

    def __init__(self, img, question_num, file_directory):
        self.question_num = question_num
        self.img = img
        self.img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        ret, self.img_to_zero_inv = cv2.threshold(self.img_gray, 230, 0, cv2.THRESH_TOZERO_INV)
        ret, self.img_binary = cv2.threshold(self.img_to_zero_inv, 0, 1, cv2.THRESH_BINARY)
        self.contours, hierarchy = cv2.findContours(self.img_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        self.file_directory = file_directory

    def visualize(self, color=(0, 255, 0), thickness=2):
        img = copy.deepcopy(self.img)
        for i in self.contours:
            x, y, w, h = cv2.boundingRect(i)
            pos1 = (x, y)
            pos2 = (x+w, y+h)
            cv2.rectangle(img, pos1, pos2, color, thickness)
        cv2.imshow('visualized image', img)
        cv2.waitKey(0)

    def find_nums(self):
        img = copy.deepcopy(self.img)
        file_directory = self.file_directory + str(self.question_num) + '/'
        file_names = []
        imgs = []

        w_min = 3
        w_max = 35
        h_min = 20
        h_max = 47

        for contour in self.contours:
            x, y, w, h = cv2.boundingRect(contour)

            if w_min < w < w_max and h_min < h < h_max:
                file_name = str(x) + '_' + str(y) + '.png'
                roi = img[y:y+h, x:x+w]
                file_names.append(file_name)
                imgs.append(roi)

        return file_directory, file_names, imgs
