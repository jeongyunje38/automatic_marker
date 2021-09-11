import copy
import os
import pickle


class Partitioner:

    def __init__(self, img, num_of_questions, file_directory, save_file_name='xywh.pkl'):
        self.img = img
        self.question_num_divided_by_two = num_of_questions // 2
        self.divide_question_num_by_two = num_of_questions % 2
        self.file_directory = file_directory
        self.save_file_name = save_file_name
        self.x1 = 0
        self.x2 = 0
        self.y = 0
        self.w = 0
        self.h = 0

        if os.path.isfile(self.save_file_name):
            self.load_positions()

    def partition_img(self):
        img = copy.deepcopy(self.img)
        file_directory = self.file_directory
        file_names = []
        imgs = []
        img_num = 1

        for i in range(self.question_num_divided_by_two):
            file_names, imgs = self.basic_actions(img, img_num, file_names, imgs, 'L')
            img_num += 1

        if self.divide_question_num_by_two:
            file_names, imgs = self.basic_actions(img, img_num, file_names, imgs, 'L')
            img_num += 1

        for i in range(self.question_num_divided_by_two):
            file_names, imgs = self.basic_actions(img, img_num, file_names, imgs, 'R')
            img_num += 1

        return file_directory, file_names, imgs

    def basic_actions(self, img, img_num, file_names, imgs, flag):
        file_name = str(img_num) + '.png'

        if flag == 'L':
            x1 = self.x1
            x2 = self.x1 + self.w
            y1 = self.y + self.h * (img_num - 1)
            y2 = self.y + self.h * img_num

        elif flag == 'R':
            x1 = self.x2
            x2 = self.x2 + self.w

            if self.divide_question_num_by_two:
                y1 = self.y + self.h * (img_num - self.question_num_divided_by_two - 2)
                y2 = self.y + self.h * (img_num - self.question_num_divided_by_two - 1)

            else:
                y1 = self.y + self.h * (img_num - self.question_num_divided_by_two - 1)
                y2 = self.y + self.h * (img_num - self.question_num_divided_by_two)

        roi = img[y1:y2, x1:x2]

        file_names.append(file_name)
        imgs.append(roi)

        return file_names, imgs

    def load_positions(self):
        with open(self.save_file_name, 'rb') as f:
            xywh = pickle.load(f)

        self.x1 = xywh[0]
        self.x2 = xywh[1]
        self.y = xywh[2]
        self.w = xywh[3]
        self.h = xywh[4]

        print('complete loading positions')
