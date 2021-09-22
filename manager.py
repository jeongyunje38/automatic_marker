import copy
import os
import shutil

import cv2
import numpy as np

from converter import Converter
from finder import Finder
from neuralnet import Neuralnet
from partitioner import Partitioner
from saver import Saver


class Manager:

    def __init__(self, img, num_of_questions, answers, distribution_of_marks, partitioner_file_directory='./answers/',
                 finder_file_directory='./found_numbers/', converter_file_directory='./converted_number_images/'):
        self.img = img
        self.num_of_questions = num_of_questions
        self.answers = answers
        self.distribution_of_marks = distribution_of_marks
        self.partitioner_file_directory = partitioner_file_directory
        self.finder_file_directory = finder_file_directory
        self.converter_file_directory = converter_file_directory

    def initialize(self):
        try:
            shutil.rmtree(self.partitioner_file_directory)
            shutil.rmtree(self.finder_file_directory)
            shutil.rmtree(self.converter_file_directory)
        except:
            None

    def save_img(self, file_directory, file_name, img):
        saver = Saver()
        saver.save_imgs(file_directory, file_name, img)

    def save_imgs(self, file_directory, file_names, imgs):
        saver = Saver()
        for file_name, img in zip(file_names, imgs):
            saver.save_imgs(file_directory, file_name, img)

    def partition_img(self, img, num_of_questions, partitioner_file_directory):
        img = copy.deepcopy(img)
        partitioner = Partitioner(img, num_of_questions, partitioner_file_directory)
        r_file_directory, r_file_names, r_imgs = partitioner.partition_img()
        self.save_imgs(r_file_directory, r_file_names, r_imgs)

    def find_nums(self, num_of_questions, partitioner_file_directory, finder_file_directory):
        for question_num in range(1, num_of_questions + 1):
            file_directory = partitioner_file_directory
            file_name = str(question_num) + '.png'
            file_location = file_directory + file_name
            question_img = cv2.imread(file_location)

            finder = Finder(question_img, question_num, finder_file_directory)
            # finder.visualize()
            r_file_directory, r_file_names, r_imgs = finder.find_nums()

            self.save_imgs(r_file_directory, r_file_names, r_imgs)

    def convert_num_imgs(self, num_of_questions, finder_file_directory, converter_file_directory):
        for question_num in range(1, num_of_questions + 1):
            file_directory = finder_file_directory + str(question_num) + '/'
            try:
                file_names = os.listdir(file_directory)

                for file_name in file_names:
                    file_location = file_directory + file_name
                    found_num_img = cv2.imread(file_location)

                    converter = Converter(found_num_img)
                    r_file_directory = converter_file_directory + '/' + str(question_num) + '/'
                    r_file_name = file_name
                    r_img = converter.convert_img()

                    self.save_img(r_file_directory, r_file_name, r_img)
            except:
                None

    def get_answers_as_num(self, neuralnet, num_of_questions, converter_file_directory):
        student_answers = {}
        for question_num in range(1, num_of_questions + 1):
            file_directory = converter_file_directory + str(question_num) + '/'
            student_answers[question_num] = None

            try:
                file_names = os.listdir(file_directory)

                for file_name in file_names:
                    file_location = file_directory + file_name
                    found_answer_img = cv2.imread(file_location, cv2.IMREAD_GRAYSCALE)
                    found_answer_img = np.reshape(found_answer_img, 784)

                    y = neuralnet.predict(found_answer_img)
                    answer_as_num = np.argmax(y)

                    answer = answer_as_num
                    if student_answers[question_num] is not None:
                        answer = student_answers[question_num] * 10 + answer
                    student_answers[question_num] = answer
            except:
                student_answers[question_num] = -1

        return student_answers

    def mark(self, answers, student_answers, distribution_of_marks):
        correct_answers = {i: answers[i] for i in answers if i in student_answers and answers[i] == student_answers[i]}

        score = 0
        for key, val in correct_answers.items():
            score += distribution_of_marks[key]

        return score

    def activate(self):
        self.initialize()
        neuralnet = Neuralnet(input_size=784, hidden_size=50, output_size=10)
        self.partition_img(self.img, self.num_of_questions, self.partitioner_file_directory)
        self.find_nums(self.num_of_questions, self.partitioner_file_directory, self.finder_file_directory)
        self.convert_num_imgs(self.num_of_questions, self.finder_file_directory, self.converter_file_directory)
        student_answers = self.get_answers_as_num(neuralnet, self.num_of_questions, self.converter_file_directory)
        score = self.mark(self.answers, student_answers, self.distribution_of_marks)
        print('score: ' + str(score))
