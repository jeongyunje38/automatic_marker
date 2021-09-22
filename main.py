import cv2
from manager import Manager

img = cv2.imread('answersheet_converted.jpg')
cv2.imshow('img', img)
num_of_questions = 18
answers = {1: 6, 2: 54, 3: 3, 4: 0, 5: 211, 6: 1, 7: 80, 8: 27, 9: 343,
           10: 512, 11: 1004, 12: 98, 13: 15, 14: 77, 15: 829, 16: 73, 17: 8, 18: 96}
distribution_of_marks = {1: 4.3, 2: 4.4, 3: 4.6, 4: 4.7, 5: 4.9, 6: 5.0, 7: 5.1, 8: 5.2, 9: 5.5,
                         10: 5.7, 11: 5.9, 12: 6.0, 13: 6.1, 14: 6.2, 15: 6.4, 16: 6.5, 17: 6.7, 18: 6.8}
manager = Manager(img, num_of_questions, answers, distribution_of_marks)
manager.activate()
cv2.waitKey(0)

'''
network = Neuralnet(input_size=784, hidden_size=50, output_size=10)

path = './images/'
dir_list = os.listdir(path)
for item in dir_list:
    file_name = path + item
    transformer = Transformer(file_name)
    img = transformer.convert()
    cv2.imshow('image', img)
    img = np.reshape(img, 784)
    network.predict(img, verbose=True)
    cv2.waitKey(0)
'''

'''
(x_train, t_train), (x_test, t_test) = load_mnist()
n = np.random.randint(1, 60000)
sample = x_train[n]
sample = sample.reshape(28, 28)
plt.imshow(sample)

sample = sample.reshape(784)
network = Neuralnet(input_size=784, hidden_size=50, output_size=10)
network.predict(sample, verbose=True)
print('The answer is ... ' + str(t_train[n]))

plt.show()
'''