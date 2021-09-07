import cv2
from manager import Manager
import cv2

from manager import Manager

img = cv2.imread('answersheet_converted.jpg')
num_of_questions = 18
manager = Manager(img, num_of_questions)
manager.activate()

# TODO ctrl+alt+O -> import 정리

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