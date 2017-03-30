# Create first network with Keras
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils.np_utils import to_categorical
import numpy as np

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


# fizz buzz specific things that have nothing to do with keras

binary_digits = 10


def binary_encode(number, num_digits):
    backwards = np.array([(number >> digit) & 1 for digit in range(num_digits)])
    forwards = np.flipud(backwards)
    return forwards


def fizz_buzz_encode(number):
    norm, fizz, buzz, fizz_buzz = range(4)
    if (number % 15) == 0: return fizz_buzz
    if (number % 5) == 0: return buzz
    if (number % 3) == 0: return fizz
    return norm


def fizz_buzz_predict(number, prediction):
    return [str(number), "fizz", "buzz", "fizzbuzz"][prediction]

# split into input (X) and output (Y) variables

numbers = np.arange(101,1000)  # train on 101 - 1000 test on 1 - 100
X = np.array([binary_encode(i, binary_digits) for i in numbers])  # use our binary encoder for the input numbers
Y_int = np.array([fizz_buzz_encode(i) for i in numbers])  # our categories are integer i.e. cat 1, cat 2 ...
Y = to_categorical(Y_int, 4)  # switch categories to numpy arrays 1 -> [0,0,0,1], 2 -> [0,0,1,0] ...

# create the model

model = Sequential()  # generic sequential layer model init
# the input layer should be the same size as the the number of inputs
model.add(Dense(binary_digits, input_dim=binary_digits, init='uniform', activation='relu'))
# make a hidden layer with 1024 neurons (or hidden nodes or whatever terminology you're using)
# notice that there is a relu activation
model.add(Dense(1024, init='uniform', activation='relu'))
# create the last layer which has the softmax activation and maps to the four output states
model.add(Dense(4, init='uniform', activation='softmax'))

# compile the model with our loss function and our optimizer

model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.05), metrics=['accuracy'])
# (Categorical entropy is the same as softmax with softmax_cross_entropy_with_logits Im pretty sure)

# the heavy lifting, actually training the model

model.fit(X, Y, nb_epoch=3000, batch_size=128, verbose=False)

# make predictions based on our model with the test data that the model has never seen

test_nums = np.arange(1,101)
test_data = np.array([binary_encode(i, binary_digits) for i in test_nums])
test_label_int = np.array([fizz_buzz_encode(i) for i in test_nums])
test_label = to_categorical(test_label_int, 4)
predictions = model.predict(test_data)

# Print the results (this is just stylistic and has nothing to do with the model)

intpredicts = np.argmax(predictions,1)
fizz_buzzified = [fizz_buzz_predict(i, intpredicts[i - 1]) for i in test_nums]
proto = "{:^8}" * 10
for i in range(len(test_data) // 10):
    print(proto.format(*[fizz_buzzified[i * 10 + j] for j in range(10)]))

# get our accuracy and print it

scores = model.evaluate(test_data, test_label)
print("\n\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))