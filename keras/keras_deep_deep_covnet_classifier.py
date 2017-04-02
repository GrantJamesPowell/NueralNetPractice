import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils, to_categorical
from keras import metrics
from keras.optimizers import SGD
from keras import backend as K
K.set_image_dim_ordering('th')

import string as base_string

# longest name to use
input_width = 10
num_classes = 3

# load the data
with open('data/boys_names_large.txt') as f:
    boys_names = [x.strip().lower() for x in f.readlines() if len(x.strip()) <= input_width]
    boys_names_set = set(boys_names)

with open('data/girls_names_large.txt') as f:
    girls_names = [x.strip().lower() for x in f.readlines() if len(x.strip()) <= input_width]
    girls_names_set = set(girls_names)

all_names_set = boys_names_set | girls_names_set

# encode the strings as pictures
letters = base_string.ascii_lowercase


def encode_string_as_array(string):
    arr = np.zeros(shape=(input_width, len(letters)))

    for pos, char in enumerate(string):
        arr[pos][letters.index(char)] = 1.

    return arr

# classify the names as an int, we need to use cargorical transformations to one hot encode these
def classify_name(name):
    boy, girl, both = 0, 1, 2
    # O(1) inclusion tests FTW
    if name in boys_names_set and name in girls_names_set: return both
    if name in boys_names_set: return boy
    if name in girls_names_set: return girl


total_data = np.array([encode_string_as_array(name) for name in sorted(all_names_set)])
data_labels_ints = np.array([classify_name(name) for name in sorted(all_names_set)])
data_labels = to_categorical(data_labels_ints, num_classes)

regular_names = np.array([name for name in sorted(all_names_set)])

# reshape the input data
total_data = total_data.reshape(total_data.shape[0], 1, input_width, len(letters))

# randomize the data

perm = np.random.permutation(len(total_data))

total_data = total_data[perm]
data_labels = data_labels[perm]
regular_names = regular_names[perm]

# partition the data into training, and testing
partition = int(len(total_data) * .85)  # 85% training, 15% testing

X, testing_data = total_data[:partition], total_data[partition:]
Y, testing_labels = data_labels[:partition], data_labels[partition:]
reg_names_training, reg_names_testing = regular_names[:partition], regular_names[partition:]

# Build a model


model = Sequential()
model.add(Conv2D(30, (5, 5), input_shape=X[0].shape, activation='relu'))
model.add(Conv2D(15, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(.3))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.summary()

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=[metrics.categorical_accuracy])

model.fit(X, Y, epochs=25, batch_size=128, verbose=True, validation_data=(testing_data, testing_labels))

scores = model.evaluate(X, Y)
print("Accuracy on Testing Data: \n\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# see how we did
predictions = np.argmax(model.predict(testing_data), 1)

# visualize

class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'

num_correct = 0
proto = '|' + '{:^12}|' * 4
print(proto.format('Name', 'Guess', 'Correct', 'Match?'))
for i in range(len(testing_data)):
    name = reg_names_testing[i]
    guess = ['boy', 'girl', 'both'][predictions[i]]
    answer = ['boy', 'girl', 'both'][classify_name(name)]
    match = guess == answer
    col = colors.ok if match else colors.fail
    string = col + proto.format(name, guess, answer, match) + colors.close
    print(string)
    num_correct += match

print()
per_correct = round(num_correct / len(testing_data), 2) * 100
print("Correct on {} of {} ({}%)".format(num_correct, len(testing_data), per_correct))

