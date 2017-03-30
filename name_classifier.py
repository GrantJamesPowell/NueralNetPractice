import tensorflow as tf
import numpy as np

import string as base_string

# longest name to use
input_width = 10

# load the data
with open('data/boys_names.txt') as f:
    boys_names = [x.strip().lower() for x in f.readlines() if len(x.strip()) <= input_width]

with open('data/girls_names.txt') as f:
    girls_names = [x.strip().lower() for x in f.readlines() if len(x.strip()) <= input_width]


# encode the strings as pictures
letters = base_string.ascii_lowercase


def encode_string_as_array(string):
    arr = np.zeros(shape=(input_width, len(letters)))

    for pos, char in enumerate(string):
        arr[pos][letters.index(char)] = 1.

    return arr

boy = np.array([0,1])
girl = np.array([1,0])

# get the data and the labels together
total_data = np.array([encode_string_as_array(name) for name in (boys_names + girls_names)])
data_labels = np.array([0 for _ in boys_names] + [1 for _ in girls_names])  # 0 for boys, 1 for girls


# shuffle the data
permutation = np.random.permutation(range(len(total_data)))
total_data = total_data[permutation]
data_labels = data_labels[permutation]


# partition the data into training, validation, and testing
p1 = int(len(total_data) * .8)
p2 = p1 + int(len(total_data) * .1)

training_data, validation_data, testing_data = total_data[:p1], total_data[p1:p2], total_data[p2:]
training_labels, validation_labels, testing_labels = data_labels[:p1], data_labels[p1:p2], data_labels[p2:]


# create the place holders

X = tf.placeholder("float", [input_width, len(letters)])
Y = tf.placeholder("float", [None, 1])




