import tensorflow as tf
import numpy as np

# helper functions to encode the data replace as needed

# the number of digits our binary representation will have
number_of_digits_in_binary_representation = 12

def binary_encode(number, num_digits):
    # bit shift the number to the right by a certain number of digits and then see if the digit you moved is a one
    # this also returns the array in reverse in the original example, I switched it, not that it matters
    backwards = np.array([(number >> digit) & 1 for digit in range(num_digits)])
    forwards = np.flipud(backwards)
    return forwards


def fizz_buzz_encode(number):
    fizz_buzz = np.array([0,0,0,1])
    buzz =      np.array([0,0,1,0])
    fizz =      np.array([0,1,0,0])
    norm =      np.array([1,0,0,0])

    if (number % 15) == 0:
        return fizz_buzz
    if (number % 5) == 0:
        return buzz
    if (number % 3) == 0:
        return fizz
    return norm

def fizz_buzz_predict(number, prediction):
    return [str(number), "fizz", "buzz", "fizzbuzz"][prediction]


def normal_fizz_buzz(number):
    if (number % 15) == 0:
        return "fizzbuzz"
    if (number % 5) == 0:
        return "buzz"
    if (number % 3) == 0:
        return "fizz"
    return str(number)

# defining the encoding of the data to the model

def input_data_encode(number):
    # if you want to do binary encoding of the number uncomment this line
    # number = np.array(number)
    number = binary_encode(number, number_of_digits_in_binary_representation)
    return number

# the training data iterator, literally any number iterator is fine

def get_training_data_iterator():
    # return an iterator to the training data
    return range(101, 2 ** number_of_digits_in_binary_representation)

def get_training_data_width():
    return input_data_encode(get_training_data_iterator()[0]).size


# label encoding

def label_encode(number):
    return fizz_buzz_encode(number)

def get_label_encode_width():
    return label_encode(get_training_data_iterator()[0]).size


# Create a training set
# in this example I'm using the integer representation of the number

training_set_x = np.array([input_data_encode(i) for i in get_training_data_iterator()])
training_set_labels = np.array([label_encode(i) for i in get_training_data_iterator()])


#initalize the weights

def get_initial_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

number_of_units_in_hidden_layer = 1024
weights_hidden = get_initial_weights([get_training_data_width(), number_of_units_in_hidden_layer])
weights_output = get_initial_weights([number_of_units_in_hidden_layer, get_label_encode_width()])

# Create the place holders for the model, these are what we feed our data through

X = tf.placeholder("float", [None, get_training_data_width()])
Y = tf.placeholder("float", [None, 4])

# make the model, this is done by chaining and logits the relu's together

def make_model(x_place_holder, hidden_weights, output_weights):
    first_hidden_logits = tf.matmul(x_place_holder, hidden_weights)
    added_relu_layer = tf.nn.relu(first_hidden_logits)
    added_output_layer = tf.matmul(added_relu_layer, output_weights)
    return added_output_layer

predict_y_given_x_model = make_model(X, weights_hidden, weights_output)

# make the cost function

soft_max_with_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(predict_y_given_x_model, Y)
cost = tf.reduce_mean(soft_max_with_cross_entropy)

# make the training function based on the cost

learning_rate = .06

# the training operation
training_operation = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# the predition is just taking the greatest value from each of the sets
prediction_operation = tf.argmax(predict_y_given_x_model, 1)


# Now we can actually run the session

batch_size = 128
num_runs = 3500

with tf.Session() as sess:
    tf.initialize_all_variables().run()

    # we will rerun the data until our accuracy is above .995
    for run in range(num_runs):

        #  generate a permutation of equal length to training set
        permutation = np.random.permutation(range(len(training_set_x)))

        #  apply the shuffle
        training_set_x = training_set_x[permutation]
        training_set_labels = training_set_labels[permutation]

        # train in batches
        for start_pos in range(0, len(training_set_x), batch_size):  # range(start, stop, step)
            end_pos = start_pos + batch_size
            # pick out the data we will feed to the model, feed our place holders from before
            feed_dict = {X: training_set_x[start_pos:end_pos], Y: training_set_labels[start_pos:end_pos]}
            # run the training model with the feed dict from the data
            sess.run(training_operation, feed_dict=feed_dict)

        # Every iteration print out the run number and the accuracy of
        accuracy_of_run = np.mean(
            np.argmax(training_set_labels, axis=1) ==sess.run(prediction_operation, feed_dict={X:training_set_x,
                                                                                               Y: training_set_labels})
        )
        print(run, accuracy_of_run)

        # Early Termination
        if accuracy_of_run > .995:
            break

    # Run our validation data
    numbers = np.arange(1, 101)
    final_x = np.transpose(binary_encode(numbers, number_of_digits_in_binary_representation))
    predicted_y = sess.run(prediction_operation, feed_dict={X: final_x})
    print(predicted_y)
    output = [fizz_buzz_predict(i, predicted_y[i - 1]) for i in numbers]

    # Print out the Answer
    proto = "{:^8}" * 10
    for i in range(10):
        print(proto.format(*[output[i * 10 + j] for j in range(10)]))

    # Calculate the accuracy of the run
    is_correct = [output[i - 1] == normal_fizz_buzz(i) for i in numbers]
    accuracy = sum(is_correct) / len(is_correct)
    print("Accuracy {}".format(accuracy))






