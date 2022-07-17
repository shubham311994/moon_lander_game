""" Importing the required libraries and this contains the code for Neural Network """
import math
from random import randint
import matplotlib.pyplot as plt


def dot_product(vector1, vector2):
    product = []
    for val in range(len(vector2)):
        product.append(sum(x * y for x, y in zip(vector1, vector2[val])))
    return product


def randomize_weights(x, y):
    weights_array = []
    for val1 in range(x):
        temp = []
        for val2 in range(y):
            temp.append(float(randint(1, 10) / 10))
            val2 += 1
        weights_array.append(temp)
        val1 += 1
    return weights_array


class NeuralNetwork:

    def __init__(self, num_inputs, num_hidden, num_outputs):
        """
        Initializing the network and creating weights matrices with the required dimensions specified by user
        also initializing various parameters required for Forward and Backward propagation while training the model
        """
        self.input_neuron = num_inputs
        self.hidden_neuron = num_hidden
        self.output_neurons = num_outputs
        self.error_value = []
        # reference from machine learning mastery tutorial
        self.hidden_layer_bias = [1]  # initializing the bias value for the input layer
        self.learning_rate = 0.9
        self.momentum_rate = 0.1
        self.output_layer_delta_weights = [[0 for col in range(self.hidden_neuron + 1)] for row in range(
            self.output_neurons)]  # weight vector containing the previous step output layer delta weights
        self.hidden_layer_delta_weights = [[0 for col in range(self.input_neuron + 1)] for row in range(
            self.hidden_neuron)]  # weight vector containing the previous step hidden layer delta weights
        self.hidden_layer_weights = randomize_weights(self.hidden_neuron,
                                                      self.input_neuron + 1)  # the hidden bias weights selected 
        # randomly
        self.output_layer_bias = [1]  # initializing the bias value for the hidden layer
        self.output_layer_weights = randomize_weights(self.output_neurons,
                                                      self.hidden_neuron + 1)  # the output layer bias weights selected 
        # randomly
        self.lambda_value = 0.9

    def activation_function(self, product):
        """ 
        sigmoid activation function to be used for both Hidden and Output layers values compression.
        :param product: value to be compressed using activation function.
        :return activation_values_list: value obtained after compressing.
        """
        activation_values_list = []
        for i in range(len(product)):
            activation_values_list.append(1 / (1 + math.exp(-self.lambda_value * product[i])))
        return activation_values_list

    def derivative_activation_function(self, activated_values):
        """ 
        first order derivative of sigmoid activation function.
        
        :param activated_values: values after applying activation function
        :return derivative_values_list: values obtained after doing first order differentiation.
         """
        derivative_values_list = []
        for i in range(len(activated_values)):
            derivative_values_list.append((self.lambda_value * activated_values[i]) * (1 - activated_values[i]))
        return derivative_values_list

    def local_gradient_of_output_layer(self, output_layer_values, calculated_error):

        """ 
        calculating the local gradients of the output layer during back propagation.
        """

        local_gradient_output_list = []
        for i in range(len(output_layer_values)):
            local_gradient_output_list.append(output_layer_values[i] * calculated_error[i])
        return local_gradient_output_list

    def local_gradient_of_hidden_layer(self, derivative_values, output_layer_local_gradient_values,
                                       hidden_layer_weights_value):

        """ 
        Calculating the local gradients of the Hidden layer during back propagation.
        """
        local_gradient_hidden_value_list = []
        for i in range(len(hidden_layer_weights_value[0])):
            for j in range(len(output_layer_local_gradient_values)):
                out = output_layer_local_gradient_values[j] * hidden_layer_weights_value[j][i]
            local_gradient_hidden_value_list.append(derivative_values[i] * out)
        return local_gradient_hidden_value_list

    def output_layer_delta_weight_update(self, output_layer_gradient_values, hidden_layer_output):
        """ 
        Update the weights based on the lacal gradient values of the output layer.
        """

        delta_output_weights = []
        for i in range(len(output_layer_gradient_values)):
            for j in range(len(hidden_layer_output)):
                delta_output_weights.append(
                    (self.learning_rate * output_layer_gradient_values[i] * hidden_layer_output[j]) + (
                            self.momentum_rate * self.output_layer_delta_weights[i][j]))
        self.output_layer_delta_weights = delta_output_weights
        return delta_output_weights

    def hidden_layer_delta_weight_update(self, hidden_layer_gradient_values, input_values):
        """ 
        Update the weights based on the gradient values of the Hidden layer.
        """

        delta_hidden_weights = []
        for i in range(len(hidden_layer_gradient_values)):
            temp = []
            for j in range(len(input_values)):
                temp.append((self.learning_rate * hidden_layer_gradient_values[i] * input_values[j]) + (
                        self.momentum_rate * self.hidden_layer_delta_weights[i][j]))
            delta_hidden_weights.append(temp)
        self.hidden_layer_delta_weights = delta_hidden_weights
        return delta_hidden_weights

    def updated_output_layer_weights(self, current_output_weight, updated_output_layer_values):
        """ 
        Update weights based on the delta weights calculated for Output Layer.
        """
        updated_output_weights = []
        for i in range(len(current_output_weight)):
            temp = []
            current_output_weight_values = current_output_weight[i]
            updated_output_layer = updated_output_layer_values[i]
            for j in range(len(current_output_weight_values)):
                temp.append(current_output_weight_values[j] + updated_output_layer[j])
            updated_output_weights.append(temp)
        return updated_output_weights

    def updated_hidden_layer_weights(self, current_hidden_weight, updated_hidden_layer_values):
        """ 
        Update the weights based on the delta weights calculated for Hidden Layer
        """
        updated_hidden_weights = []
        for i in range(len(current_hidden_weight)):
            temp = []
            current_hidden_weight_values = current_hidden_weight[i]
            updated_hidden_layer = updated_hidden_layer_values[i]
            for j in range(len(current_hidden_weight_values)):
                temp.append(updated_hidden_layer[j] + updated_hidden_layer[j])
            updated_hidden_weights.append(temp)
        return updated_hidden_weights

    def remove_bias(self, weight_vector):
        """
        Remove the bias from weight vector for gradient calculation
        """
        bias_filtered_weights = []
        for i in range(len(weight_vector)):
            temp = [0] * (len(weight_vector[i]) - 1)
            for j in range(len(weight_vector[i])):
                if j != 0:
                    temp[j - 1] = weight_vector[i][j]
            bias_filtered_weights.append(temp)
        return bias_filtered_weights

    def train(self, training_features, training_targets, validation_features, validation_targets, num_of_epochs=100):
        """ Train the network with the passed inputs, and it also used validation data along with to perform the cross
        validation testing on the dataset. A user can also specify maximum epochs for training """
        epoch_length = 0
        epoch_test_error = 0
        validation_error_values = []
        training_error_values = []
        epoch_val_error = 0
        val_inc_count = 0
        prev_epoch_v_error = 0
        epoch_output_layer_weights = []
        epoch_hidden_layer_weights = []
        while epoch_length < num_of_epochs:
            for row in range(len(training_features)):
                training_feature = training_features[row]
                training_target = training_targets[row]
                input_value = self.hidden_layer_bias + training_feature
                hidden_layer_output = self.activation_function(dot_product(input_value, self.hidden_layer_weights))
                sum_of_hidden_outputs = self.output_layer_bias + hidden_layer_output
                calculated_output = self.activation_function(
                    dot_product(sum_of_hidden_outputs, self.output_layer_weights))
                calculated_error = self.error(training_target, calculated_output)
                output_layer_derivative = self.derivative_activation_function(calculated_output)
                hidden_layer_derivative = self.derivative_activation_function(hidden_layer_output)
                output_layer_local_gradient = self.local_gradient_of_output_layer(output_layer_derivative,
                                                                                  calculated_error)
                bias_filtered_weight = self.remove_bias(self.output_layer_weights)
                hidden_layer_local_gradient = self.local_gradient_of_hidden_layer(hidden_layer_derivative,
                                                                                  output_layer_local_gradient,
                                                                                  bias_filtered_weight)
                output_layer_delta_weight = self.output_layer_delta_weight_update(output_layer_local_gradient,
                                                                                  sum_of_hidden_outputs)
                hidden_layer_delta_weight = self.hidden_layer_delta_weight_update(hidden_layer_local_gradient,
                                                                                  input_value)
                hidden_layer_updated_weight = self.updated_hidden_layer_weights(self.hidden_layer_weights,
                                                                                hidden_layer_delta_weight)
                self.hidden_layer_weights = hidden_layer_updated_weight
                output_layer_updated_weight = self.updated_output_layer_weights(self.output_layer_weights,
                                                                                output_layer_delta_weight)
                self.output_layer_weights = output_layer_updated_weight
                current_error = ((calculated_error[0]) ** 2 + (calculated_error[1]) ** 2) * 0.5
                epoch_test_error = epoch_test_error + current_error
            for val_row in range(len(validation_features)):
                """ During the cross validation we feed forward the input and make prediction but its not back
                    propagated.    
                """
                validation_feature = validation_features[val_row]
                validation_target = validation_targets[val_row]
                input_value = self.hidden_layer_bias + validation_feature
                hidden_layer_output = self.activation_function(dot_product(input_value, self.hidden_layer_weights))
                sum_of_hidden_outputs = self.output_layer_bias + hidden_layer_output
                calculated_output = self.activation_function(
                    dot_product(sum_of_hidden_outputs, self.output_layer_weights))
                calculated_error = self.error(validation_target, calculated_output)
                current_error = ((calculated_error[0]) ** 2 + (calculated_error[1]) ** 2) * 0.5
                epoch_val_error = epoch_val_error + current_error

            print("The " + str(epoch_length) + " epoch train error is " + str(
                math.sqrt(epoch_test_error / (len(training_features)))))
            print("The " + str(epoch_length) + " epoch validation error is " + str(
                math.sqrt(epoch_val_error / (len(validation_features)))))
            epoch_output_layer_weights.append(self.output_layer_weights)
            epoch_hidden_layer_weights.append(self.hidden_layer_weights)
            if epoch_val_error > prev_epoch_v_error:
                val_inc_count = val_inc_count + 1
            else:
                val_inc_count = 0
            prev_epoch_v_error = epoch_val_error
            epoch_val_error = math.sqrt(epoch_val_error / len(validation_features))
            epoch_test_error = math.sqrt(epoch_test_error / len(training_features))
            validation_error_values.append(epoch_val_error)
            training_error_values.append(epoch_test_error)
            epoch_test_error = 0
            epoch_val_error = 0
            if val_inc_count == 5:
                print("The neural Network Converged at epoch ", epoch_length - 4)
                print("The Optimal Hidden Layer Weights", epoch_hidden_layer_weights[epoch_length - 5])
                print("The Optimal Output Layer Weights", epoch_output_layer_weights[epoch_length - 5])
                with open("hidden_layer_weights.txt", 'w') as file:
                    file.write(str(epoch_hidden_layer_weights[epoch_length - 5]))
                with open("output_layer_weights.txt", 'w') as file:
                    file.write(str(epoch_output_layer_weights[epoch_length - 5]))
                self.output_layer_weights = epoch_output_layer_weights[epoch_length - 5]
                self.hidden_layer_weights = epoch_hidden_layer_weights[epoch_length - 5]
                break
            epoch_length = epoch_length + 1
            self.plot_curve(training_error_values, validation_error_values)

    def test(self, test_features, test_targets):
        """ During the testing phase all the test set data is forwarded and processed through feed forward phase. """
        test_error = 0
        for test_row in range(len(test_features)):
            input_value = self.hidden_layer_bias + test_features[test_row]
            hidden_layer_output = self.activation_function(dot_product(input_value, self.hidden_layer_weights))
            sum_of_hidden_outputs = self.output_layer_bias + hidden_layer_output
            calculated_output = self.activation_function(
                dot_product(sum_of_hidden_outputs, self.output_layer_weights))
            calculated_error = self.error(test_targets[test_row], calculated_output)
            test_cur = ((calculated_error[0]) ** 2 + (calculated_error[1]) ** 2) * 0.5
            test_error = test_error + test_cur
        print("The test set error is ", math.sqrt(test_error / (len(test_features))))

    def predict(self, current_position):
        """ predict the x and y velocities for the given X & Y distances """
        input_value = self.hidden_layer_bias + current_position
        hidden_layer_output = self.activation_function(dot_product(input_value, self.hidden_layer_weights))
        sum_of_hidden_outputs = self.output_layer_bias + hidden_layer_output
        calculated_output = self.activation_function(dot_product(sum_of_hidden_outputs, self.output_layer_weights))
        return calculated_output

    def error(self, training_target, calculated_output):
        """ Calculate Error based on actual and calculated. """

        for i in range(len(training_target)):
            self.error_value.append(training_target[i] - calculated_output[i])
        return self.error_value

    def plot_curve(self, training_error_values, validation_error_values):
        """
        Plot the curve.
        :return: It returns nothing but stores the image in current directory
        """
        plt.plot(training_error_values, label="train_error", color="blue")
        plt.plot(validation_error_values, label="validation_error", color="orange")
        plt.legend(loc="best")
        plt.savefig(f"{self.hidden_neuron}_score.png")
