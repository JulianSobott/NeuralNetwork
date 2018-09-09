import numpy as np
from matplotlib import pyplot as plt
import Activationfunctions


class NeuralNetwork:

    LEARNRATE = 0.1
    NUM_EPOCHS = 100
    USE_BIAS = True
    activation_function = Activationfunctions.linear

    def __init__(self, layers, in_values, out_target_values):
        self.layers = layers
        self.in_values = in_values
        self.out_target_values = out_target_values
        self.weights = []
        self.neuron_values = []
        #self.normalize_list(self.in_values)
        #self.normalize_list(self.out_target_values)
        self.init_weights()
        self.init_neurons()

    def normalize_list(self, value_list):
        min_val = min(min(val) for val in value_list)
        max_val = max(max(val) for val in value_list) + (min_val if min_val < 0 else -min_val)
        for idx_list in range(len(value_list)):
            for idx_out in range(len(value_list[idx_list])):
                value_list[idx_list][idx_out] = (value_list[idx_list][idx_out] + (min_val if min_val < 0 else -min_val))/max_val


    def init_weights(self):
        np.random.seed(2)
        for idx_weight_layer in range(len(self.layers) - 1):
            num_in_neurons = self.layers[idx_weight_layer] + (1 if self.USE_BIAS else 0)
            num_out_neurons = self.layers[idx_weight_layer + 1]
            weight_layer = np.random.rand(num_out_neurons, num_in_neurons)
            self.weights.append(weight_layer)
        print(self.weights)

    def init_neurons(self):
        for idx_neuron_layer in range(len(self.layers)):
            neuron_layer = [0.0 for idx_neuron in range(self.layers[idx_neuron_layer])]
            if self.USE_BIAS and idx_neuron_layer < len(self.layers) - 1:
                neuron_layer.append(1.0)
            self.neuron_values.append(neuron_layer)
        print("Init Neurons: " +str(self.neuron_values))

    def forwardpass(self, idx_data, in_data=None):
        if in_data is None:
            in_data = self.in_values[idx_data].copy()
        self.neuron_values[0] = in_data
        if self.USE_BIAS:
            self.neuron_values[0].append(1.0)
        for idx_layer in range(1, len(self.layers)):
            loc_in_values = np.array(self.neuron_values[idx_layer - 1])
            #loc_in_values = np.reshape(loc_in_values, )
            loc_weights = self.weights[idx_layer - 1]
            #print("in values: ")
            #print(loc_in_values)
            #print("weights")
            #print(loc_weights)
            loc_out_in = np.dot(loc_weights, loc_in_values)
            loc_out_out = self.activation_function(loc_out_in)

            if self.USE_BIAS:
                for idx_neuron in range(self.layers[idx_layer]):
                    self.neuron_values[idx_layer][idx_neuron] = loc_out_out[idx_neuron]
            else:
                self.neuron_values[idx_layer] = loc_out_out

            #print("output")
            #print(loc_out_out)
            #print("")
        print(str(self.neuron_values))

    def calculate_error(self, idx_data):
        target_out = self.out_target_values[idx_data]
        calculated_out = self.neuron_values[len(self.layers) - 1]
        error = sum([(target_out[idx_neuron] - calculated_out[idx_neuron])**2 for idx_neuron in range(len(target_out))])
        return error

    def backpropagation(self, idx_data):
        deltas = []

        def calculate_deltas():
            for idx_layer in reversed(range(1, len(self.layers))):
                deltas.append([])
                loc_errors = []
                loc_layer = self.layers[idx_layer]
                if idx_layer == len(self.layers) - 1:  # Output layer
                    for idx_neuron in range(loc_layer):
                        calculated_out_value = self.neuron_values[len(self.layers) - 1][idx_neuron]
                        target_out_value = self.out_target_values[idx_data][idx_neuron]
                        error = target_out_value - calculated_out_value
                        loc_errors.append(error)

                else:
                    for idx_neuron in range(loc_layer + (1 if self.USE_BIAS else 0)):
                        error = 0.0
                        for idx_following_neuron in range(self.layers[idx_layer + 1]):
                            # TODO Check if constellation of indices is correct
                            weight = self.weights[idx_layer][idx_following_neuron][idx_neuron]
                            following_delta = deltas[len(self.layers) - idx_layer - 2][idx_following_neuron]
                            error += weight * following_delta
                        loc_errors.append(error)

                for idx_neuron in range(loc_layer):
                    neuron_value = self.neuron_values[idx_layer][idx_neuron]
                    delta = loc_errors[idx_neuron] * self.activation_function(neuron_value, derivative=True)
                    deltas[len(deltas) - 1].append(delta)

        def adapt_weights():
            for idx_weights_layer in range(len(self.weights)):
                for idx_start_neuron in range(self.layers[idx_weights_layer]):
                    for idx_end_neuron in range(self.layers[idx_weights_layer + 1]):
                        weight_value = self.weights[idx_weights_layer][idx_end_neuron][idx_start_neuron]
                        delta_end_neuron = deltas[len(self.layers) - idx_weights_layer - 2][idx_end_neuron]
                        activation_value_start_neuron = self.neuron_values[idx_weights_layer][idx_start_neuron]
                        weight_value += self.LEARNRATE * activation_value_start_neuron * delta_end_neuron
                        self.weights[idx_weights_layer][idx_end_neuron][idx_start_neuron] = weight_value
                        pass

        calculate_deltas()
        adapt_weights()

    def learn(self):
        overall_errors = []
        epochs = []
        outputs = []
        for idx_epoch in range(self.NUM_EPOCHS):
            print("Epoch: " + str(idx_epoch))
            #print("Weights: " + str(self.weights))
            sum_error = 0.0
            for idx_data in range(len(self.in_values)):
                self.forwardpass(idx_data)
                sum_error += self.calculate_error(idx_data)
                outputs.append(self.neuron_values[len(self.layers) - 1])
                self.backpropagation(idx_data)
            overall_errors.append(sum_error)
            print("Errors: " +str(overall_errors[idx_epoch]))
            epochs.append(idx_epoch)


        self.plot_errors(overall_errors, epochs, outputs)

    def plot_errors(self, overall_errors, epochs, outputs):
        plt.figure(1)
        plt.title('Fehlerfunktion')
        plt.plot(epochs, overall_errors, color="black")
        #plt.figure(2)
        #plt.title('Berechnete Outputs')
        #plt.scatter([i for i in range(len(outputs))], outputs, color="green")
        plt.show()

    def test(self, test_value):
        print("-----------Testwert-----------------------")
        self.forwardpass(0, in_data=test_value)


def main():
    layers = [2, 4, 1]
    in_values = [[0, 0], [0, 1], [1, 0], [1, 1]]
    out_target_values = np.array([[0], [1], [1], [1]], dtype=float)
    #in_values = [[0.0], [0.1], [0.2], [0.3], [0.4], [0.5], [0.6], [0.7], [0.8]]
    #out_target_values = np.array([[0.0], [0.1], [0.2], [0.3], [0.4], [0.5], [0.6], [0.7], [0.8]], dtype=float)
    #in_values = [[0.5, 0.5], [0.5, 0.731], [0.731, 0.5], [0.731, 0.731]]
    #out_target_values = np.array([[0.5], [0.731], [0.731], [0.731]], dtype=float)
    neural_network = NeuralNetwork(layers, in_values, out_target_values)
    neural_network.learn()
    neural_network.test([0, 0.5])



if __name__ == "__main__":
    main()