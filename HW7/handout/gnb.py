import sys
import math
import random
import csv
import matplotlib.pyplot as plt
import numpy as np

def interpret(infile):
    input_data = []
    expected = []
    with open(infile) as csv_infile:
        lines = csv.reader(csv_infile, delimiter = ",")
        next(lines)
        for line in lines:
            expected_label = line[-1]
            row_data = line[:-1]
            row_data = [float(i) for i in row_data]
            input_data.append(row_data)
            expected.append(expected_label)
    expected_binary = [0 if label == "tool" else 1 for label in expected]
    return np.array(input_data), expected, np.array(expected_binary)

def gaussian_params(input_data, expected_binary):
    building_vec = expected_binary
    tool_vec = [1 if num == 0 else 0 for num in building_vec]
    tool_count = sum(tool_vec)
    building_count = sum(building_vec)

    tool_means = (input_data.T).dot(tool_vec) / tool_count
    building_means = (input_data.T).dot(building_vec) / building_count
    tool_means = np.array([tool_means])
    building_means = np.array([building_means])
    means = np.concatenate((tool_means, building_means), axis = 0)

    desc_order = np.flip(np.argsort(np.abs(tool_means - building_means), 
                                    axis = None))

    tool_sqrs = np.square(input_data - tool_means)
    building_sqrs = np.square(input_data - building_means)
    tool_vars = (tool_sqrs.T).dot(tool_vec) / tool_count
    building_vars = (building_sqrs.T).dot(building_vec) / building_count
    tool_vars = np.array([tool_vars])
    building_vars = np.array([building_vars])
    variances = np.concatenate((tool_vars, building_vars), axis = 0)

    return desc_order, means, variances

def label_probs(expected_binary):
    prob_b = sum(expected_binary) / len(expected_binary)
    prob_t = 1 - prob_b
    return prob_t, prob_b

def log_prob(x_m, mean, variance):
    a = 1 / math.sqrt(2 * math.pi * variance)
    b = - (x_m - mean)**2 / (2 * variance)
    return math.log(a) + b

def predict_label(x, voxels, desc_order, means, variances, prob_label):
    log_sum = 0.0
    for i in range(voxels):
        m = desc_order[i]
        log_sum += log_prob(x[m], means[m], variances[m])
    return math.log(prob_label) + log_sum

def predict(input_data, voxels, desc_order, means, variances, expected_binary):
    predicted = []
    prob_t, prob_b = label_probs(expected_binary)
    for i in range(len(input_data)):
        x = input_data[i]
        prob_tool = predict_label(x, voxels, desc_order, 
                                  means[0], variances[0], prob_t)
        prob_building = predict_label(x, voxels, desc_order,
                                      means[1], variances[1], prob_b)
        if (prob_tool >= prob_building):
            predicted.append("tool")
        else:
            predicted.append("building")
    return predicted

def get_error(expected, predicted):
    count = 0
    for i in range(len(expected)):
        if expected[i] != predicted[i]:
            count += 1
    return count / len(expected) 

def write_pred(outfile, predicted):
    with open(outfile, 'w+') as out:
        for i in range(len(predicted)):
            out.write(str(predicted[i]) + '\n')
        out.close()

def write_metrics(outfile, train_error, test_error):
    with open(outfile, 'w+') as out:
        out.write('error(train): ' + str(train_error) + '\n')
        out.write('error(test): ' + str(test_error))
        out.close()

def main(args):
    train_in = args[1]
    test_in = args[2]
    train_out = args[3]
    test_out = args[4]
    metrics_out = args[5]
    num_voxels = int(args[6])
    return train_in, test_in, train_out, test_out, metrics_out, num_voxels

if __name__ == '__main__':
    train_in, test_in, train_out, test_out, metrics_out, voxels = main(sys.argv)
    train_input, train_expected, train_binary = interpret(train_in)
    test_input, test_expected, test_binary = interpret(test_in)
    '''
    test_accs = []
    for i in range(50, 21764, 200):
        desc_order, means, variances = gaussian_params(train_input, 
                                                       train_binary)
        test_pred = predict(test_input, i, desc_order,
                            means, variances, test_binary)
        test_acc = 1 - get_error(test_expected, test_pred)
        test_accs.append(test_acc)
    x = [i for i in range(50, 21764, 200)]
    plt.plot(x, test_accs)
    plt.xlabel('Top k Number of Features')
    plt.ylabel('Test Accuracy')
    plt.title('Test Accuracy v.s. Top k Number of Features Used for Prediction')
    plt.show()
                           
    desc_order, means, variances = gaussian_params(train_input, train_binary)
    train_pred = predict(train_input, voxels, desc_order,
                         means, variances, train_binary)
    test_pred = predict(test_input, voxels, desc_order, 
                        means, variances, test_binary)
    train_error = get_error(train_expected, train_pred)
    test_error = get_error(test_expected, test_pred)
    write_pred(train_out, train_pred)
    write_pred(test_out, test_pred)
    write_metrics(metrics_out, train_error, test_error)
    '''







