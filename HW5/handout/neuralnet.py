import sys
import math
import random
import csv
import numpy as np

if __name__ == '__main__':
    train_in = sys.argv[1]
    valid_in = sys.argv[2]
    train_out = sys.argv[3]
    valid_out = sys.argv[4]
    metrics_out = sys.argv[5]
    num_epoch = int(sys.argv[6])
    hidden_units = int(sys.argv[7])
    init_flag = int(sys.argv[8])
    learning_rate = float(sys.argv[9])

# Vital gloal variables
EP = num_epoch
HU = hidden_units
IF = init_flag
LR = learning_rate
CL = 10

# Read the csv file and return two np arrays containing input and out values
def interpret(infile):
    expected = []
    in_layers = []
    with open(infile) as csv_infile:
        lines = csv.reader(csv_infile, delimiter = ",")
        for line in lines:
            in_list = [1]
            for i in range(len(line)):
                if i == 0: expected.append(int(line[i]))
                else: in_list.append(int(line[i]))
            in_layers.append(in_list)
    return np.array(expected), np.array(in_layers)

# Initialize alpha and beta parameters based on the init flag
def init_params(h_units, init_flag, in_len):
    alpha = []
    beta = []
    for i in range(h_units):
        a_list = [0.0]
        if init_flag == 1:
            a_list += [random.uniform(-1, 1) for j in range(in_len)]
        elif init_flag == 2:
            a_list += [0.0 for j in range(in_len)]
        alpha.append(a_list)
    for i in range(CL):
        b_list = [0.0]
        if init_flag == 1:
            b_list += [random.uniform(-1, 1) for j in range(h_units)]
        elif init_flag == 2:
            b_list += [0.0 for j in range(h_units)]
        beta.append(b_list)
    return np.array(alpha), np.array(beta)

# Forward propogation with respect to one input vector x
def forprop(x_in, y_ept, alpha, beta):
    a = alpha.dot(x_in)
    hidden = 1 / (1 + np.exp(-1 * a))
    hidden = np.concatenate((np.array([1]), hidden))
    b = beta.dot(hidden)
    e = np.exp(b)
    y_pdt = e / np.sum(e)
    loss = -1 * math.log(y_pdt[y_ept])
    return a, hidden, b, y_pdt, loss

# Helps calculating the gradient of softmax
def gd_softmax(y_ept, y_pdt):
    gd_b = np.array([0.0 for i in range(CL)])
    for i in range(CL):
        if (i == y_ept):
            gd_b[i] = -1 + y_pdt[i]
        else:
            gd_b[i] = y_pdt[i]
    return gd_b

# Helps calculating the gradient of sigmoid
def gd_sigmoid(hidden, gd_z):
    z = np.delete(hidden, 0)
    gd_a = np.array([0.0 for i in range(HU)])
    for i in range(HU):
        gd_a[i] = z[i] * (1 - z[i]) * gd_z[i]
    return gd_a

# Forward propogation with respect to one input vector x
# Return the gradients with respect to alpha and beta
def backprop(x_in, y_ept, alpha, beta, a, hidden, b, y_pdt, loss):
    gd_b = gd_softmax(y_ept, y_pdt)
    gd_beta = gd_b.reshape(-1, 1).dot(hidden.reshape(1, -1))
    gd_hidden = (beta.T).dot(gd_b)
    gd_z = np.delete(gd_hidden, 0)
    gd_a = gd_sigmoid(hidden, gd_z)
    gd_alpha = gd_a.reshape(-1, 1).dot(x_in.reshape(1, -1))
    return gd_alpha, gd_beta

#predict the softmax layer and y label using current alpha and beta values
def predict(x_ins, y_epts, alpha, beta):
    y_pdts = []
    labels = []
    for i in range(len(x_ins)):
        y_pdt = forprop(x_ins[i], y_epts[i], alpha, beta)[3]
        y_pdts.append(y_pdt)
        labels.append(np.argmax(y_pdt))
    return np.array(y_pdts), np.array(labels)

# Perform SGD in order on all the input x vectors
def SGD(train_x_ins, valid_x_ins, train_y_epts, valid_y_epts, alpha, beta, 
        train_CEs, valid_CEs):
    for i in range(len(train_x_ins)):
        x_in = train_x_ins[i]
        y_ept = train_y_epts[i]
        a, hidden, b, y_pdt, loss = forprop(x_in, y_ept, alpha, beta)
        gd_alpha, gd_beta = backprop(x_in, y_ept, alpha, beta, 
                                     a, hidden, b, y_pdt, loss)
        alpha -= LR * gd_alpha
        beta -= LR * gd_beta
    train_y_pdts = predict(train_x_ins, train_y_epts, alpha, beta)[0]
    valid_y_pdts = predict(valid_x_ins, valid_y_epts, alpha, beta)[0]
    train_CEs.append(mean_cross_entropy(train_y_epts, train_y_pdts))
    valid_CEs.append(mean_cross_entropy(valid_y_epts, valid_y_pdts))

def train(train_x_ins, valid_x_ins, train_y_epts, valid_y_epts, alpha, beta, 
          train_CE_list, valid_CE_list):
    for i in range(EP):
        SGD(train_x_ins, valid_x_ins, train_y_epts, valid_y_epts, alpha, beta, 
            train_CE_list, valid_CE_list)

def mean_cross_entropy(expected, predicted):
    S_sum = 0.0
    for i in range(len(expected)):
        S_sum += math.log(predicted[i][expected[i]])
    return (-1 * S_sum) / len(expected)

def cal_error(expected, predicted):
    count = 0
    for i in range(len(expected)):
        if expected[i] != predicted[i]:
            count += 1
    return count / len(expected) 

def write_pred(outfile, labels):
    with open(outfile, 'w+') as out:
        for i in range(len(labels)):
            out.write(str(labels[i]) + '\n')
        out.close()

def write_metrics(outfile, train_CE_list, valid_CE_list, 
                  train_error, valid_error):
    with open(outfile, 'w+') as out:
        for i in range(len(train_CE_list)):
            out.write('epoch=' + str(i + 1) + ' crossentropy(train): ' + 
                      str(train_CE_list[i]) + '\n')
            out.write('epoch=' + str(i + 1) + ' crossentropy(validation): ' + 
                      str(valid_CE_list[i]) + '\n')
        out.write('error(train): ' + str(train_error) + '\n')
        out.write('error(validation): ' + str(valid_error) + '\n')
        out.close()

train_epts, train_ins = interpret(train_in)
valid_epts, valid_ins = interpret(valid_in)
alpha, beta = init_params(HU, IF, len(train_ins[0]) - 1)
train_CEs = []
valid_CEs = []
train(train_ins, valid_ins, train_epts, valid_epts, alpha, beta, 
      train_CEs, valid_CEs)
train_labels = predict(train_ins, train_epts, alpha, beta)[1]
valid_labels = predict(valid_ins, valid_epts, alpha, beta)[1]
train_error = cal_error(train_epts, train_labels)
valid_error = cal_error(valid_epts, valid_labels)
write_pred(train_out, train_labels)
write_pred(valid_out, valid_labels)
write_metrics(metrics_out, train_CEs, valid_CEs, 
              train_error, valid_error)