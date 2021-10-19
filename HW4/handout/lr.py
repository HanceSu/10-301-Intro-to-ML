import sys
import math
import csv
import time

if __name__ == '__main__':
    fmt_train = sys.argv[1]
    fmt_valid = sys.argv[2]
    fmt_test = sys.argv[3]
    dtn_in = sys.argv[4]
    train_out = sys.argv[5]
    test_out = sys.argv[6]
    metrics_out = sys.argv[7]
    epochs = int(sys.argv[8])

LEARNING_RATE = [0.0001, 0.1, 0.5]

def dot(theta, x):
    product = theta[0] * x['b']
    for i in x:
        if (i != 'b'):
            product += theta[i + 1]
    return product

def J(theta, y, x):
    sum_prob = 0.0
    for i in range(len(y)):
        product = dot(theta, x[i])
        prob = -1 * y[i] * product + math.log(1 + math.exp(product))
        sum_prob += prob
    return sum_prob / len(y)

def SGD(theta, y, x, lr):
    for i in range(len(y)):
        d = dot(theta, x[i])
        sigmoid = 1 / (1 + math.exp(-1 * d))
        for j in x[i]:      
            if j == 'b':
                partial = (-1 / len(y)) * (y[i] - sigmoid)
                theta[0] = theta[0] - LEARNING_RATE[lr] * partial
            else:
                partial = (-1 / len(y)) * (y[i] - sigmoid)
                theta[j + 1] = theta[j + 1] - LEARNING_RATE[lr] * partial

def update(theta, y, x, lr):
    train_J = list()
    for i in range(epochs):
        train_prob = J(theta, y, x)
        train_J.append(train_prob)
        SGD(theta, y, x, lr)
    return train_J

def predict(theta, x):
    predict = list()
    for review in x:
        prob = 1 / (1 + math.exp(-1 * dot(theta, review)))
        predict.append(1) if prob >= 0.5 else predict.append(0)
    return predict

def get_error(expect, predict):
    count = 0
    for i in range(len(expect)):
        if expect[i] != predict[i]:
            count += 1
    return count / len(expect)        

def create_dict(dtn_in):
    dtn = dict()
    with open(dtn_in) as lines:
        for line in lines:
            strs = line[0:-1].split()
            dtn[strs[0]] = int(strs[1])
    return dtn

def interpret(infile):
    ratings = list()
    reviews = list()
    with open(infile) as lines:
        for line in lines:
            line_dtn = dict()
            features = line.split('\t')
            ratings.append(int(features[0]))
            features.pop(0)
            line_dtn['b'] = 1
            for feature in features:
                (i, x) = feature.split(':')
                line_dtn[int(i)] = int(x)
            reviews.append(line_dtn)
    return (ratings, reviews)

def train_theta(ratings, reviews, dtn):
    thetas = list()
    train_J_list = list()
    for i in range(3):
        theta = [0.0 for i in range(len(dtn) + 1)]
        train_J = update(theta, ratings, reviews, i)
        thetas.append(theta)
        train_J_list.append(train_J)
    return thetas, train_J_list

def write_pred(outfile, pred):
    with open(outfile, 'w+') as out:
        for i in range(len(pred)):
            out.write(str(pred[i]) + '\n')
        out.close()

def write_metrics(outfile, train_error, test_error):
    with open(outfile, 'w+') as out:
        out.write('error(train): ' + str(train_error) + '\n')
        out.write('error(test): ' + str(test_error) + '\n')
        out.close()

def write_J(outfile, train_J_list):
    with open(outfile, 'w+') as out:
        for i in range(len(train_J_list[0])):
            out.write(str(train_J_list[0][i]) + '\t')
            out.write(str(train_J_list[1][i]) + '\t')
            out.write(str(train_J_list[2][i]) + '\n')
        out.close()

dtn = create_dict(dtn_in)

train_ratings, train_reviews = interpret(fmt_train)
valid_ratings, valid_reviews = interpret(fmt_valid)
test_ratings, test_reviews = interpret(fmt_test)

thetas, train_J_list = train_theta(train_ratings, train_reviews, dtn)
train_pred = predict(thetas[1], train_reviews)
valid_pred = predict(thetas[1], valid_reviews)
test_pred = predict(thetas[1], test_reviews)

train_error = get_error(train_ratings, train_pred)
test_error = get_error(test_ratings, test_pred)

write_pred(train_out, train_pred)
write_pred(test_out, test_pred)
write_J(metrics_out, train_J_list)