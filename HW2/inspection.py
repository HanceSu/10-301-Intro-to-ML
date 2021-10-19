import sys
import math
import csv

if __name__ == '__main__':
    data_infile = sys.argv[1]
    inspect_outfile = sys.argv[2]

# Count the number of positive and negative labels in a list
def count_labels(sample, p_label, n_label):
    count_pos = 0
    count_neg = 0
    for example in sample:
        if example == p_label:
            count_pos += 1
        elif example == n_label:
            count_neg += 1
    return count_pos, count_neg

# Calculate the entropy of the given list of binary labels
def get_entropy(sample, count_pos, count_neg):
    if (count_pos == 0 or count_neg == 0):
        return 0.0
    pos_prob = count_pos/(count_pos + count_neg)
    neg_prob = count_neg/(count_pos + count_neg)
    entropy = -(pos_prob*math.log2(pos_prob) + neg_prob*math.log2(neg_prob))
    return entropy

# Predict the final label using a majority vote
def predict_majority(sample, p_label, n_label, count_pos, count_neg):
    if (count_pos > count_neg):
        return p_label
    elif (count_pos < count_neg):
        return n_label
    elif (p_label < n_label):
        return n_label
    else:
        return p_label    

# Calculate the error rate given actual results and predicted results
def get_error(actual, predict):
    error_count = 0
    for i in range(len(actual)):
        if actual[i] != predict[i]:
            error_count += 1
    return error_count/len(actual)	

# Positive and negative lables
pos_label = ''
neg_label = ''
# Lists of all the actual labels
labels = list()
# Lists of all the predicted labels
pred = list()

# Convert the tsv files into python interpretable lists
# Obtain the positive and negative labels 
# Store all the actual labels into a list
with open(data_infile) as infile:
    reader = csv.reader(infile, delimiter = "\t")
    count = 0
    infile_len = 0
    for row in reader:
        if count == 0:
            infile_len = len(row)
        elif count == 1:
            labels.append(row[infile_len - 1])
            pos_label = row[infile_len - 1]
        else:
            labels.append(row[infile_len - 1])
            if row[infile_len - 1] != pos_label:
                neg_label = row[infile_len - 1]
        count += 1

pos, neg = count_labels(labels, pos_label, neg_label)
entropy = get_entropy(labels, pos, neg)
pred_label = predict_majority(labels, pos_label, neg_label, pos, neg)
pred = [pred_label] * len(labels)
error = get_error(labels, pred)

# Write the entropy and error of the input labels into the output file
with open(inspect_outfile, 'w+') as outfile:
    outfile.write('entropy: ' + str(entropy) + '\n')
    outfile.write('error: ' + str(error))
    outfile.close()

