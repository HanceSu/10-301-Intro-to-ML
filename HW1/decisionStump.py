import sys
import math
import csv

if __name__ == '__main__':
	train_infile = sys.argv[1]
	test_infile = sys.argv[2]
	split_index = sys.argv[3]
	train_outfile = sys.argv[4]
	test_outfile = sys.argv[5]
	metrics_outfile = sys.argv[6]


# Reserve this function for the next assignment, I thought that I needed it
# for this assignment
def get_entropy(sample, pos, neg):
    count_pos = 0
    count_neg = 0
    for example in sample:
        if example == pos:
            count_pos += 1
        elif example == neg:
            count_neg += 1
    pos_prob = count_pos/(count_pos + count_neg)
    neg_prob = count_neg/(count_pos + count_neg)
    entropy = -(pos_prob*math.log2(pos_prob) + neg_prob*math.log2(neg_prob))
    return entropy

# Calculate the error rate given actual results and predicted results
def get_error(actual, predict):
	error_count = 0
	for i in range(len(actual)):
		if actual[i] != predict[i]:
			error_count += 1
	return error_count/len(actual)		

# Partition the labels into two lists based on the split attribute
# Decide the majority label of each branch
def partition_examples(attr, labels, p_attr, n_attr, p_label, n_label):
	p_p_count = 0
	p_n_count = 0
	n_p_count = 0
	n_n_count = 0
	p_pred = ''
	n_pred = ''
	for i in range(len(attr)):
		if (attr[i] == p_attr and labels[i] == p_label):
			p_p_count += 1
		elif (attr[i] == p_attr and labels[i] == n_label):
			p_n_count += 1
		elif (attr[i] == n_attr and labels[i] == p_label):
			n_p_count += 1
		elif (attr[i] == n_attr and labels[i] == n_label):
			n_n_count += 1		
	if (p_p_count >= p_n_count):
		p_pred = p_label
	else:
		p_pred = n_label
	if (n_p_count >= n_n_count):
		n_pred = p_label
	else:
		n_pred = n_label
	return p_pred, n_pred

# Using the majority label of each branch of the split attribute 
# to predict the final labels
def output_labels(attr, p_attr, p_pred, n_pred, labels_pred):
	for i in range(len(attr)):
		if (attr[i] == p_attr):
			labels_pred.append(p_pred)
		else:
			labels_pred.append(n_pred)

# Positive and negative examples of the split attribute
pos_attr = ''
neg_attr = ''
# Positive and negative lables
pos_label = ''
neg_label = ''
# The labels that the positive and negative examples of the attribute 
# should predict to
pos_pred = ''
neg_pred = ''
# Lists of all the examples in the attribute at split_index
train_split_attr = list()
test_split_attr = list()
# Lists of all the actual labels
train_labels = list()
test_labels = list()
# Lists of all the predicted labels
train_pred = list()
test_pred = list()

# Convert the tsv files into python interpretable lists
# Store all the attribute examples of the attribute at split_index into a list
# Obtain the positive and negative labels 
# Also obtain the positive and negative attribute labels
# Store all the actual labels into a list
with open(train_infile) as infile:
	reader = csv.reader(infile, delimiter = "\t")
	count = 0
	infile_len = 0
	for row in reader:
		if count == 0:
			infile_len = len(row)
		elif count == 1:
			train_split_attr.append(row[int(split_index)])
			train_labels.append(row[infile_len - 1])
			pos_label = row[infile_len - 1]
			pos_attr = row[int(split_index)]
		else:
			train_split_attr.append(row[int(split_index)])
			train_labels.append(row[infile_len - 1])
			if row[infile_len - 1] != pos_label:
				neg_label = row[infile_len - 1]
			if row[int(split_index)] != pos_attr:
				neg_attr = row[int(split_index)]
		count += 1

with open(test_infile) as infile:
	reader = csv.reader(infile, delimiter = "\t")
	count = 0
	infile_len = 0
	for row in reader:
		if count == 0:
			infile_len = len(row)
		elif count >= 1:
			test_split_attr.append(row[int(split_index)])
			test_labels.append(row[infile_len - 1])
		count += 1

pos_pred, neg_pred = partition_examples(train_split_attr, train_labels, 
                                        pos_attr, neg_attr,
                                        pos_label, neg_label)

output_labels(train_split_attr, pos_attr, pos_pred, neg_pred, train_pred)
output_labels(test_split_attr, pos_attr, pos_pred, neg_pred, test_pred)

train_error = get_error(train_labels, train_pred)
test_error = get_error(test_labels, test_pred)


# Write the predicted labels from this decision stump into output files
with open(train_outfile, 'w+') as outfile:
	for i in range(len(train_pred)):
		outfile.write(train_pred[i] + '\n')
	outfile.close()

with open(test_outfile, 'w+') as outfile:
	for i in range(len(test_pred)):
		outfile.write(test_pred[i] + '\n')
	outfile.close()

# Write the error metrics of this decision stump into output files
with open(metrics_outfile, 'w+') as outfile:
	outfile.write('error(train): ' + str(train_error) + '\n')
	outfile.write('error(test): ' + str(test_error))
	outfile.close()













