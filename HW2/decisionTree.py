import sys
import math
import csv

if __name__ == '__main__':
    train_infile = sys.argv[1]
    test_infile = sys.argv[2]
    max_depth = int(sys.argv[3])
    #train_outfile = sys.argv[4]
    #test_outfile = sys.argv[5]
    metrics_outfile = sys.argv[4]

# Count the number of positive and negative labels in a list
def count_labels(sample, p_label, n_label):
    p_count = 0
    n_count = 0
    for label in sample:
        if label == p_label:
            p_count += 1
        elif label == n_label:
            n_count += 1
    return p_count, n_count

# Calculate the entropy based on the numbers of positive and negative labels
# S denotes entropy
def get_S(p_count, n_count):
    if (p_count == 0 or n_count == 0):
        return 0.0
    pos_prob = p_count/(p_count + n_count)
    neg_prob = n_count/(p_count + n_count)
    S = -(pos_prob*math.log2(pos_prob) + neg_prob*math.log2(neg_prob))
    return S

# Calculate the error rate given actual results and predicted results
def get_error(actual, predict):
    error_count = 0
    for i in range(len(actual)):
        if actual[i] != predict[i]:
            error_count += 1
    return error_count/len(actual)

class Node:
    # Positive and negative lables
    p_label = ''
    n_label = ''
    # Lists of all positive and negative examples of each attribute
    p_attrs = list()
    n_attrs = list()
    # Lists of all the attribute names
    attr_names = list()

    def __init__(self, labels, indexes, depth):
        self.left = None
        self.right = None
        self.attr = ''
        self.labels = labels
        self.p_count, self.n_count = count_labels(labels, 
                                                  self.p_label, self.n_label)
        self.indexes = indexes
        self.entropy = get_S(self.p_count, self.n_count)
        self.depth = depth

    # Partition the labels into two lists based on the split attribute
    # Also include the indexes of these labels in the original list of labels
    def partition_labels(self, attr, p_attr, n_attr):
        subset_1 = list()
        subset_2 = list()
        indexes_1 = list()
        indexes_2 = list()
        for i in range(len(self.labels)):
            if (attr[self.indexes[i]] == p_attr):
                subset_1.append(self.labels[i])
                indexes_1.append(self.indexes[i])
            elif (attr[self.indexes[i]] == n_attr):
                subset_2.append(self.labels[i])
                indexes_2.append(self.indexes[i])
        return subset_1, subset_2, indexes_1, indexes_2

    # Calculate the information gain after splitting on an attribute
    # IG denotes information gain
    def get_IG(self, subset_1, subset_2):
        sub_1_pos, sub_1_neg = count_labels(subset_1, 
                                            self.p_label, self.n_label)
        sub_1_S = get_S(sub_1_pos, sub_1_neg)
        sub_2_pos, sub_2_neg = count_labels(subset_2, 
                                            self.p_label, self.n_label)
        sub_2_S = get_S(sub_2_pos, sub_2_neg)
        ratio_1 = len(subset_1) / len(self.labels)
        ratio_2 = len(subset_2) / len(self.labels)
        IG = self.entropy - ratio_1*sub_1_S - ratio_2*sub_2_S
        return IG

    # Predict the final label using a majority vote
    def predict_majority(self):
        if (self.p_count > self.n_count):
            return self.p_label
        elif (self.p_count < self.n_count):
            return self.n_label
        elif (self.p_label < self.n_label):
            return self.n_label
        else:
            return self.p_label

    # Construct a decision stump by splitting on the attribute that gives
    # the highest information gain
    def construct_stump(self, attrs):
        max_IG = -1.0
        index = -1
        min_result = ()
        for i in range(len(attrs)):
            result = self.partition_labels(attrs[i],
                                           self.p_attrs[i], self.n_attrs[i])
            IG = self.get_IG(result[0], result[1])
            if (IG > max_IG):
                max_IG = IG
                index = i
                max_result = result
        self.attr = self.attr_names[index]
        self.left = Node(max_result[0], max_result[2], self.depth + 1)
        self.right = Node(max_result[1], max_result[3], self.depth + 1)

    # Recursively construct a decision tree
    def construct_tree(self, attrs, max_depth):
        if (max_depth == 0):
            return self
        elif (self.entropy == 0):
            return self
        elif (self.depth == max_depth):
            return self
        elif (self.depth == len(self.attr_names)):
            return self
        else:
            self.construct_stump(attrs)
            self.left.construct_tree(attrs, max_depth)
            self.right.construct_tree(attrs, max_depth)

    # Predict the labels of a given dataset using constructed decision tree
    def predict(self, attrs):
        pred_labels = list()
        for col in range(len(attrs[0])):
            current_node = self
            while (current_node.left != None or current_node.right != None):
                row = self.attr_names.index(current_node.attr)
                if (attrs[row][col] == self.p_attrs[row]):
                    current_node = current_node.left
                elif (attrs[row][col] == self.n_attrs[row]):
                    current_node = current_node.right
            pred_labels.append(current_node.predict_majority())
        return pred_labels

    # Pretty print the decision tree
    def __str__(self):
        prefix = "| "
        if (self.left == None and self.right == None):
            msg = (
                f"[{self.p_count} {self.p_label}/{self.n_count} {self.n_label}]"
            )
            return msg + "\n"
        else:
            index = self.attr_names.index(self.attr)
            msg_1 = (
                f"[{self.p_count} {self.p_label}/{self.n_count} {self.n_label}]"
            )
            msg_2 = prefix * (self.depth + 1)
            msg_3 = f"{self.attr} = {self.p_attrs[index]}: "
            msg_4 = self.left.__str__()
            msg_5 = f"{self.attr} = {self.n_attrs[index]}: "
            msg_6 = self.right.__str__()
            return msg_1 + "\n" + msg_2 + msg_3 + msg_4 + msg_2 + msg_5 + msg_6

# Positive and negative lables
pos_label = ''
neg_label = ''
# Lists of all positive and negative examples of each attribute
pos_attrs = list()
neg_attrs = list()
# Lists of all the attribute names
attr_names = list()
# 2D lists of all attribute data, each row represents one column of data, 
# and column indexes from the original file becomes row indexes in these lists
train_attrs = list()
test_attrs = list()
# Lists of all the actual labels
train_labels = list()
test_labels = list()

# Convert the tsv files into python interpretable lists
# Store all the attribute examples of the input files into a 2D lists
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
            attr_names = row[0:infile_len - 1]
        elif count == 1:
            train_labels.append(row[infile_len - 1])
            pos_label = row[infile_len - 1]
            for col in range(infile_len - 1):
                train_attrs.append([row[col]])
            pos_attrs = row[0:infile_len - 1]
            neg_attrs = ["" for i in range(infile_len - 1)]
        else:
            train_labels.append(row[infile_len - 1])
            if row[infile_len - 1] != pos_label:
                neg_label = row[infile_len - 1]
            for col in range(infile_len - 1):
                train_attrs[col].append(row[col])
                if row[col] != pos_attrs[col]:
                    neg_attrs[col] = row[col]
        count += 1

with open(test_infile) as infile:
    reader = csv.reader(infile, delimiter = "\t")
    count = 0
    infile_len = 0
    for row in reader:
        if count == 0:
            infile_len = len(row)
        elif count == 1:
            test_labels.append(row[infile_len - 1])
            for col in range(infile_len - 1):
                test_attrs.append([row[col]])
        else:
            test_labels.append(row[infile_len - 1])
            for col in range(infile_len - 1):
                test_attrs[col].append(row[col])
        count += 1

#Alter the class attributes of Node for easy access
Node.p_label = pos_label
Node.n_label = neg_label
Node.p_attrs = pos_attrs
Node.n_attrs = neg_attrs
Node.attr_names = attr_names

# Construct the decision tree using training data and print it out
decision_tree = Node(train_labels, range(len(train_labels)), 0)
decision_tree.construct_tree(train_attrs, max_depth)
print(decision_tree)

# Predict the labels of training and test data using constructed decision tree
train_pred = decision_tree.predict(train_attrs)
test_pred = decision_tree.predict(test_attrs)

# Obtain the error rate of both datasets
train_error = get_error(train_labels, train_pred)
test_error = get_error(test_labels, test_pred)

'''
# Write the predicted labels from this decision tree into output files
with open(train_outfile, 'w+') as outfile:
	for i in range(len(train_pred)):
		outfile.write(train_pred[i] + '\n')
	outfile.close()

with open(test_outfile, 'w+') as outfile:
	for i in range(len(test_pred)):
		outfile.write(test_pred[i] + '\n')
	outfile.close()
'''

# Write the error metrics of this decision stump into output files
with open(metrics_outfile, 'a+') as outfile:
	outfile.write('error(train): ' + str(train_error) + '\n')
	outfile.write('error(test): ' + str(test_error) + '\n')
	outfile.close()