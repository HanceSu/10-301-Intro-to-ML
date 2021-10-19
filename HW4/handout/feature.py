import sys
import math
import csv

if __name__ == '__main__':
    train_in = sys.argv[1]
    valid_in = sys.argv[2]
    test_in = sys.argv[3]
    dtn_in = sys.argv[4]
    fmt_train = sys.argv[5]
    fmt_valid = sys.argv[6]
    fmt_test = sys.argv[7]
    flag = int(sys.argv[8])

TRIM_THRESHOLD = 4

def create_dict(dtn_in):
    dtn = dict()
    with open(dtn_in) as lines:
        for line in lines:
            strs = line[0:-1].split()
            dtn[strs[0]] = int(strs[1])
    return dtn

def model_1(infile, dtn):
    fmt_reviews = list()
    with open(infile) as lines:
        for line in lines:
            line_dtn = dict()
            (rating, review) = line.split("\t")
            line_dtn['rating'] = int(rating)
            for word in review.split():
                if (word in dtn) and not (dtn[word] in line_dtn):
                    line_dtn[dtn[word]] = 1
            fmt_reviews.append(line_dtn)
    return fmt_reviews

def model_2(infile, dtn):
    fmt_reviews = list()
    with open(infile) as lines:
        for line in lines:
            line_dtn = dict()
            (rating, review) = line.split("\t")
            line_dtn['rating'] = int(rating)
            for word in review.split():
                if (word in dtn):
                    if not (dtn[word] in line_dtn):
                        line_dtn[dtn[word]] = 1
                    else:
                        line_dtn[dtn[word]] += 1
            for word in list(line_dtn)[1:]:
                if (line_dtn[word]) < TRIM_THRESHOLD:
                    line_dtn[word] = 1
                else:
                    line_dtn.pop(word, None)
            fmt_reviews.append(line_dtn)
    return fmt_reviews

def format(infile, dtn, flag):
    if flag == 1:
        return model_1(infile, dtn)
    elif flag == 2:
        return model_2(infile, dtn)

def write(outfile, fmt_list):
    with open(outfile, 'w+') as out:
        for i in range(len(fmt_list)):
            line_list = list(fmt_list[i])
            for j in range(len(line_list)):
                if (j == 0):
                    out.write(str(fmt_list[i]['rating']) + '\t')
                elif (j == len(line_list) - 1):
                    out.write(str(line_list[j]) + ':' + 
                              str(fmt_list[i][line_list[j]]))
                else:
                    out.write(str(line_list[j]) + ':' + 
                              str(fmt_list[i][line_list[j]]) + '\t')
            out.write('\n')
        out.close()

dtn = create_dict(dtn_in)
train_fmt_reviews = format(train_in, dtn, flag)
valid_fmt_reviews = format(valid_in, dtn, flag)
test_fmt_reviews = format(test_in, dtn, flag)
write(fmt_train, train_fmt_reviews)
write(fmt_valid, valid_fmt_reviews)
write(fmt_test, test_fmt_reviews)






