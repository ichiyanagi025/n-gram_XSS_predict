import numpy as np
from sklearn import svm
import pickle
import csv
import time
from ngram import make_ngram
from topl import topl
from vector import vector


test_file = "./test/test_all.csv"
topL_file = "./topL.txt"
model_file = "./finalized_model.sav"
test_sample = []

t1 = time.time() 

with open(test_file,encoding="utf-8") as f:
    csv_reader = csv.reader(f)
    test_data_list = list(csv_reader)

topL_list = []

with open(topL_file,encoding="utf-8") as f:
    lines = f.read()
    for line in lines.split("\n"):
        top = line.split(",")
        topL_list.append(top)

print(topL_list)

for test in test_data_list:
    test[1] = test[1].split()
    test_ngram = make_ngram(test)
    test_vector = vector(test_ngram[1],topL_list)
    test_vector.insert(0, int(test[0]))
    test_sample.append(test_vector)


test_sample = np.array(test_sample)
test_label = test_sample[:, 0].astype(int)
test_data = test_sample[:, 1:10]

print(test_sample)
print(test_label)
print(test_data)


with open(model_file, mode='rb') as fp:
    model = pickle.load(fp)

print('Correct Label :', test_label)
print('Predicted Label:', model.predict(test_data))
print('Test Score: ', model.score(test_data, test_label))

t2 = time.time()
elapsed_time = t2-t1

print('elapsed time:',elapsed_time)