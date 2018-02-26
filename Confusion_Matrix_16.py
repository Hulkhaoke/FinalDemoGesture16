from numpy import *
import numpy as np
import operator
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
quality = 16


def kNN_classify(cap_array, data_array, labels, k):
    difference = tile(cap_array, (600, 1, 1, 1)) - data_array
    # square
    sq_diff = difference ** 2
    # get the sum
    sq_distance = sq_diff.sum(axis=1)
    sq_distance = sq_distance.sum(axis=1)
    # square root
    distance = sq_distance ** 0.5
    dis = arange(600).reshape(600)
    for ii in range(600):
        dis[ii] = distance[ii]
    # sort from smallest to biggest and return the subscript
    sort_dist = dis.argsort()
    # store the vote of labels
    class_count = {}
    # print sort_dist
    for m in range(k):
        vote_label = labels[sort_dist[m]]
        class_count[vote_label] = class_count.get(vote_label, 0) + 1
    # sort the labels
    sort_count = sorted(class_count.iteritems(), key=operator.itemgetter(1), reverse=True)

    return sort_count[0][0]


def confusion_matrix_plot_matplotlib(y_truth, y_predict, cmap=plt.cm.Blues):
    cm = confusion_matrix(y_truth, y_predict)
    plt.matshow(cm, cmap=cmap)
    plt.colorbar()
    for x in range(len(cm)):
        for y in range(len(cm)):
            plt.annotate(cm[x, y], xy=(y, x), horizontalalignment='center', verticalalignment='center')
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7], ['clockwise', 'anticlockwise', 'frontpalm', 'backpalm', 'holdon', 'lighta', 'lightb', 'takeoff'], size="xx-small", rotation=20)
    plt.yticks([0, 1, 2, 3, 4, 5, 6, 7], ['clockwise', 'anticlockwise', 'frontpalm', 'backpalm', 'holdon', 'lighta', 'lightb', 'takeoff'], size="xx-small", rotation=70)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    #plt.show()
    plt.savefig('confusion_matrix', format='png', dpi=300)


gesture = ['clockwise', 'anticlockwise', 'frontpalm', 'backpalm', 'holdon', 'lighta', 'lightb', 'takeoff']

# read sample data
np_array = arange(quality*quality*600).reshape(quality*600, quality).reshape(600, quality, quality).reshape(600, quality, quality, 1)

for index_label in range(8):
    if index_label < 4:
        for index in range(50):
            np_array[index + index_label * 50] = np.load('data/' + gesture[index_label] + str(index + 1) + '.npy')
    else:
        for index in range(100):
            np_array[index + (index_label - 4) * 100 + 200] = np.load('data/' + gesture[index_label] + str(index + 1) + '.npy')

labels = [0 for k in range(600)]
for index_label in range(8):
    if index_label < 4:
        for index in range(50):
            labels[index + index_label*50] = gesture[index_label]
    else:
        for index in range(100):
            labels[index + (index_label - 4) * 100 + 200] = gesture[index_label]

pre_label = [0 for j in range(600)]
correct = 0
for i in range(600):
    result = kNN_classify(np_array[i], np_array, labels, 10)
    pre_label[i] = result

print confusion_matrix(labels, pre_label)
confusion_matrix_plot_matplotlib(labels, pre_label)
