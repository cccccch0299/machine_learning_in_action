from numpy import *
import numpy as np
from os import listdir
import operator
def knn_class(inX,dataset,labels,k):
    """
        :inX:输入向量
        :dataset:数据集
        :lables :对应标签
    """
    #距离计算
    size = dataset.shape[0]
    diffmat = tile(inX,(size,1)) -dataset
    sqrdiffmat = diffmat ** 2
    #按行为轴相加（横着累加）
    sqdistances = sqrdiffmat.sum(axis = 1)
    distances = sqdistances ** 0.5
    sortdistance = distances.argsort()
    classcount ={}

    #选择距离最小的k个点
    for i in range(k):
        votelable = labels[sortdistance[i]]
        classcount[votelable] = classcount.get(votelable,0) + 1#将对应标签投票数+1，若标签未出现过则默认1
    sortedclasscount = sorted(classcount.items(),key=operator.itemgetter(1), reverse=True)#第二个参数意为按元组中第二个元素进行排序即投票数
    #print(sortedclasscount)
    return sortedclasscount[0][0]

#二维图像转一维
def img2vector(filename):
    returnvector =zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        linestr = fr.readline()
        for j in range(32):
            returnvector[0,i*32+j] = int(linestr[j])
    return returnvector

#def handwriting():
hwlabels = []
trainfilelist = listdir('trainingDigits')
trainlen = len(trainfilelist)
trainvector = zeros((trainlen,1024))
for i in range(trainlen):
    trainvector[i, :] = img2vector('trainingDigits/%s' %trainfilelist[i])
    train_numclass = int(trainfilelist[i].split('.')[0].split('_')[0])
    hwlabels.append(train_numclass)
testfilelist = listdir('testDigits')
testlen = len(testfilelist)
errorcount = 0.0
for i in range(testlen):
    testvector = img2vector('testDigits/%s' % testfilelist[i])
    prediction = knn_class(testvector,trainvector,hwlabels,5)
    test_numclass = int(testfilelist[i].split('.')[0].split('_')[0])
    if(prediction != test_numclass):
        errorcount += 1.0
        print(f"样本 {test_numclass} 预测类别为: {prediction}")
    #print(f"样本 {test_numclass} 预测类别为: {prediction}")
print("\nthe total error is: %f" % errorcount)
print("\nthe total error rate is: %f" % (errorcount/float(testlen)))