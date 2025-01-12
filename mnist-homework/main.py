from __future__ import print_function
import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import argparse
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import model.CNN as CNN
import model.Logistic as logisticRg
# 参数设定
EPOCH = 10             
BATCH_SIZE = 32
LR = 0.001       
DOWNLOAD_MNIST = True
methods = ['tree','logistic','CNN']
# 
if not(os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
    DOWNLOAD_MNIST = True
    
# mnist数据集下载
train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,                                     
    transform=torchvision.transforms.ToTensor(),    
    download=DOWNLOAD_MNIST,
)

# 准备训练数据
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
X_train = train_data.data.view(-1, 28*28).numpy() / 255.0  # Flatten the images and normalize
y_train = train_data.targets.numpy()

# 准备测试数据
test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)/255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_y = test_data.test_labels

# 
for method in methods:
    # 模型选择
    if method == 'CNN':
        model_method = CNN.CNN()
        optimizer = torch.optim.Adam(model_method.parameters(), lr=LR)   # optimize all logistic parameters
        loss_func = nn.CrossEntropyLoss()  
    elif method == 'logistic':
        model_method = logisticRg.logisticRg()
        optimizer = torch.optim.Adam(model_method.parameters(), lr=LR)   # optimize all logistic parameters
        loss_func = nn.CrossEntropyLoss()  
    elif method == 'tree':
        model_method = DecisionTreeClassifier()
    print(model_method) 

    # 训练&测试
    for epoch in range(EPOCH):
        if method != 'tree':
            for step, (b_x, b_y) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader
                if method == 'logistic': 
                    b_x = b_x.view(-1, 28*28)
                    
                output = model_method(b_x)[0]               # logistic output
                loss = loss_func(output, b_y)   # cross entropy loss
                optimizer.zero_grad()           # clear gradients for this training step
                loss.backward()                 # backpropagation, compute gradients
                optimizer.step()                # apply gradients

                if step % 50 == 0:
                    if method == 'logistic':
                        test_output, last_layer = model_method(test_x.view(-1,28*28))
                    else:
                        test_output, last_layer = model_method(test_x)
                    pred_y = torch.max(test_output, 1)[1].data.numpy()
                    accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
                    print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.4f' % accuracy)
        
        elif method == 'tree':
            model_method.fit(X_train, y_train)  # 训练决策树
            pred_y = model_method.predict(test_x.view(-1,28*28))  # 预测测试数据
            accuracy = accuracy_score(test_y, pred_y)  # 计算准确率
            print('Decision Tree | test accuracy: %.4f' % accuracy)
    