import numpy as np
import pandas as pd
from dataset import *
from torch.utils.data import DataLoader
from networks import Extractor, Class_classifier, Domain_classifier
import torch.optim as optim
import torch.nn as nn
import os

if __name__ == '__main__': 

    exp_name = 'dann'    
    model_path = './models/' + exp_name + '/model.pt'

    datadir = './data/processed_data/split_data.npz'
    data = np.load(datadir, allow_pickle=True)
    source = pd.DataFrame(data['source'], columns=['X', 'y', 'state'])
    target = pd.DataFrame(data['target'], columns=['X', 'y', 'state'])

    source_test = source[source['state']=='test'].reset_index(drop=True)
    target_test = target[target['state']=='test'].reset_index(drop=True)

    source_testset = TrafficSignal(source_test, transform=ToTensor())
    source_testloader = DataLoader(source_testset, batch_size=256)
    target_testset = TrafficSignal(target_test, transform=ToTensor())
    target_testloader = DataLoader(target_testset, batch_size=256)
    
    #device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    #model instance
    feature_extractor = Extractor()
    class_classifier = Class_classifier()
    
    #load model parameters
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    feature_extractor.load_state_dict(checkpoint['extractor_state_dict'])
    class_classifier.load_state_dict(checkpoint['class_classifier_state_dict'])
    
    feature_extractor = feature_extractor.to(device)
    class_classifier = class_classifier.to(device)
    
    feature_extractor.eval()
    class_classifier.eval()
    
    source_correct = 0.0
    target_correct = 0.0
    
    #TEST
    for i, (sdata, tdata) in enumerate(zip(source_testloader, target_testloader)):

        # get the inputs
        input1, label1 = sdata
        input2, label2 = tdata
        size = min((input1.shape[0], input2.shape[0]))
        input1, label1 = input1[0:size], label1[0:size]
        input2, label2 = input2[0:size], label2[0:size]

        input1, label1 = input1.to(device, dtype=torch.float), label1.to(device)
        input2, label2 = input2.to(device, dtype=torch.float), label2.to(device)

        output1 = class_classifier(feature_extractor(input1))
        pred1 = output1.data.max(1, keepdim = True)[1]
        source_correct += pred1.eq(label1.data.view_as(pred1)).cpu().sum()
        
        output2 = class_classifier(feature_extractor(input2))
        pred2 = output2.data.max(1, keepdim=True)[1]
        target_correct += pred2.eq(label2.data.view_as(pred2)).cpu().sum()
        
    print('\nSource Accuracy: {}/{} ({:.4f}%)\nTarget Accuracy: {}/{} ({:.4f}%)\n'.
            format(
            source_correct, len(source_testset), 100. * float(source_correct) / len(source_testset),
            target_correct, len(target_testset), 100. * float(target_correct) / len(target_testset),
        ))