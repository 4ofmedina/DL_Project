import numpy as np
import pandas as pd
from dataset import *
from torch.utils.data import DataLoader
from networks import Extractor, Class_classifier, Domain_classifier
import torch.optim as optim
import torch.nn as nn
import os
from tensorboardX import SummaryWriter

# parameters
N_STEPS = 1
N_INPUTS = 300
N_NEURONS = 512
N_OUTPUTS = 1
N_EPOCHS = 1
BATCH_SIZE = 64
N_LAYERS = 2
gamma = 10
theta = 0.1

if __name__ == '__main__': 

    exp_name = 'dann'    
    model_name = 'model.pt'
    PATH = './models/' + exp_name

    if not os.path.exists('./models/'):
                os.mkdir('./models/')

    datadir = './data/processed_data/split_data.npz'
    data = np.load(datadir, allow_pickle=True)
    source = pd.DataFrame(data['source'], columns=['X', 'y', 'state'])
    target = pd.DataFrame(data['target'], columns=['X', 'y', 'state'])

    source_train = source[source['state']=='train'].reset_index(drop=True)
    source_val = source[source['state']=='val'].reset_index(drop=True)
    target_train = target[target['state']=='train'].reset_index(drop=True)
    target_val = target[target['state']=='val'].reset_index(drop=True)

    source_trainset = TrafficSignal(source_train, transform=ToTensor())
    source_trainloader = DataLoader(source_trainset, batch_size=256)
    target_trainset = TrafficSignal(target_train, transform=ToTensor())
    target_trainloader = DataLoader(target_trainset, batch_size=256)

    source_valset = TrafficSignal(source_val, transform=ToTensor())
    source_valloader = DataLoader(source_valset, batch_size=256)
    target_valset = TrafficSignal(target_val, transform=ToTensor())
    target_valloader = DataLoader(target_valset, batch_size=256)

    #device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    #model instance
    feature_extractor = Extractor()
    class_classifier = Class_classifier()
    domain_classifier = Domain_classifier()

    feature_extractor.to(device)
    class_classifier.to(device)
    domain_classifier.to(device)

    #criterion
    class_criterion = nn.NLLLoss()
    domain_criterion = nn.NLLLoss()

    # init optimizer
    optimizer = optim.SGD([{'params': feature_extractor.parameters()},
                            {'params': class_classifier.parameters()},
                            {'params': domain_classifier.parameters()}], lr= 0.001, momentum= 0.9)

    # Summary writers
    writer_train = SummaryWriter(PATH + '/train')
    writer_val = SummaryWriter(PATH + '/val')

    best_val_loss = np.inf
    for epoch in range(N_EPOCHS):
        for i, (sdata, tdata) in enumerate(zip(source_trainloader, target_trainloader)):

            train_running_loss = 0.0
            val_running_loss = 0.0

            feature_extractor.train()
            class_classifier.train()
            domain_classifier.train()

            # steps
            start_steps = epoch * len(source_trainloader)
            total_steps = N_EPOCHS * len(source_trainloader)

            # setup hyperparameters
            p = float(i + start_steps) / total_steps
            constant = 2. / (1. + np.exp(-gamma * p)) - 1

            # zero the parameter gradients
            optimizer.zero_grad()

            # get the inputs
            input1, label1 = sdata
            input2, label2 = tdata
            size = min((input1.shape[0], input2.shape[0]))
            input1, label1 = input1[0:size], label1[0:size]
            input2, label2 = input2[0:size], label2[0:size]

            input1, label1 = input1.to(device, dtype=torch.float), label1.to(device)
            input2, label2 = input2.to(device, dtype=torch.float), label2.to(device)

            # prepare domain labels
            source_labels = torch.zeros((input1.size()[0])).type(torch.LongTensor).to(device)
            target_labels = torch.ones((input2.size()[0])).type(torch.LongTensor).to(device)

            # Exctract features
            source_features = feature_extractor(input1)
            target_features = feature_extractor(input2)

            # Class classifiction
            source_preds = class_classifier(source_features)
            #source_preds = source_preds.view(1, -1)        

            # Class classification loss
            class_loss = class_criterion(source_preds, label1)

            # DOMAIN CLASSIFIER
            src_preds = domain_classifier(source_features, constant)
            src_preds = src_preds.squeeze()
            tgt_preds = domain_classifier(target_features, constant)
            tgt_preds = tgt_preds.squeeze()

            # compute the domain loss 
            tgt_loss = domain_criterion(tgt_preds, target_labels)
            src_loss = domain_criterion(src_preds, source_labels)
            domain_loss = tgt_loss + src_loss

            loss = class_loss + theta * domain_loss
            loss.backward()  # Does backpropagation and calculates gradients
            optimizer.step()  # Updates the weights accordingly

            if (i + 1) % 10 == 0:
                    print('[{}/{} ({:.0f}%)]'.format(i * len(input2), len(target_trainset), 100. * i / len(target_trainloader)))

            train_running_loss += loss.detach().item()

        feature_extractor.eval()
        n_batches_train = np.copy(i)

        # VALIDATION ROUND
        for i, (sdata, tdata) in enumerate(zip(source_valloader, target_valloader)):

            # get the inputs
            input1, label1 = sdata.astype('torch.int64')
            input2, label2 = tdata.astype('torch.int64')
            size = min((input1.shape[0], input2.shape[0]))
            input1, label1 = input1[0:size], label1[0:size]
            input2, label2 = input2[0:size], label2[0:size]

            input1, label1 = input1.to(device), label1.to(device)
            input2, label2 = input2.to(device), label2.to(device)

            # forward
            val_features = feature_extractor(input1)
            preds = class_classifier(val_features)

            val_loss = class_criterion(preds, label1)
            val_running_loss += val_loss.detach().item()

        n_batches_val = np.copy(i)

        print('Epoch:  {:3d} | Class Loss: {:.4f} | Domain Loss: {:.4f} | Val Loss: {:.4f} '.format(epoch, 
                                                class_loss, domain_loss, val_running_loss / n_batches_val))

        # write summaries
        writer_train.add_scalar('loss', train_running_loss / n_batches_train, epoch)
        writer_val.add_scalar('loss', val_running_loss / n_batches_val, epoch)

        # save model at minimum validation loss
        if (val_running_loss / i) < best_val_loss:
            print('Saving Model...')
            best_val_loss = val_running_loss / n_batches_val
            # save model
            if not os.path.exists(PATH):
                os.mkdir(PATH)

            #torch.save(lstm.state_dict(), os.path.join(PATH, model_name))
            torch.save({
                'epoch': epoch,
                'extractor_state_dict': feature_extractor.state_dict(),
                'class_classifier_state_dict': class_classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, os.path.join(PATH, 'model.pt'))

    print('Optimization finished!')