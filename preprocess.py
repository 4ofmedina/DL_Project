#Imports
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import csv

def create_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)
        print('Created folder at path: {}'.format(path))
    return True

#if __name__ == '__main__':

def preprocess():
    print('Preprocessing data...\n')
    #define pahts
    raw_data_path = os.path.join(".", "data", "raw_data")
    processed_data_path = os.path.join(".", "data", "processed_data")
    r_source_path = os.path.join(raw_data_path, "source")
    r_target_path = os.path.join(raw_data_path, "target")
    p_source_path = os.path.join(processed_data_path, "source")
    p_target_path = os.path.join(processed_data_path, "target")
    
    #create folders
    create_folder(processed_data_path)
    create_folder(p_source_path)
    create_folder(p_target_path)

    #read source
    source_data = []
    source_labels = os.path.join(r_source_path, "train_labelling.txt")
    num_lines = sum(1 for line in open(source_labels, 'r'))
    print('Source lenght: {}'.format(num_lines))
    with open(source_labels, 'r') as f:
        count = 0

        for line in f:
            line = line.split(" ")   #['train/1.png', '21', '0\n']
            try:
                img = mpimg.imread(os.path.join(r_source_path, line[0]))
                label = line[1]
                source_data.append([img, label])
                count+=1
            except:
                pass  
            
            if count%(int(num_lines/5))==0:
                print('Source : {:.2f}%'.format(count/num_lines*100))
            #if count>199: break   
            #break #####comentar
    
    #write processed source     
    #np.savez(p_source_path+'/source', a=source_data)
    source = pd.DataFrame(source_data, columns=['X','y'])
    print('Source processed!\n')
    
    #read target
    target_data = []
    target_train = pd.read_csv(r_target_path+'/Train.csv')#.head(100)
    target_test = pd.read_csv(r_target_path+'/Test.csv')#.head(100)
    target = pd.concat([target_train,target_test])
    num_lines = len(target)
    print('Target lenght: {}'.format(num_lines))
    count = 0
    for path, lbl in zip(target['Path'], target['ClassId']):
        img=mpimg.imread(r_target_path+"/"+path)
        target_data.append([img, lbl])
        count+=1
        
        if count%(int(num_lines/5))==0:
            print('Target : {:.2f}%'.format(count/num_lines*100))
        
    #write processed target
    target = pd.DataFrame(target_data, columns=['X','y'])
    print('Target processed!\n')

    np.savez('./data/processed_data/data', target=target, source=source)

    return source, target