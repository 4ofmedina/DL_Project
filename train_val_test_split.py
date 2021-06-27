from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None 

def train_val_test_split(df, save, domain, size=0.15):
    df_train, df_val = train_test_split(df, test_size=size, random_state=42)
    df_train, df_test = train_test_split(df_train, test_size=size/(1-size), random_state=42)
    
    df_train['state'] = 'train'
    df_val['state'] = 'val'
    df_test['state'] = 'test'
    
    '''if save:
        np.savez('./data/processed_data/'+domain+'/train', source=df_train)
        np.savez('./data/processed_data/'+domain+'/val', source=df_val)
        np.savez('./data/processed_data/'+domain+'/test', source=df_test)'''
    
    data = pd.concat([df_train, df_val, df_test], ignore_index=True)
    
    return data

def split(save=True):
    print('Splitting data...\n')
    datadir = './data/processed_data/data.npz'
    data = np.load(datadir, allow_pickle=True)
    source = pd.DataFrame(data['source'], columns=['X', 'y'])
    target = pd.DataFrame(data['target'], columns=['X', 'y'])
    source = train_val_test_split(source, save, 'source')
    print('Source split!')
    target = train_val_test_split(target, save, 'target')
    print('Target split!')
    if save:
        #np.savez('./data/processed_data/source', source=source)
        #np.savez('./data/processed_data/target', target=target)
        np.savez('./data/processed_data/split_data', source=source, target=target)
    return source, target