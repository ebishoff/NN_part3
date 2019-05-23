import codecs
import os
import collections
from six.moves import cPickle
import numpy as np

"""
Implement a class object that should have the following functions:

1) object initialization:
This function should be able to take arguments of data directory, batch size and sequence length.
The initialization should be able to process data, load preprocessed data and create training and 
validation mini batches.

2)helper function to preprocess the text data:
This function should be able to do:
    a)read the txt input data using encoding='utf-8'
    b)
        b1)create self.char that is a tuple contains all unique character appeared in the txt input.
        b2)create self.vocab_size that is the number of unique character appeared in the txt input.
        b3)create self.vocab that is a dictionary that the key is every unique character and its value is a unique integer label.
    c)split training and validation data.
    d)save your self.char as pickle (pkl) file that you may use later.
    e)map all characters of training and validation data to their integer label and save as 'npy' files respectively.

3)helper function to load preprocessed data

4)helper functions to create training and validation mini batches


"""
            
class TextLoader():        
    def __init__(self, data_dir, batch_size, sequence_length,preprocess=False):
        self.batch_size=batch_size
        self.sequence_length=sequence_length
        if preprocess==False:
            data=self.preprocess_data(data_dir,batch_size,sequence_length)
        else:
            data=load_preprocessed_data(data_dir)
            
    def preprocess_data(self,data_dir,batch_size,sequence_length):
        with open(data_dir,'r',encoding='utf-8') as file:
            data=file.read()
        #chars    
        char=list(set(data))
        self.char=char
        
        #data size and vocab size
        data_size, vocab_size=len(data),len(char)
        self.data_size=data_size
        self.vocab_size=vocab_size
        
        #dictionary
        charkey_to_integerlabel={ch:i for i, ch in enumerate(char)}
        self.vocab=charkey_to_integerlabel
        
        v=[]
        for i in data:
            v.append(self.vocab[i])
        v=np.array(v)

        #split training and validation data
        new_data=hankel(v[0:sequence_length],v[sequence_length:])
        x=new_data[0:-1]
        y=new_data[-1]
        x=np.transpose(x)
        x_train, x_val, y_train, y_val=train_test_split(x,y,test_size=.1,shuffle=False,stratify=None)
        
        #save self.char as pickle(pkl) file 
        with open('char_shakespeare.pkl','wb') as fo:
            cPickle.dump(self.char,fo)
        
        np.save('x_train_integer_data.npy',x_train)
        np.save('x_val_integer_data.npy',x_val)
        np.save('y_train_integer_data.npy',y_train)
        np.save('y_val_integer_data.npy',y_val)
        
    def load_preprocessed_data(filename):
        with open(filename,'rb') as file:
            preprocessed_data=np.load(file)
        return preprocessed_data
        
    def mini_batch(x_data,y_data,batch_size):
#batch size is number of samples in each iteration 
        features, labels=utils.shuffle(x_data,y_data)
        data_size=np.shape(labels)[0]
        its=data_size//mini_batch_size
        i=0
        j=mini_batch_size
        k=0
        while k != its:
            mb_features=features[i:j,:]
            mb_labels=labels[i:j,:]
            i+=mini_batch_size
            j+=mini_batch_size
            k+=1
            yield mb_features, mb_labels

        
            



        
        