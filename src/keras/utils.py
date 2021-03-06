from keras.preprocessing import text, sequence
from keras.preprocessing.sequence import pad_sequences
import os
import csv
import numpy as np
import tqdm

def one_hot_y(y):
    out_train=[]
    for i in y:
        if i=='NoArgument':
            out_train.append([1,0,0])
        elif i=='Argument_for':
            out_train.append([0,1,0])
        elif i=='Argument_against':
            out_train.append([0,0,1])
    return out_train

def read_input(input_path):
    input_train=[]
    y_train=[]
    input_val=[]
    y_val=[]
    train_topic=[]
    topic_val=[]
    for file in os.listdir(input_path):
        if file!='school_uniforms.tsv':
            with open(input_path+file) as fd:
                rd = csv.reader(fd, delimiter="\t")
                next(rd)
                for raw in rd:
                    if  raw[-1] and raw[-2] and raw[-3] and raw[0]:
                        if raw[-1]=='train' or raw[-1]=='val':
                            input_train.append(raw[-3])
                            y_train.append(raw[-2])
                            train_topic.append(raw[0])
                        else:
                            topic_val.append(raw[0])
                            input_val.append(raw[-3])
                            y_val.append(raw[-2])

    out_train = one_hot_y ( y_train )
    out_test = one_hot_y ( y_val )
    out_train = np.asarray ( out_train )
    out_test = np.asarray ( out_test )
    return (input_train,train_topic,out_train,input_val,topic_val,out_test)





def text_precocess(input_train,input_val,topic_train,topic_val):
    max_features = 1000000
    sentenceLength = 150
    topic_sentenceLength = 100
    embed_size = 300
    tok = text.Tokenizer ( num_words = max_features , lower = True )
    tok.fit_on_texts ( list ( input_train ) + list ( input_val ) + list ( topic_train )+list(topic_val) )
    word_index = tok.word_index
    input_train = tok.texts_to_sequences ( input_train )
    input_val = tok.texts_to_sequences ( input_val )
    topic_train = tok.texts_to_sequences ( topic_train )
    topic_val = tok.texts_to_sequences ( topic_val )
    input_train = pad_sequences ( input_train , maxlen = sentenceLength )
    input_val = pad_sequences ( input_val , maxlen = sentenceLength )
    topic_train = pad_sequences ( topic_train , maxlen = sentenceLength )
    topic_val = pad_sequences ( topic_val , maxlen = sentenceLength )
    return (input_train,input_val,topic_train,topic_val,word_index)


def get_coefs(word , *arr) : return word , np.asarray ( arr , dtype = 'float32' )


EMB_PATH = '../flaskproject/crawl-300d-2M.vec'
def load_embeddings(embed_dir = EMB_PATH) :
    embedding_index = dict ( get_coefs ( *o.strip ( ).split ( " " ) ) for o in  open ( embed_dir ) )
    return embedding_index



def build_embedding_matrix(word_index , embeddings_index , max_features , lower = True , verbose = True) :
    embedding_matrix = np.zeros ( (max_features , 300) )
    for word , i in tqdm ( word_index.items ( ) , disable = not verbose ) :
        if lower :
            word = word.lower ( )
        if i >= max_features : continue
        try :
            embedding_vector = embeddings_index[ word ]
        except :
            embedding_vector = embeddings_index[ "unknown" ]
        if embedding_vector is not None :
            # words not found in embedding index will be all-zeros.
            embedding_matrix[ i ] = embedding_vector
    return embedding_matrix



def build_matrix(word_index , embeddings_index) :
    embedding_matrix = np.zeros ( (len ( word_index ) + 1 , 300) )
    for word , i in word_index.items ( ) :
        try :
            embedding_matrix[ i ] = embeddings_index[ word ]
        except :
            embedding_matrix[ i ] = embeddings_index[ "unknown" ]
    return embedding_matrix
