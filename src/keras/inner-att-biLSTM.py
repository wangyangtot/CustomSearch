import numpy as np
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import csv
from keras.models import Model , load_model
from keras.preprocessing.sequence import pad_sequences
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from tqdm import tqdm
import os
from keras.preprocessing import text , sequence
import keras.layers as L
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers , regularizers , constraints , optimizers , layers



def one_hot_y(y) :
    out_train = [ ]
    for i in y :
        if i == 'NoArgument' :
            out_train.append ( [ 1 , 0 , 0 ] )
        elif i == 'Argument_for' :
            out_train.append ( [ 0 , 1 , 0 ] )
        elif i == 'Argument_against' :
            out_train.append ( [ 0 , 0 , 1 ] )
    return out_train



def read_input(input_path) :
    input_train = [ ]
    y_train = [ ]
    input_val = [ ]
    y_val = [ ]
    train_topic = [ ]
    topic_val = [ ]
    # print(os.listdir('input'))
    for file in os.listdir ( input_path ) :
        if file != 'school_uniforms.tsv' :
            with open ( input_path + file ) as fd :
                rd = csv.reader ( fd , delimiter = "\t" )
                next ( rd )
                for raw in rd :
                    if raw[ -1 ] and raw[ -2 ] and raw[ -3 ] and raw[ 0 ] :
                        if raw[ -1 ] == 'train' or raw[ -1 ] == 'val' :
                            input_train.append ( raw[ -3 ] )
                            y_train.append ( raw[ -2 ] )
                            train_topic.append ( raw[ 0 ] )
                        else :
                            topic_val.append ( raw[ 0 ] )
                            input_val.append ( raw[ -3 ] )
                            y_val.append ( raw[ -2 ] )

    out_train = one_hot_y ( y_train )
    out_test = one_hot_y ( y_val )
    out_train = np.asarray ( out_train )
    out_test = np.asarray ( out_test )
    return (input_train , train_topic , out_train , input_val , topic_val , out_test)



def text_precocess(input_train , input_val , topic_train , topic_val) :
    max_features = 1000000
    sentenceLength = 150
    topic_sentenceLength = 100
    embed_size = 300
    tok = text.Tokenizer ( num_words = max_features , lower = True )
    tok.fit_on_texts ( list ( input_train ) + list ( input_val ) + list ( topic_train ) + list ( topic_val ) )
    word_index = tok.word_index
    input_train = tok.texts_to_sequences ( input_train )
    input_val = tok.texts_to_sequences ( input_val )
    topic_train = tok.texts_to_sequences ( topic_train )
    topic_val = tok.texts_to_sequences ( topic_val )
    input_train = pad_sequences ( input_train , maxlen = sentenceLength )
    input_val = pad_sequences ( input_val , maxlen = sentenceLength )
    topic_train = pad_sequences ( topic_train , maxlen = sentenceLength )
    topic_val = pad_sequences ( topic_val , maxlen = sentenceLength )
    return (input_train , input_val , topic_train , topic_val , word_index)



def get_coefs(word , *arr) : return word , np.asarray ( arr , dtype = 'float32' )



def load_embeddings(embed_dir = EMB_PATH) :
    embedding_index = dict ( get_coefs ( *o.strip ( ).split ( " " ) ) for o in tqdm ( open ( embed_dir ) ) )
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



class Attention ( Layer ) :
    def __init__(self , step_dim ,
                 W_regularizer = None , b_regularizer = None ,
                 W_constraint = None , b_constraint = None ,
                 bias = True , **kwargs) :
        self.supports_masking = True
        self.init = initializers.get ( 'glorot_uniform' )

        self.W_regularizer = regularizers.get ( W_regularizer )
        # self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get ( W_constraint )
        # self.b_constraint = constraints.get(b_constraint)

        # self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super ( Attention , self ).__init__ ( **kwargs )



    def build(self , input_shape) :
        assert isinstance ( input_shape , list ) and len ( input_shape ) == 2
        print ( input_shape )
        sentence_shape , topic_input_shape = input_shape
        print ( 'sentence_shape' )
        print ( sentence_shape )
        print ( 'topic_input_shape' )
        print ( topic_input_shape )
        self.W = self.add_weight ( (sentence_shape[ -1 ] , topic_input_shape[ -1 ]) ,
                                   initializer = self.init ,
                                   name = '{}_W'.format ( self.name ) ) ,
        # regularizer=self.W_regularizer,
        # constraint=self.W_constraint)
        self.features_dim = sentence_shape[ -1 ]
        super ( Attention , self ).build ( input_shape )



    def compute_mask(self , input , input_mask = None) :
        return None



    def call(self , arg , mask = None) :
        features_dim = self.features_dim
        step_dim = self.step_dim
        x , topic_x = arg
        t = K.dot ( topic_x , K.reshape ( self.W , (features_dim , features_dim) ) )

        e = K.batch_dot ( t , K.permute_dimensions ( x , (0 , 2 , 1) ) )

        weighted_input = K.sum ( e , axis = 1 )
        weighted_input = K.sigmoid ( weighted_input )

        if mask is not None :
            weighted_input *= K.cast ( mask , K.floatx ( ) )

        weighted_input = K.expand_dims ( weighted_input )
        weighted_input = x * weighted_input

        return weighted_input



    def compute_output_shape(self , input_shape) :
        return input_shape[ 0 ] , self.features_dim



    def get_config(self) :
        config = {
            'W_regularizer' : self.W_regularizer ,
            'W_constraint' : self.W_constraint ,
            'step_dim' : self.step_dim ,
        }
        base_config = super ( Attention , self ).get_config ( )
        return dict ( list ( base_config.items ( ) ) + list ( config.items ( ) ) )



def build_model(sentenceLength , word_index , verbose = False , compile = True) :
    sequence_input = L.Input ( shape = (sentenceLength ,) , dtype = 'int32' )
    print ( sequence_input[ 0 ] )
    topic_sequence_input = L.Input ( shape = (sentenceLength ,) , dtype = 'int32' )
    print ( topic_sequence_input.shape )
    embedding_layer = L.Embedding ( len ( word_index ) + 1 ,
                                    300 ,
                                    weights = [ embedding_matrix ] ,
                                    input_length = sentenceLength ,
                                    trainable = False )
    print ( embedding_layer )
    topic_embedding_layer = L.Embedding ( len ( word_index ) + 1 ,
                                          300 ,
                                          weights = [ embedding_matrix ] ,
                                          input_length = sentenceLength ,
                                          trainable = False )
    x = embedding_layer ( sequence_input )

    topic_x = topic_embedding_layer ( topic_sequence_input )
    att = Attention ( sentenceLength ) ( [ x , topic_x ] )

    print ( att.shape )
    print ( 'x.shape' )
    print ( x.shape )
    # att=K.Dropout(0.15)(att)
    x = L.Bidirectional ( L.CuDNNLSTM ( 128 , return_sequences = True ) ) ( att )
    print ( x.shape )

    avg_pool1 = L.GlobalAveragePooling1D ( ) ( x )
    max_pool1 = L.GlobalMaxPooling1D ( ) ( x )

    x = L.concatenate ( [ avg_pool1 , max_pool1 ] )
    print ( x )

    preds = L.Dense ( 3 , activation = 'sigmoid' ) ( x )

    print ( preds )
    model = Model ( inputs = [ sequence_input , topic_sequence_input ] , outputs = preds )
    if verbose :
        model.summary ( )
    if compile :
        model.compile ( loss = 'binary_crossentropy' , optimizer = Adam ( 0.005 ) , metrics = [ 'acc' ] )
    return model



def train_model(model , input_train , topic_train , out_train , input_val , topic_val , out_val) :
    epochs = 10
    batch_size = 521
    model.fit ( [ input_train , topic_train ] , out_train , epochs = epochs , batch_size = batch_size , verbose = 1 , \
                validation_data = ([ input_val , topic_val ] , out_val) )
    model.summary ( )
    model.save ( 'inner_att-bilstm.h5' )



if __name__ == "__main__" :
    EMB_PATH = 'crawl-300d-2M.vec'
    # Input data files are available in the "../input/" directory.
    input_path = 'input'
    sentenceLength = 150
    input_train , topic_train , out_train , input_val , topic_val , out_val = read_input ( input_path )
    (input_train , input_val , topic_train , topic_val , word_index) = text_precocess ( input_train , input_val ,
                                                                                        topic_train , topic_val )
    embeddings_index = load_embeddings ( )
    embedding_matrix = build_matrix ( word_index , embeddings_index )

    model = build_model ( sentenceLength , word_index , verbose = False , compile = True )
    train_model ( model , input_train , topic_train , out_train , input_val , topic_val , out_val )

### to reuse the model
###t_model = load_model('inner_att-bilstm.h5',custom_objects={'Attention':Attention})



