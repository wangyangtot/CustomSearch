from utils import read_input,text_precocess
import os
from keras.layers import Dense,Input,LSTM,Bidirectional,Activation,Conv1D,GRU
from keras.layers import Dropout,Embedding,GlobalMaxPooling1D, MaxPooling1D, Add, Flatten
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from keras.models import Model
from keras.optimizers import Adam



def build_model():
    maxlen = 150
    embed_size = 300
    max_features = 100000
    inp = Input ( shape = (maxlen ,) )
    x = Embedding ( max_features , embed_size ) ( inp )  # maxlen=200 as defined earlier
    x = Bidirectional ( LSTM ( 96 , return_sequences = True , dropout = 0.15 , recurrent_dropout = 0.15 ) ) ( x )
    x = Conv1D ( 64 , kernel_size = 3 , padding = "valid" , kernel_initializer = "glorot_uniform" ) ( x )
    avg_pool = GlobalAveragePooling1D ( ) ( x )
    max_pool = GlobalMaxPooling1D ( ) ( x )
    x = concatenate ( [ avg_pool , max_pool ] )
    preds = Dense ( 3 , activation = "sigmoid" ) ( x )
    print ( preds.shape )
    model = Model ( inp , preds )
    model.compile ( loss = 'binary_crossentropy' , optimizer = Adam ( lr = 1e-3 ) , metrics = [ 'accuracy' ] )
    return model
def train_model(model,input_train , out_train,input_val , output_val):
    batch_size = 128
    epochs = 7
    model.fit ( input_train , out_train , batch_size = batch_size , validation_data = (input_val , output_val) ,
                epochs = epochs , verbose = 1 )
    model.summary ( )
    model.save ( 'biLSTM.h5' )






if __name__ == "__main__" :

    EMB_PATH = 'crawl-300d-2M.vec'
    # Input data files are available in the "../input/" directory.
    input_path='input'
    sentenceLength = 150
    input_train , topic_train , out_train , input_val , topic_val , out_val=read_input ( input_path )
    (input_train , input_val , topic_train , topic_val,word_index)=text_precocess(input_train,input_val,topic_train,topic_val)

    model=build_model ( )
    train_model(model,input_train,out_train,input_val,out_val)


