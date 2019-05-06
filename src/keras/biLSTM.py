from utils import read_input,text_precocess

import keras.layers as L
from keras.layers import Dense,Input,LSTM,Bidirectional,Activation,Conv1D,GRU
from keras.layers import Dropout,Embedding,GlobalMaxPooling1D, MaxPooling1D, Add, Flatten
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score


class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: {:d} - score: {:.6f}".format(epoch+1, score))


def build_model():
    maxlen = 150
    embed_size = 300
    max_features = 100000
    inp = Input ( shape = (maxlen ,) )
    x = Embedding ( max_features , embed_size ) ( inp )  # maxlen=200 as defined earlier
    #dropout = 0.15 , recurrent_dropout = 0.15 )
    x = Bidirectional ( L.CuDNNLSTM  ( 96 , return_sequences = True )  ) ( x )
    x = Conv1D ( 64 , kernel_size = 3 , padding = "valid" , kernel_initializer = "glorot_uniform" ) ( x )
    avg_pool = GlobalAveragePooling1D ( ) ( x )
    max_pool = GlobalMaxPooling1D ( ) ( x )
    x = concatenate ( [ avg_pool , max_pool ] )
    preds = Dense ( 3 , activation = "sigmoid" ) ( x )
    model = Model ( inp , preds )
    model.compile ( loss = 'binary_crossentropy' , optimizer = Adam ( lr = 1e-3 ) , metrics = [ 'accuracy' ] )
    return model


def train_model(model,input_train , out_train,input_val , output_val):
    batch_size = 256
    epochs = 7

    # filepath="../input/best-model/best.hdf5"
    filepath = "bilstm.bestWeights.hdf5"
    checkpoint = ModelCheckpoint ( filepath , monitor = 'val_acc' , verbose = 1 , save_best_only = True , mode = 'max' )
    early = EarlyStopping ( monitor = "val_acc" , mode = "max" , patience = 5 )
    ra_val = RocAucEvaluation ( validation_data = (input_val , output_val) , interval = 1 )
    callbacks_list = [ ra_val , checkpoint , early ]

    model.fit ( input_train , out_train , batch_size = batch_size , validation_data = (input_val , output_val) ,
                epochs = epochs ,callbacks = callbacks_list, verbose = 1 )
    model.summary ( )
    model.save ( 'biLSTM.h5' )




if __name__ == "__main__" :

    # Input data files are available in the "../input/" directory.
    input_path='input/'
    sentenceLength = 150
    input_train , topic_train , out_train , input_val , topic_val , out_val=read_input ( input_path )
    (input_train , input_val , topic_train , topic_val,word_index)=text_precocess(input_train,input_val,topic_train,topic_val)

    model=build_model ( )
    train_model(model,input_train,out_train,input_val,out_val)


