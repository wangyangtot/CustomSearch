from utils import read_input,text_precocess,load_embeddings,build_matrix

import keras.layers as L
from keras.layers import Dense,Input,LSTM,Bidirectional,Activation,Conv1D,GRU,Lambda
from keras.layers import Dropout,Embedding,GlobalMaxPooling1D, MaxPooling1D, Add, Flatten
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score,precision_score,recall_score,f1_score


class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            print (y_pred)
            roc_auc = roc_auc_score(self.y_val, y_pred)
            #precision=precision_score(self.y_val, y_pred)
            #f1=f1_score(self.y_val, y_pred)
            #recall=recall_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: {:d} - score: {:.6f}".format(epoch+1, roc_auc))
            #print ( "\n Precision - epoch: {:d} - score: {:.6f}".format ( epoch + 1 , precision ) )
            #print ( "\n F1 - epoch: {:d} - score: {:.6f}".format ( epoch + 1 , f1 ) )
            #print ( "\n recall - epoch: {:d} - score: {:.6f}".format ( epoch + 1 , recall ) )






def cosine_distance(vests) :
        x , y = vests
        x = K.l2_normalize ( x , axis = -1 )
        y = K.l2_normalize ( y , axis = -1 )
        y=K.expand_dims ( y,1 )
        mul= K.tf.multiply(x,y)
        res= K.sum (mul, axis = -1 , keepdims = True )
        return res

def cos_dist_output_shape(shapes) :
        shape1 , shape2 = shapes
        return (shape1[ 0 ] , shape1[1],1)
def topic_mean(x):
        x = K.mean ( x , axis = 1 )
        return x

def topic_mean_output_shape(input_shape) :
    assert len ( input_shape ) == 3  # only valid for 3D tensors
    return (input_shape[0],input_shape[-1])

def build_model(sentenceLength , word_index ,):
    maxlen = 150
    embed_size = 300
    max_features = 100000
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
                                          trainable = False
                                    )
    x = embedding_layer ( sequence_input )
    topic_x = topic_embedding_layer ( topic_sequence_input )
    print ( 'topic_x.shape' )
    print ( topic_x.shape )
    topic_mean_x=Lambda (topic_mean,output_shape = topic_mean_output_shape)(topic_x)
    print ( 'topic_mean_x.shape' )
    print(topic_mean_x.shape)

    distance = Lambda ( cosine_distance , output_shape = cos_dist_output_shape ) ( [ x , topic_mean_x ])
    print ( "distance.shape" )

    print ( distance.shape )
    x = concatenate ( [ x , distance ] )
    print("concatenate")
    print(x.shape)

    #dropout = 0.15 , recurrent_dropout = 0.15 )
    x = Bidirectional ( L.CuDNNLSTM  ( 96 , return_sequences = True )  ) ( x )
    x = Conv1D ( 64 , kernel_size = 3 , padding = "valid" , kernel_initializer = "glorot_uniform" ) ( x )
    avg_pool = GlobalAveragePooling1D ( ) ( x )
    max_pool = GlobalMaxPooling1D ( ) ( x )
    x = concatenate ( [ avg_pool , max_pool ] )
    preds = Dense ( 3 , activation = "sigmoid" ) ( x )
    print ( preds.shape )
    model = Model ( inputs = [ sequence_input , topic_sequence_input ] , outputs = preds )
    model.compile ( loss = 'binary_crossentropy' , optimizer = Adam ( lr = 1e-3 ) , metrics = [ 'accuracy' ] )
    return model



def train_model(model , input_train , topic_train , out_train , input_val , topic_val , out_val):
    batch_size = 256
    epochs = 7


    filepath = "bilstm_cos.bestModel.hdf5"
    checkpoint = ModelCheckpoint ( filepath , monitor = 'val_acc' , verbose = 1 ,save_weights_only=False, save_best_only = True , mode = 'max' )
    early = EarlyStopping ( monitor = "val_acc" , mode = "max" , patience = 5 )
    ra_val = RocAucEvaluation ( validation_data = ([input_val ,topic_val ], out_val) , interval = 1 )
    callbacks_list = [ ra_val , checkpoint , early ]

    model.fit ( [ input_train , topic_train ]  , out_train , batch_size = batch_size , validation_data = ([ input_val , topic_val ] , out_val) ,
                epochs = epochs, verbose = 1,callbacks = callbacks_list)
    model.summary ( )
    model.save ( 'biLSTM_Cos.h5' )






if __name__ == "__main__" :

    # Input data files are available in the "../input/" directory.
    input_path='input/'
    sentenceLength = 150
    input_train , topic_train , out_train , input_val , topic_val , out_val=read_input ( input_path )
    (input_train , input_val , topic_train , topic_val,word_index)=text_precocess(input_train,input_val,topic_train,topic_val)
    embeddings_index = load_embeddings ( )
    embedding_matrix = build_matrix ( word_index , embeddings_index )
    model=build_model (sentenceLength , word_index )
    train_model ( model , input_train , topic_train , out_train , input_val , topic_val , out_val )


