
from utils import read_input,text_precocess,load_embeddings,build_matrix


import keras.layers as L
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Dense,Input,LSTM,Bidirectional,Activation,Conv1D,GRU,Lambda,concatenate
from keras import initializers , regularizers , constraints , optimizers , layers
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score,accuracy_score


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
        weighted_input=K.tf.multiply ( x , weighted_input )

        return weighted_input



    def compute_output_shape(self , input_shape) :
        return input_shape



    def get_config(self) :
        config = {
            'W_regularizer' : self.W_regularizer ,
            'W_constraint' : self.W_constraint ,
            'step_dim' : self.step_dim ,
        }
        base_config = super ( Attention , self ).get_config ( )
        return dict ( list ( base_config.items ( ) ) + list ( config.items ( ) ) )



def cosine_distance(vests) :
        x , y = vests
        x = K.l2_normalize ( x , axis = -1 )
        y = K.l2_normalize ( y , axis = -1 )
        y = K.expand_dims ( y , 1 )
        mul = K.tf.multiply ( x , y )
        res = K.sum ( mul , axis = -1 , keepdims = True )
        return res



def cos_dist_output_shape(shapes) :
    shape1 , shape2 = shapes
    return (shape1[ 0 ] , shape1[ 1 ] , 1)



def topic_mean(x) :
    x = K.mean ( x , axis = 1 )
    return x



def topic_mean_output_shape(input_shape) :
    assert len ( input_shape ) == 3  # only valid for 3D tensors
    return (input_shape[ 0 ] , input_shape[ -1 ])



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
    att_x = Attention ( sentenceLength ) ( [ x , topic_x ] )

    topic_mean_x=Lambda (topic_mean,output_shape = topic_mean_output_shape)(topic_x)

    distance = Lambda ( cosine_distance , output_shape = cos_dist_output_shape ) ( [ att_x , topic_mean_x ] )

    x = concatenate ( [ att_x , distance ] )


    # att=K.Dropout(0.15)(att)
    x = L.Bidirectional ( L.CuDNNLSTM ( 128 , return_sequences = True ) ) ( x )
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
        model.compile ( loss = 'binary_crossentropy' , optimizer = Adam ( 0.005 ) , metrics = [ 'accuracy' ] )
    return model



def train_model(model , input_train , topic_train , out_train , input_val , topic_val , out_val) :
    filepath = "inner_att-bilstm_cos_best.hdf5"
    checkpoint = ModelCheckpoint ( filepath , monitor = 'val_acc' , verbose = 1 , save_best_only = True , mode = 'max' )
    early = EarlyStopping ( monitor = "val_acc" , mode = "max" , patience = 5 )
    ra_val = RocAucEvaluation ( validation_data = ([input_val ,topic_val ], out_val) , interval = 1 )
    callbacks_list = [ ra_val , checkpoint , early ]
    epochs = 7
    batch_size = 521
    model.fit ( [ input_train , topic_train ] , out_train , epochs = epochs , batch_size = batch_size , verbose = 1 , \
                validation_data = ([ input_val , topic_val ] , out_val),callbacks = callbacks_list, )
    model.summary ( )
    #model.save ( 'inner_att_bilstm_cos.h5' )



if __name__ == "__main__" :

    # Input data files are available in the "../input/" directory.
    input_path = 'input/'
    sentenceLength = 150
    input_train , topic_train , out_train , input_val , topic_val , out_val = read_input ( input_path )
    (input_train , input_val , topic_train , topic_val , word_index) = text_precocess ( input_train , input_val ,
                                                                                        topic_train , topic_val )
    embeddings_index = load_embeddings ( )
    embedding_matrix = build_matrix ( word_index , embeddings_index )

    model = build_model ( sentenceLength , word_index , verbose = False , compile = True )
    train_model ( model , input_train , topic_train , out_train , input_val , topic_val , out_val )

### to reuse the model
###t_model = load_model('inner_att-bilstm_cos.h5',custom_objects={'Attention':Attention})



