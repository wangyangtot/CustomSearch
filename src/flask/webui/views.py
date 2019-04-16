from webui import app
from elasticsearch import Elasticsearch
from flask import render_template , request
from keras.models import Model , load_model
from nltk.tokenize import sent_tokenize
import nltk
import tqdm
import numpy as np
from keras.preprocessing import text , sequence
import pandas as pd
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers, layers
import gc

modelPath='webui/py_model.h5'
nltk.download ( 'punkt' )

EMB_PATH='webui/crawl-300d-2M.vec'

def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

def load_embeddings(embed_dir=EMB_PATH):
    embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in open(embed_dir))
    print('load successfully')
    return embedding_index

def build_embedding_matrix(word_index, embeddings_index, max_features, lower = True, verbose = True):
    embedding_matrix = np.zeros((max_features, 300))
    for word, i in tqdm(word_index.items(),disable = not verbose):
        if lower:
            word = word.lower()
        if i >= max_features: continue
        try:
                embedding_vector = embeddings_index[word]
        except:
                embedding_vector = embeddings_index["unknown"]
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

def build_matrix(word_index, embeddings_index):
    embedding_matrix = np.zeros((len(word_index) + 1,300))
    for word,i in  word_index.items():
        try:
            embedding_matrix[i] = embeddings_index[word]
        except:
            embedding_matrix[i] = embeddings_index["unknown"]
    return embedding_matrix

embeddings_index = load_embeddings()

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
        # self.kernel = self.add_weight(name='kernel',
        #                             shape=(sentence_shape[1], self.output_dim),
        #                            initializer='uniform',
        #                           trainable=True)
        self.W = self.add_weight ( (sentence_shape[ -1 ] , topic_input_shape[ -1 ]) ,
                                   initializer = self.init ,
                                   name = '{}_W'.format ( self.name ) ) ,
        # regularizer=self.W_regularizer,
        # constraint=self.W_constraint)
        self.features_dim = sentence_shape[ -1 ]

        super ( Attention , self ).build ( input_shape )
        # self.built = True



    def compute_mask(self , input , input_mask = None) :
        return None



    def call(self , arg , mask = None) :
        features_dim = self.features_dim
        step_dim = self.step_dim
        x , topic_x = arg
        # eij = K.reshape(K.dot(),
        # K.reshape(self.W, (features_dim, 1))), (-1, step_dim))
        # topic_x 1*220*300 x:?220*300
        # print(self.W)
        # print('self.W')
        # print(K.reshape(self.W, (features_dim, features_dim)))
        t = K.dot ( topic_x , K.reshape ( self.W , (features_dim , features_dim) ) )
        # print(t)
        e = K.batch_dot ( t , K.permute_dimensions ( x , (0 , 2 , 1) ) )
        print ( 'e' )
        print ( e )
        weighted_input = K.sum ( e , axis = 1 )
        weighted_input = K.sigmoid ( weighted_input )

        # tem=K.reshape(topic_x, (-1, features_dim))
        # 1*220*300*300*300*300*
        # print(tem)
        # re=K.dot(tem,K.reshape(self.W, (features_dim, features_dim)))
        # print('re')
        # print(re)
        # eij=K.reshape(K.dot(re,K.reshape(x,(features_dim,-1))),(-1, step_dim,step_dim))
        # print('eij.shape')
        # print(eij.shape)
        # if self.bias:
        #   eij += self.b

        # eij = K.tanh(eij)

        # a = K.exp(eij)

       # if mask is not None :
        #    a *= K.cast ( mask , K.floatx ( ) )

        # a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        weighted_input = K.expand_dims ( weighted_input )
        weighted_input = x * weighted_input

        print ( 'weighted_input' )
        print ( weighted_input )
        return weighted_input



    def compute_output_shape(self , input_shape) :
        return input_shape[ 0 ] , self.features_dim



    def get_config(self) :
        config = {
            'W_regularizer' : self.W_regularizer ,
            'W_constraint' : self.W_constraint ,
            'step_dim' : self.step_dim ,
            # 'features_dim': 0
        }
        base_config = super ( Attention , self ).get_config ( )
        return dict ( list ( base_config.items ( ) ) + list ( config.items ( ) ) )


ES_settings = {
    'ES_hosts' : 'ec2-52-34-223-218.us-west-2.compute.amazonaws.com' ,
    'ES_user' : 'elastic' ,
    'ES_password' : 'elasticdevpass' ,
    'ES_type' : 'plainDocs' ,
    'ES_index' : 'extractccs'
}
hosts = [ ES_settings[ 'ES_hosts' ] ]
es_client = Elasticsearch (
    hosts ,
    port = 9200 ,
    http_auth = (ES_settings[ 'ES_user' ] , ES_settings[ 'ES_password' ]) ,
    verify_certs = False ,
    sniff_on_start = True ,  # sniff before doing anything
    sniff_on_connection_fail = True ,  # refresh nodes after a node fails to respond
    sniffer_timeout = 60 ,  # and also every 60 seconds
    timeout = 15

)





def compare(no_Argument , Argument_for , Argument_against) :
    maxval = max ( no_Argument , Argument_for , Argument_against )
    if maxval == no_Argument :
        return 'no_Argument'
    elif maxval == Argument_for :
        return 'Argument_for'
    return 'Argument_against'






@app.route ( '/',methods=['GET','POST'] )
@app.route ( '/index',methods=['GET','POST'] )
def index( ) :
    if request.method == 'GET':
        return  render_template ( "index.html" )
    elif request.method == 'POST' :
        model = load_model ( modelPath , custom_objects = {'Attention' : Attention} )
        search_text=request.form['query']
        try :
            response = es_client.search ( index = ES_settings[ 'ES_index' ] , body = {
                "query" : {
                    "match" : {
                        "doc" : search_text
                    }
                },
                "size" : 100
            } )

        except :
            jsonrespons = [ ]
            return render_template ( "index.html" )
        if response[ 'timed_out' ] == True :
            jsonresponse = [ ]
            return render_template ( "index.html" )
        else :
            search_results = response[ 'hits' ][ 'hits' ]
            jsonresponse = [ ]
            urllist=[]
            for X in search_results :
                plaintext = X[ '_source' ][ 'doc' ]
                url=X[ '_source' ][ 'url' ]
                tok_sentence = sent_tokenize ( plaintext )
                jsonresponse += tok_sentence
                tok_sentence_lenght = len ( tok_sentence )
                urllist += tok_sentence_lenght * [url]

    max_features = 100000
    maxlen = 150
    embed_size = 300
    tok = text.Tokenizer ( num_words = max_features , lower = True )
    total_sentence = len ( jsonresponse )
    tok.fit_on_texts ( jsonresponse + [ search_text ] * total_sentence )
    word_index = tok.word_index
    X_test = tok.texts_to_sequences ( jsonresponse )
    topic_test = tok.texts_to_sequences ( [ search_text ] * total_sentence )
    X_test = sequence.pad_sequences ( X_test , maxlen = maxlen )
    topic_test = sequence.pad_sequences ( topic_test , maxlen = maxlen )
    del tok
    gc.collect ( )
    embedding_matrix = build_matrix ( word_index , embeddings_index )
    y_pred = model.predict ( [ X_test , topic_test ] , verbose = 1 , batch_size = 512 )
    K.clear_session ( )
    index = np.arange ( len ( y_pred ) )
    y_pred = np.c_[ index , y_pred ]
    ratings = pd.DataFrame (
        {'index' : y_pred[ : , 0 ] , 'no_Argument' : y_pred[ : , 1 ] , 'Argument_for' : y_pred[ : , 2 ] ,
         'Argument_against' : y_pred[ : , 3 ]} )


    ratings[ 'res' ] = ratings.apply (
        lambda x : compare ( x[ 'no_Argument' ] , x[ 'Argument_for' ] , x[ 'Argument_against' ] ) , axis = 1 )



    sortedforratings = ratings[ ratings[ 'res' ] == 'Argument_for' ].sort_values ( by = 'Argument_for' ,
                                                                                   ascending = False )
    sortedAgainstRatings = ratings[ ratings[ 'res' ] == 'Argument_against' ].sort_values ( by = 'Argument_against' ,
                                                                                       ascending = False )
    all_sentence=[]
    print(sortedforratings)
    for_dict_all={}
    agaist_dict_all={}
    for index, row in sortedAgainstRatings.iterrows():
        agaist_dict_all[urllist[(int ( row[ 'index' ] ))]]={'res':row['res'],'score':row['Argument_for'],'text':jsonresponse[ (int ( row[ 'index' ] )) ]}
    for index, row in sortedforratings.iterrows():
        for_dict_all[urllist[(int ( row[ 'index' ] ))]]={'res':row['res'],'score':row['Argument_against'],'text':jsonresponse[ (int ( row[ 'index' ] )) ]}
    return render_template ( "result.html" , for_output = for_dict_all,against_outout=agaist_dict_all)



