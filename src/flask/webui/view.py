from webui import app
from elasticsearch import Elasticsearch
from flask import render_template , request
from keras.models import Model , load_model
from nltk.tokenize import sent_tokenize
from keras import backend as K

import gc
from keras.preprocessing import text, sequence
import numpy as np
import pandas as pd
import nltk
import os
from keras.layers import Dense,Input,LSTM,Bidirectional,Activation,Conv1D,GRU
from keras.callbacks import Callback
from keras.layers import Dropout,Embedding,GlobalMaxPooling1D, MaxPooling1D, Add, Flatten
from keras.preprocessing import text, sequence
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
nltk.download ( 'punkt' )

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

model = load_model ('webui/my_model.h5')
#model.summery()



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
        search_text=request.form['query']
        try :
            response = es_client.search ( index = ES_settings[ 'ES_index' ] , body = {
                "query" : {
                    "match" : {
                        "doc" : search_text
                    }
                },
                "size" : 20
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
            K.clear_session ( )
            max_features = 1000
            maxlen = 150
            embed_size = 300
            tok = text.Tokenizer ( num_words = max_features , lower = True )
            total_sentence = len ( jsonresponse )
            tok.fit_on_texts ( jsonresponse )
            #word_index = tok.word_index
            X_test = tok.texts_to_sequences ( jsonresponse )
            #topic_test = tok.texts_to_sequences ( [ search_text ] * total_sentence )
            X_test = sequence.pad_sequences ( X_test , maxlen = maxlen )
            #topic_test = sequence.pad_sequences ( topic_test , maxlen = maxlen )
            #del tok
            #gc.collect ( )
            #embedding_matrix = build_matrix ( word_index , embeddings_index )
            y_pred = model.predict ( X_test , verbose = 1  )
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
            print(sortedforratings)
            for_dict_all={}
            agaist_dict_all={}
            for index, row in sortedAgainstRatings.iterrows():
                agaist_dict_all[urllist[(int ( row[ 'index' ] ))]]={'res':row['res'],'text':jsonresponse[ (int ( row[ 'index' ] )) ]}
            for index, row in sortedforratings.iterrows():
                for_dict_all[urllist[(int ( row[ 'index' ] ))]]={'res':row['res'],'text':jsonresponse[ (int ( row[ 'index' ] )) ]}
            return render_template ( "result.html" , for_output = for_dict_all,against_outout=agaist_dict_all)


