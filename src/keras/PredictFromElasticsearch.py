from elasticsearch import Elasticsearch
from keras.models import Model,load_model
from nltk.tokenize import sent_tokenize
import nltk
import numpy as np
from tqdm import tqdm
from keras.preprocessing import text, sequence
import pandas as pd
from keras import backend as K
import keras as ks
nltk.download('punkt')
ES_settings={
    'ES_hosts':'ec2-52-34-223-218.us-west-2.compute.amazonaws.com',
'ES_user':'elastic',
'ES_password':'elasticdevpass',
    'ES_type':'plainDocs',
    'ES_index':'extractccs'
}




def connect_to_elasticsearch(ES_settings):
    hosts = [ ES_settings[ 'ES_hosts' ] ]
    es = Elasticsearch (
        hosts ,
        port = 9200 ,
        http_auth = (ES_settings[ 'ES_user' ] , ES_settings[ 'ES_password' ]) ,
        verify_certs = False ,
        sniff_on_start = True ,  # sniff before doing anything
        sniff_on_connection_fail = True ,  # refresh nodes after a node fails to respond
        sniffer_timeout = 60 ,  # and also every 60 seconds
        timeout = 15

    )
    ### search_text example
    search_text = 'self-driving'
    response = es.search ( index = ES_settings[ 'ES_index' ] , body = {
        "query" : {
            "match" : {
                "doc" : search_text
            }
        } ,
        "size" : 30

    }
                           )
    return response

def textProcessResult(response,maxlen,embed_size):
    if response[ 'timed_out' ] == True :
        jsonresponse = [ ]
    else :
        search_results = response[ 'hits' ][ 'hits' ]
        total_took = response[ 'took' ]
        jsonresponse = [ ]
        urllist = [ ]
        for result in search_results:
            plaintext=result['_source']['doc']
            tok_sentence=sent_tokenize(plaintext)
            jsonresponse+=tok_sentence
            num_sentence=len(tok_sentence)
            urllist+=num_sentence*[result['_source']['url'] ]
        max_features = 100000
        tok = text.Tokenizer ( num_words = max_features , lower = True )
        tok.fit_on_texts(jsonresponse)
        input_predict=tok.texts_to_sequences(jsonresponse)
        input_predict=sequence.pad_sequences(input_predict, maxlen=maxlen)
        return input_predict


def predic(model,input_predict):
        y_pred = model.predict(input_predict,batch_size=1024,verbose=1)
        index=np.arange(len(y_pred))
        y_pred=np.c_[ index,y_pred ]
        return y_pred
