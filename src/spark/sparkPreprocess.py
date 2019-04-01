#from .preprocess import preprocessor
from langdetect import detect
import justext
#from pyspark import SparkConf, SparkContext
#from elasticsearch import Elasticsearch
from sparkcc import CCSparkJob

class sparkPreprocess(CCSparkJob):
    """ Count server names sent in HTTP response header
            (WARC and WAT is allowed as input)"""

    name = "CountServers"
    fallback_server_name = '(no server in HTTP header)'
    ES_user = 'elastic'
    ES_password = 'elasticdevpass'
    ES_hosts = 'ec2-52-34-223-218.us-west-2.compute.amazonaws.com'
    #es = Elasticsearch ( host = ES_hosts , http_auth = (ES_user , ES_password) , verify_certs = False )
    DESIRED_LANGUAGE = 'en'
    ALLOWED_CONTENT_TYPES = {"application/http; msgtype=response" , "message/http"}

    @staticmethod
    def _ignoreWARCRecord(record) :
        ALLOWED_CONTENT_TYPES = {"application/http; msgtype=response" , "message/http"}
        # print ( record.rec_type )
        # print ( record.content_type )
        print ( record.length )

        if record.rec_type != 'response' :
            return True
        if record.length >= 10000000 :
            return True

        if not record.content_type or record.content_type not in ALLOWED_CONTENT_TYPES :
            return True

        return False



    @staticmethod
    def _removeBolerplate(html) :
        try :
            paragraphs = justext.justext ( html , justext.get_stoplist ( "English" ) )
            plainText = [ ]
            for paragraph in paragraphs :
                if not paragraph.is_boilerplate :
                    plainText.append ( paragraph.text )
            return "".join ( plainText )

        except :
            return ""



    @staticmethod
    def _isDesiredLanguage(self,plainText) :
        DESIRED_LANGUAGE='en'
        try :
            pre_language = detect ( plainText )
        except :
            return False
            # if pre_language != 'en' :
            #   print ( pre_language )
        return pre_language == DESIRED_LANGUAGE


    def process_record(self , record) :
        if not sparkPreprocess._ignoreWARCRecord ( record) :
            html = record.raw_stream.read ( )
            if html :
                plainText = sparkPreprocess._removeBolerplate ( html )
                url=record.rec_headers
                if plainText:
                    yield "yes"
                      #document={'url':url,'doc':plainText}
                      #return  [('key', document)]
        return



    def create_es_index(self,es):
        ES_INDEX='commonCrawl'
        ES_TYPE = 'plainDoc'
        es_mapping = {ES_TYPE : {"properties" : {"url" : {"type" : "text"} , "doc" : {"type" : "text"}}} }
        es_settings = {'number_of_shards' : 3 , 'number_of_replicas' : 1 , 'refresh_interval' : '1s' ,
                       'index.translog.flush_threshold_size' : '1gb'}
        response = es.indices.create ( index = ES_INDEX , body = {'settings' : es_settings , 'mappings' : es_mapping} )



    def run_job(self , sc , es) :
        ES_INDEX = 'commonCrawl'
        ES_TYPE = 'plainDoc'
        ES_RESOURCE = '/'.join ( [ ES_INDEX , ES_TYPE ] )
        es_conf = {'es.nodes' : 'ec2-52-34-223-218.us-west-2.compute.amazonaws.com' , 'es.resource' : ES_RESOURCE ,
                   'es.port' : '9200' , 'es.net.http.auth.user' : 'elastic' ,
                   'es.net.http.auth.pass' : 'elasticdevpass' ,
                   'es.nodes.wan.only' : 'true'}
        output = None
        if self.args.input != '' :
            input_data = sc.textFile (
                self.args.input ,
                minPartitions = self.args.num_input_partitions )
            output = input_data.mapPartitionsWithIndex ( self.process_warcs ).collect()
            for index,t in output:
                print(index)


        #if not es.indices.exists ( ES_INDEX ) :
         #       self.create_es_index ( es)


       #if output is not None :
           #output.saveAsNewAPIHadoopFile ( path = '-' , \
            #                         outputFormatClass = 'org.elasticsearch.hadoop.mr.EsOutputFormat' , \
             #                        keyClass = 'org.apache.hadoop.io.NullWritable' , \
              #                       valueClass = 'org.elasticsearch.hadoop.mr.LinkedMapWritable' , \
               #                      conf = es_conf )
        #self.log_aggregators ( sc )

if __name__ == "__main__":
    job = sparkPreprocess()
    job.run()
