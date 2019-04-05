from flask_app import app
from elasticsearch import Elasticsearch
import jsonify
ES_settings={
    'ES_hosts':'ec2-52-34-223-218.us-west-2.compute.amazonaws.com',
'ES_user':'elastic',
'ES_password':'elasticdevpass',
    'ES_type':'plainDocs',
    'ES_index':'crawls'
}
hosts=[ES_settings['ES_hosts']]
es_client = Elasticsearch(
       	    		hosts,
            		port=9200,
	    		http_auth=(ES_settings['ES_user'], ES_settings['ES_password']),
	    		verify_certs=False,
            		sniff_on_start=True,    # sniff before doing anything
            		sniff_on_connection_fail=True,    # refresh nodes after a node fails to respond
            		sniffer_timeout=60, # and also every 60 seconds
            		timeout=15

)
results_size=20


@app.route('/')
@app.route('/index')
def index():
    """
    Direct to default home page
    """
    return "Welcome to  Argument Search Engine!!!"


@app.route('/search/')
def all_products():
    """
    Find all the popular products in the database
    :return: list of products matching the criteria
    """
    body = \
        {
            "size": results_size,
            "query": {
                "match_all": {}
            },
            "sort": {
                "total_reviews": {
                    "order": "desc"
                }
            }
        }
    response = es_client.search(index=ES_settings['index'], doc_type=ES_settings['ES_type'], body=body)
    return jsonify(response)
