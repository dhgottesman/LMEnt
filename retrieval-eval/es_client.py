from elasticsearch import Elasticsearch

def get_esclient(scheme="https", host="132.67.130.202", port=9200):
    return Elasticsearch(
        f"{scheme}://{host}:{port}", 
        basic_auth=("elastic", "G*+2PQqsqZ2NCn5aCSoA"), 
        request_timeout=3000, 
        max_retries=10, 
        retry_on_timeout=True,
        verify_certs=False,
        ssl_show_warn=False
    )