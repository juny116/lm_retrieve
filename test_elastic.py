from elasticsearch import Elasticsearch

es = Elasticsearch() 

print(es.cluster.health())