from clip_processor import *
from elasticsearch import Elasticsearch
import os


def list_metadata_files(path):
    files = []
    for file in os.listdir(path):
        if file.endswith(".json"):
            files.append(file)

    return files


def upload_document(es: Elasticsearch, doc, index):
    res = es.index(document=doc, index=index)
    print(res)
    if res["result"] == "created":
        print(f"Uploaded file {doc['image_filename']}")
    else:
        print(f"Failed to upload file {doc['image_filename']}")



def index_logic():
    host = os.getenv('ES_HOST')
    api_key = os.getenv('ES_API_KEY')
    index = os.getenv('ES_INDEX')

    es = Elasticsearch(hosts=host, api_key=api_key)

    metadata_files = list_metadata_files('images_metadata/')

    for doc in metadata_files:
        data = add_embeddings(doc)
        upload_document(es, data, index)


index_logic()