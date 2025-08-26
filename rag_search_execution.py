from elasticsearch_dsl import Search, Q
from elasticsearch import Elasticsearch
from clip_processor import create_text_embedding
import os


index = os.getenv('ES_INDEX')


def rrf_search(host, api_key, index_name, lat, lon, distance, text_query, k=10,
               num_candidates=100):
    """
    Create an RRF search object bound to a specific index.

    Args:
        index_name (str): Name of the Elasticsearch index
        lat (float): Latitude for geo filtering
        lon (float): Longitude for geo filtering
        distance (int/str): Distance for geo filtering
        text_query (str): Text to search in description fields
        k (int): Number of top results for KNN search
        num_candidates (int): Number of candidates for KNN search

    Returns:
        Search: elasticsearch_dsl Search object ready to execute
    """

    embedding = create_text_embedding(text_query).tolist()
    # Create geo distance query
    geo_filter = Q('geo_distance',
                   distance=distance,
                   geolocation={'lat': lat, 'lon': lon})

    # Create text search queries
    text_queries = [
        Q('match', generated_description=text_query),
        Q('match', description=text_query)
    ]

    # Create boolean query for standard search
    standard_query = Q('bool', filter=[geo_filter], should=text_queries)

    # Create search object bound to index
    s = Search(index=index_name)
    s = s.source(["image_filename", "generated_description"])

    # Build RRF configuration
    retrievers = [
        # Standard retriever
        {
            "standard": {
                "query": standard_query.to_dict()
            }
        },
        # Text KNN retriever
        {
            "knn": {
                "filter": geo_filter.to_dict(),
                "field": "text_embedding",
                "query_vector": embedding,
                "k": k,
                "num_candidates": num_candidates
            }
        },
        # Image KNN retriever
        {
            "knn": {
                "filter": geo_filter.to_dict(),
                "field": "image_embedding",
                "query_vector": embedding,
                "k": k,
                "num_candidates": num_candidates
            }
        }
    ]

    # Apply RRF configuration
    s = s.extra(retriever={'rrf': {'retrievers': retrievers}}, size=3)

    es = Elasticsearch(hosts=host, api_key=api_key)

    results = s.using(es).execute()["hits"]["hits"]

    return results


def execute_rrf_search_dsl(es_client, search_obj):
    """
    Execute the RRF search using elasticsearch_dsl.

    Args:
        es_client: Elasticsearch client instance
        search_obj: Search object created with create_rrf_search_from_index

    Returns:
        Response: elasticsearch_dsl Response object
    """
    # Execute using the elasticsearch_dsl client integration
    response = search_obj.using(es_client).execute()
    return response

