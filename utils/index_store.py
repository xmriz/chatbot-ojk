# Import basic libraries
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
)

from llama_index.core import load_index_from_storage

import pymongo
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.core import VectorStoreIndex, StorageContext
import os

from llama_index.core import get_response_synthesizer
from llama_index.core import DocumentSummaryIndex


def store_vector_index(embed_model, nodes=None, delete=False):
    uri = os.getenv("DATABASE_URI")
    # Initialize MongoDB client
    mongodb_client = pymongo.MongoClient(uri)

    # Database and collection names
    db_name = "vector_db"
    collection_name = "vector_index_bi_ojk"

    if delete:
        # Get the database and collection
        db = mongodb_client[db_name]
        collection = db[collection_name]

        # Delete all documents in the collection
        delete_result = collection.delete_many({})
        print(
            f"Deleted {delete_result.deleted_count} index from the collection.")

    # Initialize MongoDBAtlasVectorSearch
    vector_store_all = MongoDBAtlasVectorSearch(
        mongodb_client=mongodb_client, db_name=db_name, collection_name=collection_name, index_name='vector_index')

    # Initialize StorageContext
    storage_context_all = StorageContext.from_defaults(
        vector_store=vector_store_all)

    if not nodes:
        raise ValueError("Nodes must be provided to store the index.")
    else:
        index_all = VectorStoreIndex(
            nodes=nodes, show_progress=True, storage_context=storage_context_all, embed_model=embed_model)
        print("Storing the vector index completed.")

    return index_all


def load_vector_index():
    uri = os.getenv("DATABASE_URI")
    # Initialize MongoDB client
    mongodb_client = pymongo.MongoClient(uri)

    # Database and collection names
    db_name = "vector_db"
    collection_name = "index_bi_ojk"

    # Initialize MongoDBAtlasVectorSearch
    vector_store_all = MongoDBAtlasVectorSearch(
        mongodb_client=mongodb_client, db_name=db_name, collection_name=collection_name, index_name='vector_index')

    # Load the VectorStoreIndex
    vector_index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store_all)

    print("Loading the vector index completed.")

    return vector_index


def store_summary_index(llm, embed_model, nodes=None, delete=False):
    uri = os.getenv("DATABASE_URI")
    # # Initialize MongoDB client
    # mongodb_client = pymongo.MongoClient(uri)

    # # Database and collection names
    # db_name = "vector_db"
    # collection_name = "summary_index_bi_ojk"

    # if delete:
    #     # Get the database and collection
    #     db = mongodb_client[db_name]
    #     collection = db[collection_name]

    #     # Delete all documents in the collection
    #     delete_result = collection.delete_many({})
    #     print(
    #         f"Deleted {delete_result.deleted_count} index from the collection.")

    # # Initialize MongoDBAtlasVectorSearch
    # vector_store = MongoDBAtlasVectorSearch(
    #     mongodb_client=mongodb_client, db_name=db_name, collection_name=collection_name, index_name='vector_index')

    # # Initialize StorageContext
    # storage_context = StorageContext.from_defaults(
    #     vector_store=vector_store)

    

    if not nodes:
        raise ValueError("Nodes must be provided to store the index.")
    else:
        response_synthesizer = get_response_synthesizer(
            response_mode="tree_summarize", use_async=True
        )
        summary_index = DocumentSummaryIndex(
            nodes=nodes, show_progress=True, embed_model=embed_model, response_synthesizer=response_synthesizer, llm=llm)

        summary_index.storage_context.persist("index")
        print("Storing the summary index completed.")

    return summary_index


def load_summary_index():
    # uri = os.getenv("DATABASE_URI")
    # # Initialize MongoDB client
    # mongodb_client = pymongo.MongoClient(uri)

    # # Database and collection names
    # db_name = "vector_db"
    # collection_name = "summary_index_bi_ojk"

    # # Initialize MongoDBAtlasVectorSearch
    # vector_store = MongoDBAtlasVectorSearch(
    #     mongodb_client=mongodb_client, db_name=db_name, collection_name=collection_name, index_name='vector_index')

    # storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # # Load the VectorStoreIndex
    # summary_index = load_index_from_storage(storage_context=storage_context)

    storage_context = StorageContext.from_defaults(persist_dir="index")
    summary_index = load_index_from_storage(storage_context)

    print("Loading the summary index completed.")

    return summary_index
