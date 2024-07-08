from llama_index.storage.docstore.mongodb import MongoDocumentStore
import pymongo
import os


def store_docstore(documents, delete=False):
    uri = os.getenv("DATABASE_URI")
    docstore = MongoDocumentStore.from_uri(
        uri=uri, db_name="vector_db", node_collection_suffix='-node-bi-ojk', metadata_collection_suffix='-metadata-bi-ojk')
    mongodb_client = pymongo.MongoClient(uri)
    db = mongodb_client["vector_db"]

    if delete:
        for collection in db.list_collection_names():
            if collection.startswith("docstore") and collection.endswith("-bi-ojk"):
                db.drop_collection(collection)
                print(f"Deleted collection {collection}")

    docstore.add_documents(documents)
    print(f"Added {len(documents)} documents to the document store")

    return docstore


def load_docstore():
    uri = os.getenv("DATABASE_URI")
    return MongoDocumentStore.from_uri(
        uri=uri, db_name="vector_db", node_collection_suffix='-node-bi-ojk', metadata_collection_suffix='-metadata-bi-ojk'
    )
