from src.db.neo4j_client import Neo4jClient
from src.retrieval.entity_documents import (
    ENTITY_QUERY,
    entity_to_document,
)
from src.retrieval.dense_retriever import DenseRetriever


LIMIT = 10_000
BATCH_SIZE = 1_000

INDEX_PATH = "indexes/dense/entities_10k.pkl"


def main():

    documents = []

    query = ENTITY_QUERY + f"\nLIMIT {LIMIT}"

    with Neo4jClient() as db:

        for batch in db.stream_query(
            query,
            batch_size=BATCH_SIZE
        ):

            for entity in batch:
                documents.append(
                    entity_to_document(entity)
                )

    print(f"Loaded {len(documents)} documents")

    retriever = DenseRetriever()

    print("Building dense index...")

    retriever.build(documents)

    retriever.save(INDEX_PATH)

    print(f"Dense index saved to {INDEX_PATH}")


if __name__ == "__main__":
    main()
