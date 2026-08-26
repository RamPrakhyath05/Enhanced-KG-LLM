from neo4j import GraphDatabase
from config.settings import (
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD,
    NEO4J_DATABASE,
)

class Neo4jClient:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    def close(self):
        self.driver.close()
    def run_query(self, query, parameters=None):
        with self.driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]
    def stream_query(self, query, parameters=None, batch_size=10000):
        with self.driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(query, parameters or {})
            batch = []
            for record in result:
                batch.append(record.data())
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
