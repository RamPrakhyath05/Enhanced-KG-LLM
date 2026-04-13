from neo4j import GraphDatabase

class Neo4jLoader:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def fetch_edges(self, limit=200):
        """
        Returns property-graph edges in the format:
        (src, relation, dst, properties)
        """
        query = """
        MATCH (a)-[r]->(b)
        RETURN a, type(r) AS rel, b, properties(r) AS props
        LIMIT $limit
        """
        edges = []
        with self.driver.session() as session:
            result = session.run(query, limit=limit)
            for record in result:
                src = record["a"].get("name", str(record["a"].id))
                dst = record["b"].get("name", str(record["b"].id))
                rel = record["rel"]
                props = record["props"]
                edges.append((src, rel, dst, props))

        return edges
