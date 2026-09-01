from src.db.neo4j_client import Neo4jClient

def extract_entities(client: Neo4jClient, limit: int = None, batch_size: int = 10000):
    if limit and limit > 0:
        query = f"""
        MATCH (e:Entity)
        RETURN e.Entity_ID AS id, e.Entity_Name AS name,
               e.Entity_Aliases AS aliases, e.Entity_Description AS description
        LIMIT {limit}
        """
    else:
        query = """
        MATCH (e:Entity)
        RETURN e.Entity_ID AS id, e.Entity_Name AS name,
               e.Entity_Aliases AS aliases, e.Entity_Description AS description
        """

    entities = {}
    for batch in client.stream_query(query, batch_size=batch_size):
        for record in batch:
            entities[record["id"]] = record
    return entities


def extract_edges_for_entities(client: Neo4jClient, entity_ids: set, chunk_size: int = 2000):
    id_list = list(entity_ids)
    edges = []

    for i in range(0, len(id_list), chunk_size):
        chunk = id_list[i : i + chunk_size]
        query_chunk = """
        MATCH (a:Entity)-[r:RELATED_TO]->(b:Entity)
        WHERE a.Entity_ID IN $chunk_ids
        RETURN a.Entity_ID AS source, b.Entity_ID AS target,
               r.Relation_ID AS relation_id, r.Relation_Name AS relation_name
        """
        records = client.run_query(query_chunk, {"chunk_ids": chunk})
        for record in records:
            if record["target"] in entity_ids:
                edges.append((record["source"], record["target"], record.get("relation_name") or record.get("relation_id")))

    return edges


def extract_all_edges(client: Neo4jClient, batch_size: int = 50000):
    query = """
    MATCH (a:Entity)-[r:RELATED_TO]->(b:Entity)
    RETURN a.Entity_ID AS source, b.Entity_ID AS target,
           r.Relation_ID AS relation_id, r.Relation_Name AS relation_name
    """
    edges = []
    for batch in client.stream_query(query, batch_size=batch_size):
        for record in batch:
            edges.append((record["source"], record["target"], record.get("relation_name") or record.get("relation_id")))
    return edges
