from src.db.neo4j_client import Neo4jClient


def get_entity_by_id(client: Neo4jClient, entity_id: str):
    query = """
    MATCH (n:Entity)
    WHERE n.Entity_ID = $entity_id
    RETURN n.Entity_ID AS id, n.Entity_Name AS name,
           n.Entity_Description AS description, n.Entity_Aliases AS aliases
    """
    result = client.run_query(query, {"entity_id": entity_id})
    return result[0] if result else None


def search_entities_by_name(client: Neo4jClient, name_fragment: str, limit: int = 10):
    query = """
    MATCH (n:Entity)
    WHERE toLower(n.Entity_Name) CONTAINS toLower($name_fragment)
    RETURN n.Entity_ID AS id, n.Entity_Name AS name,
           n.Entity_Description AS description
    LIMIT $limit
    """
    return client.run_query(query, {"name_fragment": name_fragment, "limit": limit})


def get_neighbors(client: Neo4jClient, entity_id: str, limit: int = 10):
    query = """
    MATCH (a:Entity)-[r:RELATED_TO]->(b:Entity)
    WHERE a.Entity_ID = $entity_id
    RETURN b.Entity_ID AS neighbor_id, b.Entity_Name AS neighbor_name,
           r.Relation_ID AS relation_id, r.Relation_Name AS relation_name
    LIMIT $limit
    """
    return client.run_query(query, {"entity_id": entity_id, "limit": limit})
