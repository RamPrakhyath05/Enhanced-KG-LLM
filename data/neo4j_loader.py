from neo4j import GraphDatabase


class Neo4jLoader:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def _get_node_name(self, node):
        """Extract human-readable name from a Neo4j node."""
        # Wikidata schema uses Entity_Name
        for key in ("Entity_Name", "name", "label", "Label", "title"):
            val = node.get(key)
            if val:
                return str(val)
        # Fallback to entity ID or Neo4j element id
        return node.get("Entity_ID", str(node.element_id))

    def _parse_edge(self, record):
        """Parse a single Neo4j record into an edge tuple."""
        src = self._get_node_name(record["a"])
        dst = self._get_node_name(record["b"])
        props = record["props"] or {}

        # Use the descriptive Relation_Name as the relation type
        # instead of the generic Neo4j type (e.g. "RELATED_TO")
        rel = props.get("Relation_Name", record["rel"])

        return (src, rel, dst, props)

    def fetch_edges(self, limit=5000):
        """
        Returns property-graph edges in the format:
        (src_name, relation_name, dst_name, properties)
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
                edges.append(self._parse_edge(record))

        return edges

    def search_edges(self, keyword, limit=500):
        """
        Fetch edges where source or destination entity name or aliases
        contain the keyword (case-insensitive). Searches both Entity_Name
        and Entity_Aliases for better recall (e.g. 'Bill Gates' matches
        an entity named 'william h. gates' that has 'bill gates' as an alias).
        """
        query = """
        MATCH (a)-[r]->(b)
        WHERE toLower(a.Entity_Name) CONTAINS toLower($keyword)
           OR toLower(b.Entity_Name) CONTAINS toLower($keyword)
           OR ANY(alias IN a.Entity_Aliases WHERE toLower(alias) CONTAINS toLower($keyword))
           OR ANY(alias IN b.Entity_Aliases WHERE toLower(alias) CONTAINS toLower($keyword))
        RETURN a, type(r) AS rel, b, properties(r) AS props
        LIMIT $limit
        """
        edges = []
        with self.driver.session() as session:
            result = session.run(query, limit=limit, keyword=keyword)
            for record in result:
                edges.append(self._parse_edge(record))

        return edges

    # ------------------------------------------------------------------ #
    #  Generalized query-driven edge retrieval                           #
    # ------------------------------------------------------------------ #

    _STOP_WORDS = frozenset({
        "who", "what", "where", "when", "how", "why",
        "is", "are", "was", "were", "am",
        "the", "a", "an",
        "of", "in", "on", "at", "to", "for", "from", "by", "with",
        "and", "or", "but", "not", "no", "nor",
        "that", "this", "it", "its",
        "has", "have", "had", "do", "does", "did",
        "will", "would", "could", "should", "may", "might", "can", "shall",
        "be", "been", "being", "about", "which", "there",
        "tell", "me", "give", "list", "find", "show", "get",
        "describe", "explain", "name", "many", "much",
        "all", "any", "every", "kind", "type", "types",
        "i", "you", "he", "she", "we", "they", "my", "your",
    })

    @classmethod
    def _extract_search_terms(cls, query):
        """Extract candidate entity search terms from a natural-language query.

        Generates n-grams (longest first) from non-stop-words so that
        multi-word entity names like 'Bill Gates' or 'Nobel Prize' are
        tried before individual words.
        """
        words = query.lower().strip().rstrip("?!.").split()
        words = [w for w in words if w not in cls._STOP_WORDS and len(w) > 1]

        terms = []
        n = len(words)
        # Longest n-grams first (up to 4-grams) for better entity matching
        for size in range(min(n, 4), 0, -1):
            for i in range(n - size + 1):
                term = " ".join(words[i : i + size])
                if term not in terms:
                    terms.append(term)
        return terms

    def search_edges_by_query(self, query, limit_per_term=200, max_edges=1000):
        """Automatically extract search terms from a natural-language query
        and fetch relevant edges from Neo4j.

        Works for *any* query — no hardcoded entity names required.

        Examples:
            'who is the wife of Bill Gates?'
                → searches 'bill gates', 'wife', 'bill', 'gates'
            'scientist who won a physics award'
                → searches 'physics award', 'scientist', 'physics', 'award'
            'where was Albert Einstein born?'
                → searches 'albert einstein', 'born', 'albert', 'einstein'
        """
        terms = self._extract_search_terms(query)
        print(f"Search terms extracted: {terms}")

        all_edges = []
        seen = set()

        for term in terms:
            # Unigrams are often generic in large KGs, so cap them harder.
            adaptive_limit = min(limit_per_term, 250) if " " not in term else limit_per_term
            edges = self.search_edges(term, limit=adaptive_limit)
            for e in edges:
                key = (e[0], e[1], e[2])
                if key not in seen:
                    seen.add(key)
                    all_edges.append(e)
                if len(all_edges) >= max_edges:
                    break
            if len(all_edges) >= max_edges:
                break
            if edges:
                print(f"  '{term}' → {len(edges)} edges (limit={adaptive_limit})")

        return all_edges
