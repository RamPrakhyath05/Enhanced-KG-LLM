"""
Script to generate publication-grade UML Class Diagrams as SVG and convert to PNG via macOS sips.
Generates:
1. diagrams/master_class_diagram.png
2. diagrams/sublevel_1_preprocessing_db.png
3. diagrams/sublevel_2_retrieval_routing.png
4. diagrams/sublevel_3_traversal_evidence.png
5. diagrams/sublevel_4_pipeline_generation.png
"""

import os
import subprocess

DIAGRAMS_DIR = os.path.abspath("diagrams")
os.makedirs(DIAGRAMS_DIR, exist_ok=True)

def escape_xml(s: str) -> str:
    return (s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;")
             .replace('"', "&quot;"))

class UMLBox:
    def __init__(self, name, stereotype="", attrs=None, methods=None, x=0, y=0, w=320, h=200, color="#1e3a8a"):
        self.name = name
        self.stereotype = stereotype
        self.attrs = attrs or []
        self.methods = methods or []
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.color = color

    def render_svg(self) -> str:
        header_h = 42 if self.stereotype else 32
        # calculate attribute compartment height
        attr_h = max(24, len(self.attrs) * 16 + 12)
        method_h = max(24, len(self.methods) * 16 + 12)
        total_h = header_h + attr_h + method_h
        self.h = total_h

        svg = []
        # Main shadow and box
        svg.append(f'<filter id="shadow_{id(self)}" x="-5%" y="-5%" width="115%" height="115%"><feDropShadow dx="3" dy="4" stdDeviation="4" flood-color="#0f172a" flood-opacity="0.12"/></filter>')
        svg.append(f'<rect x="{self.x}" y="{self.y}" width="{self.w}" height="{total_h}" rx="6" fill="#ffffff" stroke="#94a3b8" stroke-width="1.5" filter="url(#shadow_{id(self)})"/>')
        
        # Header rect
        svg.append(f'<path d="M {self.x} {self.y+6} Q {self.x} {self.y} {self.x+6} {self.y} L {self.x+self.w-6} {self.y} Q {self.x+self.w} {self.y} {self.x+self.w} {self.y+6} L {self.x+self.w} {self.y+header_h} L {self.x} {self.y+header_h} Z" fill="{self.color}"/>')
        
        # Header text
        if self.stereotype:
            svg.append(f'<text x="{self.x + self.w/2}" y="{self.y + 16}" fill="#cbd5e1" font-family="Helvetica, Arial, sans-serif" font-size="11" font-style="italic" text-anchor="middle">&lt;&lt;{escape_xml(self.stereotype)}&gt;&gt;</text>')
            svg.append(f'<text x="{self.x + self.w/2}" y="{self.y + 34}" fill="#ffffff" font-family="Helvetica, Arial, sans-serif" font-size="14" font-weight="bold" text-anchor="middle">{escape_xml(self.name)}</text>')
        else:
            svg.append(f'<text x="{self.x + self.w/2}" y="{self.y + 22}" fill="#ffffff" font-family="Helvetica, Arial, sans-serif" font-size="14" font-weight="bold" text-anchor="middle">{escape_xml(self.name)}</text>')

        # Divider after header
        svg.append(f'<line x1="{self.x}" y1="{self.y+header_h}" x2="{self.x+self.w}" y2="{self.y+header_h}" stroke="#64748b" stroke-width="1"/>')
        
        # Attributes compartment
        cur_y = self.y + header_h + 16
        for attr in self.attrs:
            svg.append(f'<text x="{self.x + 12}" y="{cur_y}" fill="#1e293b" font-family="Consolas, Monaco, monospace" font-size="11">{escape_xml(attr)}</text>')
            cur_y += 16
        
        # Divider between attrs and methods
        method_divider_y = self.y + header_h + attr_h
        svg.append(f'<line x1="{self.x}" y1="{method_divider_y}" x2="{self.x+self.w}" y2="{method_divider_y}" stroke="#cbd5e1" stroke-width="1"/>')
        
        # Methods compartment
        cur_y = method_divider_y + 16
        for meth in self.methods:
            svg.append(f'<text x="{self.x + 12}" y="{cur_y}" fill="#0f172a" font-family="Consolas, Monaco, monospace" font-size="11">{escape_xml(meth)}</text>')
            cur_y += 16

        return "\n".join(svg)

def make_defs():
    return '''
    <defs>
      <!-- Arrow markers -->
      <marker id="arrow" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
        <path d="M 0 1 L 10 5 L 0 9 z" fill="#475569" />
      </marker>
      <marker id="open_arrow" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
        <path d="M 0 1 L 10 5 L 0 9" fill="none" stroke="#475569" stroke-width="1.5" />
      </marker>
      <marker id="hollow_triangle" viewBox="0 0 12 12" refX="12" refY="6" markerWidth="9" markerHeight="9" orient="auto-start-reverse">
        <path d="M 0 1 L 12 6 L 0 11 z" fill="#ffffff" stroke="#334155" stroke-width="1.5"/>
      </marker>
      <marker id="diamond_filled" viewBox="0 0 16 10" refX="16" refY="5" markerWidth="11" markerHeight="8" orient="auto-start-reverse">
        <path d="M 0 5 L 8 1 L 16 5 L 8 9 z" fill="#1e293b" stroke="#1e293b" stroke-width="1.2"/>
      </marker>
      <marker id="diamond_hollow" viewBox="0 0 16 10" refX="16" refY="5" markerWidth="11" markerHeight="8" orient="auto-start-reverse">
        <path d="M 0 5 L 8 1 L 16 5 L 8 9 z" fill="#ffffff" stroke="#1e293b" stroke-width="1.4"/>
      </marker>
    </defs>
    '''

def draw_edge(x1, y1, x2, y2, label="", marker_end="", marker_start="", style="solid", color="#475569", label_pos=(0, 0)):
    stroke_dash = 'stroke-dasharray="6,4"' if style == "dashed" else ""
    m_end = f'marker-end="url(#{marker_end})"' if marker_end else ""
    m_start = f'marker-start="url(#{marker_start})"' if marker_start else ""
    
    res = [f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="1.6" {stroke_dash} {m_start} {m_end}/>']
    if label:
        lx = (x1 + x2) / 2 + label_pos[0]
        ly = (y1 + y2) / 2 + label_pos[1]
        res.append(f'<rect x="{lx-4}" y="{ly-11}" width="{len(label)*6.5+10}" height="15" rx="3" fill="#f8fafc" stroke="#e2e8f0" stroke-width="0.8"/>')
        res.append(f'<text x="{lx+2}" y="{ly}" fill="#334155" font-family="Helvetica, Arial, sans-serif" font-size="10" font-weight="600">{escape_xml(label)}</text>')
    return "\n".join(res)

def draw_polyline(points, label="", marker_end="", marker_start="", style="solid", color="#475569", label_pos=(0, 0)):
    pts_str = " ".join(f"{x},{y}" for x, y in points)
    stroke_dash = 'stroke-dasharray="6,4"' if style == "dashed" else ""
    m_end = f'marker-end="url(#{marker_end})"' if marker_end else ""
    m_start = f'marker-start="url(#{marker_start})"' if marker_start else ""
    
    res = [f'<polyline points="{pts_str}" fill="none" stroke="{color}" stroke-width="1.6" {stroke_dash} {m_start} {m_end}/>']
    if label and len(points) >= 2:
        mid_idx = len(points) // 2
        p1 = points[mid_idx - 1]
        p2 = points[mid_idx]
        lx = (p1[0] + p2[0]) / 2 + label_pos[0]
        ly = (p1[1] + p2[1]) / 2 + label_pos[1]
        res.append(f'<rect x="{lx-4}" y="{ly-11}" width="{len(label)*6.5+10}" height="15" rx="3" fill="#f8fafc" stroke="#e2e8f0" stroke-width="0.8"/>')
        res.append(f'<text x="{lx+2}" y="{ly}" fill="#334155" font-family="Helvetica, Arial, sans-serif" font-size="10" font-weight="600">{escape_xml(label)}</text>')
    return "\n".join(res)


# ==============================================================================
# DIAGRAM 1: MASTER CLASS DIAGRAM (HIGH-LEVEL ARCHITECTURAL OVERVIEW)
# ==============================================================================
def generate_master_class_diagram():
    svg_w, svg_h = 1750, 1200
    
    # Classes across all subsystems
    c_neo4j = UMLBox(
        name="Neo4jClient",
        attrs=[
            "- driver: neo4j.Driver",
        ],
        methods=[
            "+ __init__()",
            "+ run_query(query: str, params: dict): list[dict]",
            "+ stream_query(query: str, params: dict, batch: int): Generator",
            "+ close(): void",
        ],
        x=80, y=140, w=340, color="#1e3a8a"
    )

    c_builder = UMLBox(
        name="WeightedGraphBuilder",
        attrs=[
            "+ HUB_RELATIONS: set[str]",
            "- chunk_size: int = 500000",
        ],
        methods=[
            "+ filter_hub_relations(df: DataFrame): DataFrame",
            "+ build_from_embeddings(df, embs, id_map): tuple[Graph, list]",
        ],
        x=80, y=360, w=340, color="#0f766e"
    )

    c_leiden = UMLBox(
        name="LeidenPartitioner",
        attrs=[
            "- resolution: float = 1.0",
            "- n_iterations: int = 2",
            "- seed: int = 42",
            "- method: str = 'rb'",
        ],
        methods=[
            "+ partition(entities, edges): tuple[dict, dict, float]",
            "+ split_oversized_communities(g, membership): list",
        ],
        x=80, y=560, w=340, color="#0f766e"
    )

    c_bm25 = UMLBox(
        name="BM25Retriever",
        attrs=[
            "- bm25: BM25Okapi",
            "- documents: list[dict]",
        ],
        methods=[
            "+ tokenize(text: str): list[str]",
            "+ build(documents: list[dict]): void",
            "+ search(query: str, top_k: int): list[dict]",
        ],
        x=520, y=140, w=330, color="#4338ca"
    )

    c_dense = UMLBox(
        name="DenseRetriever",
        attrs=[
            "- model: SentenceTransformer",
            "- embeddings: ndarray",
            "- documents: list[dict]",
        ],
        methods=[
            "+ build(documents: list[dict], batch_size: int): void",
            "+ search(query: str, top_k: int): list[dict]",
        ],
        x=520, y=340, w=330, color="#4338ca"
    )

    c_hybrid = UMLBox(
        name="HybridRetriever",
        attrs=[
            "- bm25_retriever: BM25Retriever",
            "- dense_retriever: DenseRetriever",
            "- fusion_method: str = 'rrf'",
            "- rrf_k: int = 60",
        ],
        methods=[
            "+ search(query: str, top_k: int): list[dict]",
            "- _fuse_rrf(dense_res, bm25_res, top_k): list[dict]",
        ],
        x=520, y=550, w=330, color="#4338ca"
    )

    c_part_retriever = UMLBox(
        name="PartitionRetriever",
        attrs=[
            "- centroids: ndarray",
            "- bm25: BM25Okapi",
            "- community_id_order: list[int]",
            "- community_entities: dict[int, list[str]]",
            "- rrf_k: int = 60",
        ],
        methods=[
            "+ load_full_scale(indexes_dir, data_dir): void",
            "+ search(query: str, top_k: int, client: Neo4jClient): list[dict]",
            "- _search_full_scale(query, top_k, client): list[dict]",
        ],
        x=520, y=800, w=380, color="#6b21a8"
    )

    c_adapter = UMLBox(
        name="PartitionRAGTraversalAdapter",
        attrs=[
            "- traverser: PathScoringTraverser",
            "- context_builder: EvidenceContextBuilder",
        ],
        methods=[
            "+ retrieve(query, seeds, hits): PartitionTraversalResponse",
        ],
        x=980, y=800, w=370, color="#9d174d"
    )

    c_traverser = UMLBox(
        name="PathScoringTraverser",
        attrs=[
            "- client: QueryClient",
            "- config: PathScoringConfig",
            "- model: SentenceTransformer",
        ],
        methods=[
            "+ traverse(query, seeds, allowed_eids): TraversalResult",
            "- _fetch_neighbors(sources, allowed): list[KGEdge]",
            "- _score_candidates(q, emb, cands): list[tuple]",
            "- _local_bm25(query, docs): ndarray",
        ],
        x=980, y=430, w=370, color="#9d174d"
    )

    c_context_bld = UMLBox(
        name="EvidenceContextBuilder",
        attrs=[
            "- max_description_chars: int = 240",
        ],
        methods=[
            "+ build(result: TraversalResult): EvidenceBundle",
            "- _citation(label: str, edge: KGEdge): dict",
        ],
        x=980, y=180, w=370, color="#9d174d"
    )

    c_ev_bundle = UMLBox(
        name="EvidenceBundle",
        stereotype="dataclass",
        attrs=[
            "+ context: str",
            "+ candidate_answer_ids: tuple[str, ...]",
            "+ paths: tuple[EvidencePath, ...]",
            "+ citations: tuple[dict, ...]",
            "+ traversal_metadata: dict[str, int]",
        ],
        methods=[
            "+ as_dict(): dict",
        ],
        x=1400, y=180, w=300, color="#0369a1"
    )

    c_pipeline = UMLBox(
        name="EnhancedKGRAGPipeline",
        stereotype="facade / orchestrator",
        attrs=[
            "- partition_retriever: PartitionRetriever",
            "- traversal_adapter: PartitionRAGTraversalAdapter",
            "- llm_client: LLMClient",
        ],
        methods=[
            "+ answer_query(query: str, seed_eids: list[str]): dict",
            "+ retrieve_context(query: str, seed_eids: list[str]): EvidenceBundle",
        ],
        x=1400, y=650, w=320, color="#1e293b"
    )

    c_llm = UMLBox(
        name="LLMClient",
        stereotype="boundary / external",
        attrs=[
            "- model_name: str",
            "- temperature: float = 0.0",
        ],
        methods=[
            "+ generate_grounded_response(prompt: str): str",
        ],
        x=1400, y=940, w=320, color="#334155"
    )

    boxes = [
        c_neo4j, c_builder, c_leiden,
        c_bm25, c_dense, c_hybrid, c_part_retriever,
        c_adapter, c_traverser, c_context_bld,
        c_ev_bundle, c_pipeline, c_llm
    ]

    elements_svg = "\n".join(b.render_svg() for b in boxes)

    # Connections
    conns = []
    # WeightedGraphBuilder uses Neo4j extracted triples
    conns.append(draw_edge(250, 360, 250, 270, label="extracts from", marker_end="open_arrow", style="dashed"))
    # LeidenPartitioner uses graph
    conns.append(draw_edge(250, 560, 250, 480, label="clusters topology", marker_end="open_arrow", style="dashed"))
    
    # HybridRetriever aggregates BM25 & Dense
    conns.append(draw_edge(685, 550, 685, 275, label="1..* aggregates", marker_start="diamond_hollow", marker_end="open_arrow"))
    conns.append(draw_edge(685, 550, 685, 465, label="1..* aggregates", marker_start="diamond_hollow", marker_end="open_arrow"))
    
    # PartitionRetriever uses Neo4jClient and Hybrid indices
    conns.append(draw_edge(520, 840, 420, 240, label="live lookup", marker_end="open_arrow", style="dashed"))
    conns.append(draw_edge(685, 800, 685, 690, label="uses indices", marker_end="open_arrow", style="dashed"))
    
    # Adapter uses PartitionRetriever output & coordinates Traverser + ContextBuilder
    conns.append(draw_edge(980, 840, 900, 840, label="consumes hits", marker_end="open_arrow", style="dashed"))
    conns.append(draw_edge(1165, 800, 1165, 620, label="delegates traversal", marker_end="open_arrow"))
    conns.append(draw_polyline([(1165, 800), (1165, 300)], label="builds context", marker_end="open_arrow", style="dashed"))
    
    # Traverser queries Neo4jClient
    conns.append(draw_polyline([(980, 500), (450, 500), (420, 240)], label="queries within partition", marker_end="open_arrow", style="dashed"))
    
    # EvidenceContextBuilder produces EvidenceBundle
    conns.append(draw_edge(1350, 240, 1400, 240, label="produces", marker_end="open_arrow", style="dashed"))
    
    # Pipeline coordinates PartitionRetriever, Adapter, and LLMClient
    conns.append(draw_polyline([(1400, 720), (900, 880)], label="1. routes query", marker_end="open_arrow"))
    conns.append(draw_edge(1400, 800, 1350, 830, label="2. traverses", marker_end="open_arrow"))
    conns.append(draw_edge(1560, 800, 1560, 940, label="3. invokes", marker_end="open_arrow"))

    # Packages / subsystem backgrounds
    bg_svg = f'''
    <rect x="50" y="70" width="400" height="1060" rx="10" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.5"/>
    <text x="70" y="105" fill="#1e3a8a" font-family="Helvetica, Arial, sans-serif" font-size="16" font-weight="bold">Data Layer &amp; Preprocessing</text>
    
    <rect x="490" y="70" width="440" height="1060" rx="10" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.5"/>
    <text x="510" y="105" fill="#4338ca" font-family="Helvetica, Arial, sans-serif" font-size="16" font-weight="bold">Hybrid Routing Subsystem</text>

    <rect x="955" y="70" width="420" height="1060" rx="10" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.5"/>
    <text x="975" y="105" fill="#9d174d" font-family="Helvetica, Arial, sans-serif" font-size="16" font-weight="bold">Query-Guided Traversal Subsystem</text>

    <rect x="1390" y="70" width="340" height="1060" rx="10" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.5"/>
    <text x="1405" y="105" fill="#0f172a" font-family="Helvetica, Arial, sans-serif" font-size="16" font-weight="bold">Grounded Generation Subsystem</text>
    '''

    title_svg = f'''
    <text x="{svg_w/2}" y="42" fill="#0f172a" font-family="Helvetica, Arial, sans-serif" font-size="22" font-weight="bold" text-anchor="middle">Enhanced-KG-LLM: Master Architecture Class Diagram</text>
    '''

    svg_content = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w}" height="{svg_h}" viewBox="0 0 {svg_w} {svg_h}">
    {make_defs()}
    <rect width="{svg_w}" height="{svg_h}" fill="#f1f5f9"/>
    {title_svg}
    {bg_svg}
    {"\n".join(conns)}
    {elements_svg}
    </svg>'''
    
    return svg_content


# ==============================================================================
# DIAGRAM 2: SUB-LEVEL 1 - PREPROCESSING & DATABASE SUBSYSTEM
# ==============================================================================
def generate_sublevel_1_diagram():
    svg_w, svg_h = 1350, 920

    c_neo4j = UMLBox(
        name="Neo4jClient",
        attrs=[
            "- driver: neo4j.Driver",
        ],
        methods=[
            "+ __init__()",
            "+ close(): void",
            "+ run_query(query: str, parameters: dict = None): list[dict]",
            "+ stream_query(query: str, parameters: dict = None, batch_size: int = 10000): Generator",
            "+ __enter__(): Neo4jClient",
            "+ __exit__(exc_type, exc_val, exc_tb): void",
        ],
        x=80, y=140, w=440, color="#1e3a8a"
    )

    c_extractor = UMLBox(
        name="GraphExtractor",
        stereotype="module / utility",
        attrs=[
            "- batch_size: int = 10000",
            "- chunk_size: int = 2000",
        ],
        methods=[
            "+ extract_entities(client: Neo4jClient, limit: int): dict[str, dict]",
            "+ extract_edges_for_entities(client: Neo4jClient, entity_ids: set): list[tuple]",
            "+ extract_all_edges(client: Neo4jClient, batch_size: int): list[tuple]",
        ],
        x=80, y=520, w=440, color="#1e3a8a"
    )

    c_builder = UMLBox(
        name="WeightedGraphBuilder",
        attrs=[
            "+ HUB_RELATIONS: set[str] = {'P31','P17','P27','P131','P19','P20','P21','P106'}",
            "- chunk_size: int = 500000",
        ],
        methods=[
            "+ __init__(chunk_size: int = 500000)",
            "+ filter_hub_relations(triples_df: DataFrame): DataFrame",
            "+ build_from_embeddings(triples_df, embeddings, id_to_row): tuple[ig.Graph, list]",
        ],
        x=620, y=140, w=450, color="#0f766e"
    )

    c_leiden = UMLBox(
        name="LeidenPartitioner",
        attrs=[
            "- resolution: float = 1.0",
            "- n_iterations: int = 2",
            "- seed: int = 42",
            "- method: str = 'rb'",
        ],
        methods=[
            "+ __init__(resolution=1.0, n_iterations=2, seed=42, method='rb')",
            "+ partition(entities: dict, edges: list): tuple[dict, dict, float]",
            "+ split_oversized_communities(g: ig.Graph, membership: list): list",
        ],
        x=620, y=420, w=450, color="#0f766e"
    )

    c_storage = UMLBox(
        name="PartitionStorage",
        stereotype="module / persistence",
        attrs=[
            "- output_dir: str = 'data/partitions'",
        ],
        methods=[
            "+ save_partitions(partition_map: dict, metadata: dict, output_dir: str): void",
            "+ load_partitions(input_dir: str): tuple[dict, dict]",
        ],
        x=620, y=700, w=450, color="#0f766e"
    )

    boxes = [c_neo4j, c_extractor, c_builder, c_leiden, c_storage]
    elements_svg = "\n".join(b.render_svg() for b in boxes)

    conns = []
    conns.append(draw_edge(300, 520, 300, 320, label="streams records via", marker_end="open_arrow", style="dashed"))
    conns.append(draw_polyline([(520, 600), (570, 600), (570, 240), (620, 240)], label="provides triples & entities", marker_end="open_arrow", style="dashed"))
    conns.append(draw_edge(845, 300, 845, 420, label="produces weighted igraph", marker_end="open_arrow"))
    conns.append(draw_edge(845, 590, 845, 700, label="serializes partition maps", marker_end="open_arrow", style="dashed"))

    title_svg = f'''
    <text x="{svg_w/2}" y="48" fill="#0f172a" font-family="Helvetica, Arial, sans-serif" font-size="22" font-weight="bold" text-anchor="middle">Sub-Level 1: Database &amp; Graph Preprocessing Class Diagram</text>
    <text x="{svg_w/2}" y="76" fill="#64748b" font-family="Helvetica, Arial, sans-serif" font-size="13" text-anchor="middle">Extraction, Topological Sanitization, Semantic Weighting, and Leiden Community Decomposition</text>
    '''

    svg_content = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w}" height="{svg_h}" viewBox="0 0 {svg_w} {svg_h}">
    {make_defs()}
    <rect width="{svg_w}" height="{svg_h}" fill="#f8fafc"/>
    {title_svg}
    {"\n".join(conns)}
    {elements_svg}
    </svg>'''
    return svg_content


# ==============================================================================
# DIAGRAM 3: SUB-LEVEL 2 - HYBRID RETRIEVAL & PARTITION ROUTING
# ==============================================================================
def generate_sublevel_2_diagram():
    svg_w, svg_h = 1450, 1000

    c_entity_doc = UMLBox(
        name="EntityDocumentBuilder",
        stereotype="utility",
        attrs=[
            "- ENTITY_QUERY: str",
        ],
        methods=[
            "+ normalize_aliases(aliases: Any): list[str]",
            "+ entity_to_document(entity: dict): dict",
        ],
        x=80, y=140, w=340, color="#4338ca"
    )

    c_bm25 = UMLBox(
        name="BM25Retriever",
        attrs=[
            "- bm25: BM25Okapi = None",
            "- documents: list[dict] = []",
        ],
        methods=[
            "+ __init__()",
            "+ tokenize(text: str): list[str]",
            "+ build(documents: list[dict]): void",
            "+ search(query: str, top_k: int = 10): list[dict]",
            "+ save(path: str): void",
            "+ load(path: str): void",
        ],
        x=480, y=140, w=380, color="#4338ca"
    )

    c_dense = UMLBox(
        name="DenseRetriever",
        attrs=[
            "- model: SentenceTransformer",
            "- embeddings: ndarray = None",
            "- documents: list[dict] = []",
        ],
        methods=[
            "+ __init__(model_name: str = 'all-MiniLM-L6-v2')",
            "+ build(documents: list[dict], batch_size: int = 64): void",
            "+ search(query: str, top_k: int = 10): list[dict]",
            "+ save(path: str): void",
            "+ load(path: str): void",
        ],
        x=480, y=440, w=380, color="#4338ca"
    )

    c_hybrid = UMLBox(
        name="HybridRetriever",
        attrs=[
            "- bm25_retriever: BM25Retriever",
            "- dense_retriever: DenseRetriever",
            "- bm25_weight: float = 0.5",
            "- dense_weight: float = 0.5",
            "- fusion_method: str = 'rrf'",
            "- rrf_k: int = 60",
        ],
        methods=[
            "+ __init__(bm25, dense, bm25_w, dense_w, fusion, rrf_k)",
            "+ search(query: str, top_k: int = 10): list[dict]",
            "- _fuse_rrf(dense_res: list, bm25_res: list, top_k: int): list",
            "- _fuse_weighted(dense_res: list, bm25_res: list, top_k: int): list",
        ],
        x=940, y=260, w=430, color="#4338ca"
    )

    c_partition_retriever = UMLBox(
        name="PartitionRetriever",
        attrs=[
            "- centroids: ndarray = None",
            "- bm25: BM25Okapi = None",
            "- community_id_order: list[int] = []",
            "- community_entities: dict[int, list[str]] = {}",
            "- community_metadata: dict[int, dict] = {}",
            "- is_full_scale: bool = False",
            "- rrf_k: int = 60",
            "- hybrid_retriever: HybridRetriever = None",
        ],
        methods=[
            "+ __init__(bm25_weight=0.5, dense_weight=0.5, model='all-MiniLM-L6-v2', rrf_k=60)",
            "+ load_full_scale(indexes_dir: str, data_dir: str): void",
            "+ build_from_communities(community_metadata: dict, save_dir: str): void",
            "+ search(query: str, top_k: int = 5, client=None): list[dict]",
            "- _search_full_scale(query: str, top_k: int, client=None): list[dict]",
            "- _search_text_based(query: str, top_k: int): list[dict]",
        ],
        x=940, y=600, w=430, color="#6b21a8"
    )

    c_neo4j = UMLBox(
        name="Neo4jClient",
        attrs=[
            "- driver: neo4j.Driver",
        ],
        methods=[
            "+ run_query(query: str, params: dict): list[dict]",
        ],
        x=480, y=780, w=380, color="#1e3a8a"
    )

    boxes = [c_entity_doc, c_bm25, c_dense, c_hybrid, c_partition_retriever, c_neo4j]
    elements_svg = "\n".join(b.render_svg() for b in boxes)

    conns = []
    # Entity doc builder to retrievers
    conns.append(draw_edge(420, 200, 480, 200, label="formats corpus", marker_end="open_arrow", style="dashed"))
    # HybridRetriever aggregates BM25 and Dense
    conns.append(draw_polyline([(940, 320), (890, 320), (890, 240), (860, 240)], label="aggregates", marker_start="diamond_hollow", marker_end="open_arrow"))
    conns.append(draw_polyline([(940, 380), (890, 380), (890, 520), (860, 520)], label="aggregates", marker_start="diamond_hollow", marker_end="open_arrow"))
    # PartitionRetriever utilizes HybridRetriever for text-based mode
    conns.append(draw_edge(1155, 600, 1155, 480, label="fallback router", marker_start="diamond_hollow", marker_end="open_arrow"))
    # PartitionRetriever resolves live entity names with Neo4jClient
    conns.append(draw_edge(940, 830, 860, 830, label="resolves live hub names", marker_end="open_arrow", style="dashed"))

    title_svg = f'''
    <text x="{svg_w/2}" y="48" fill="#0f172a" font-family="Helvetica, Arial, sans-serif" font-size="22" font-weight="bold" text-anchor="middle">Sub-Level 2: Parallel Hybrid Retrieval &amp; Partition Routing Class Diagram</text>
    <text x="{svg_w/2}" y="76" fill="#64748b" font-family="Helvetica, Arial, sans-serif" font-size="13" text-anchor="middle">Dual-Channel Dense Vector and BM25 Sparse Search with Reciprocal Rank Fusion (RRF)</text>
    '''

    svg_content = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w}" height="{svg_h}" viewBox="0 0 {svg_w} {svg_h}">
    {make_defs()}
    <rect width="{svg_w}" height="{svg_h}" fill="#f8fafc"/>
    {title_svg}
    {"\n".join(conns)}
    {elements_svg}
    </svg>'''
    return svg_content


# ==============================================================================
# DIAGRAM 4: SUB-LEVEL 3 - TRAVERSAL & EVIDENCE SUBSYSTEM
# ==============================================================================
def generate_sublevel_3_diagram():
    svg_w, svg_h = 1600, 1100

    c_edge = UMLBox(
        name="KGEdge",
        stereotype="dataclass (immutable)",
        attrs=[
            "+ source_id: str",
            "+ source_name: str",
            "+ relation_id: str",
            "+ relation_name: str",
            "+ target_id: str",
            "+ target_name: str",
            "+ target_description: str = ''",
        ],
        methods=[
            "+ citation_id(): str",
            "+ text(): str",
        ],
        x=80, y=140, w=330, color="#0369a1"
    )

    c_path = UMLBox(
        name="EvidencePath",
        stereotype="dataclass (immutable)",
        attrs=[
            "+ score: float",
            "+ edges: tuple[KGEdge, ...]",
        ],
        methods=[
            "+ answer_entity_ids(): tuple[str, ...]",
        ],
        x=80, y=440, w=330, color="#0369a1"
    )

    c_res = UMLBox(
        name="TraversalResult",
        stereotype="dataclass (immutable)",
        attrs=[
            "+ query: str",
            "+ seed_entity_ids: tuple[str, ...]",
            "+ scoped_entity_count: int",
            "+ paths: tuple[EvidencePath, ...]",
            "+ edges_examined: int",
            "+ candidate_answer_ids: tuple[str, ...]",
        ],
        methods=[],
        x=80, y=660, w=330, color="#0369a1"
    )

    c_cfg = UMLBox(
        name="PathScoringConfig",
        stereotype="dataclass",
        attrs=[
            "+ max_hops: int = 3",
            "+ beam_width: int = 5",
            "+ max_paths: int = 3",
            "+ max_neighbors_per_node: int = 100",
            "+ dense_weight: float = 0.35",
            "+ lexical_weight: float = 0.35",
            "+ relation_weight: float = 0.20",
            "+ relation_novelty_weight: float = 0.10",
            "+ hop_penalty: float = 0.05",
        ],
        methods=[
            "+ __post_init__(): void",
        ],
        x=480, y=140, w=350, color="#9d174d"
    )

    c_traverser = UMLBox(
        name="PathScoringTraverser",
        attrs=[
            "- client: QueryClient",
            "- config: PathScoringConfig",
            "- _model: Encoder = None",
            "- model_name: str = 'all-MiniLM-L6-v2'",
        ],
        methods=[
            "+ __init__(client, config, model, model_name)",
            "+ traverse(query: str, seeds: Iterable[str], allowed: Iterable[str]): TraversalResult",
            "- _fetch_neighbors(source_ids, allowed: frozenset): list[KGEdge]",
            "- _score_candidates(query, emb, candidates): list[tuple[KGEdge, float]]",
            "- _encode(texts: Sequence[str]): ndarray",
            "- _local_bm25(query: str, documents: Sequence[str]): ndarray",
            "- _relation_relevance(relation: str, query: str): float",
            "- _distinct_paths(paths, maximum: int): list[EvidencePath]",
        ],
        x=480, y=490, w=440, color="#9d174d"
    )

    c_bundle = UMLBox(
        name="EvidenceBundle",
        stereotype="dataclass (immutable)",
        attrs=[
            "+ context: str",
            "+ candidate_answer_ids: tuple[str, ...]",
            "+ paths: tuple[EvidencePath, ...]",
            "+ citations: tuple[dict[str, str], ...]",
            "+ traversal_metadata: dict[str, int]",
        ],
        methods=[
            "+ as_dict(): dict",
        ],
        x=1020, y=140, w=350, color="#0369a1"
    )

    c_ctx_bld = UMLBox(
        name="EvidenceContextBuilder",
        attrs=[
            "- max_description_chars: int = 240",
        ],
        methods=[
            "+ __init__(max_description_chars: int = 240)",
            "+ build(result: TraversalResult): EvidenceBundle",
            "- _description(description: str): str",
            "- _citation(label: str, edge: KGEdge): dict[str, str]",
        ],
        x=1020, y=460, w=350, color="#9d174d"
    )

    c_adapter = UMLBox(
        name="PartitionRAGTraversalAdapter",
        attrs=[
            "- traverser: PathScoringTraverser",
            "- context_builder: EvidenceContextBuilder",
        ],
        methods=[
            "+ __init__(traverser, context_builder = None)",
            "+ retrieve(query, seeds, partition_hits): PartitionTraversalResponse",
        ],
        x=1020, y=740, w=440, color="#9d174d"
    )

    c_resp = UMLBox(
        name="PartitionTraversalResponse",
        stereotype="dataclass (immutable)",
        attrs=[
            "+ bundle: EvidenceBundle",
            "+ selected_partition_ids: tuple[int, ...]",
        ],
        methods=[],
        x=1020, y=930, w=440, color="#0369a1"
    )

    boxes = [c_edge, c_path, c_res, c_cfg, c_traverser, c_bundle, c_ctx_bld, c_adapter, c_resp]
    elements_svg = "\n".join(b.render_svg() for b in boxes)

    conns = []
    # Path contains edges
    conns.append(draw_edge(245, 440, 245, 330, label="1..* edges", marker_start="diamond_filled", marker_end="open_arrow"))
    # TraversalResult contains paths
    conns.append(draw_edge(245, 660, 245, 540, label="0..3 paths", marker_start="diamond_filled", marker_end="open_arrow"))
    # Traverser uses PathScoringConfig
    conns.append(draw_edge(655, 490, 655, 410, label="1 configured by", marker_start="diamond_hollow", marker_end="open_arrow"))
    # Traverser produces TraversalResult
    conns.append(draw_polyline([(480, 600), (430, 600), (430, 720), (410, 720)], label="produces", marker_end="open_arrow", style="dashed"))
    # ContextBuilder consumes TraversalResult
    conns.append(draw_polyline([(1020, 520), (950, 520), (950, 750), (410, 750)], label="formats", marker_end="open_arrow", style="dashed"))
    # ContextBuilder produces EvidenceBundle
    conns.append(draw_edge(1195, 460, 1195, 360, label="produces", marker_end="open_arrow", style="dashed"))
    # Adapter coordinates Traverser and ContextBuilder
    conns.append(draw_edge(1020, 790, 920, 650, label="invokes", marker_end="open_arrow"))
    conns.append(draw_edge(1195, 740, 1195, 610, label="invokes", marker_end="open_arrow"))
    # Adapter produces PartitionTraversalResponse
    conns.append(draw_edge(1195, 870, 1195, 930, label="yields", marker_end="open_arrow", style="dashed"))
    # PartitionTraversalResponse has EvidenceBundle
    conns.append(draw_polyline([(1460, 980), (1500, 980), (1500, 240), (1370, 240)], label="wraps", marker_start="diamond_filled", marker_end="open_arrow"))

    title_svg = f'''
    <text x="{svg_w/2}" y="48" fill="#0f172a" font-family="Helvetica, Arial, sans-serif" font-size="22" font-weight="bold" text-anchor="middle">Sub-Level 3: Query-Guided Subgraph Traversal &amp; Evidence Assembly Class Diagram</text>
    <text x="{svg_w/2}" y="76" fill="#64748b" font-family="Helvetica, Arial, sans-serif" font-size="13" text-anchor="middle">Bounded Beam Search (T10 Algorithm), Complete Path Scoring, and Citation Context Builder</text>
    '''

    svg_content = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w}" height="{svg_h}" viewBox="0 0 {svg_w} {svg_h}">
    {make_defs()}
    <rect width="{svg_w}" height="{svg_h}" fill="#f8fafc"/>
    {title_svg}
    {"\n".join(conns)}
    {elements_svg}
    </svg>'''
    return svg_content


# ==============================================================================
# DIAGRAM 5: SUB-LEVEL 4 - PIPELINE ORCHESTRATION & GROUNDED GENERATION
# ==============================================================================
def generate_sublevel_4_diagram():
    svg_w, svg_h = 1350, 850

    c_part_retriever = UMLBox(
        name="PartitionRetriever",
        attrs=[
            "- centroids: ndarray",
            "- bm25: BM25Okapi",
            "- rrf_k: int = 60",
        ],
        methods=[
            "+ search(query: str, top_k: int = 3, client = None): list[dict]",
        ],
        x=80, y=140, w=360, color="#6b21a8"
    )

    c_adapter = UMLBox(
        name="PartitionRAGTraversalAdapter",
        attrs=[
            "- traverser: PathScoringTraverser",
            "- context_builder: EvidenceContextBuilder",
        ],
        methods=[
            "+ retrieve(query, seeds, hits): PartitionTraversalResponse",
        ],
        x=80, y=360, w=360, color="#9d174d"
    )

    c_ev_bundle = UMLBox(
        name="EvidenceBundle",
        stereotype="dataclass",
        attrs=[
            "+ context: str",
            "+ candidate_answer_ids: tuple[str, ...]",
            "+ paths: tuple[EvidencePath, ...]",
            "+ citations: tuple[dict, ...]",
        ],
        methods=[
            "+ as_dict(): dict",
        ],
        x=80, y=560, w=360, color="#0369a1"
    )

    c_pipeline = UMLBox(
        name="EnhancedKGRAGPipeline",
        stereotype="Orchestrator",
        attrs=[
            "- retriever: PartitionRetriever",
            "- adapter: PartitionRAGTraversalAdapter",
            "- llm_client: LLMClient",
            "- top_partitions: int = 3",
        ],
        methods=[
            "+ __init__(retriever, adapter, llm_client, top_partitions=3)",
            "+ query(question: str, seed_entity_ids: list[str]): GroundedAnswer",
            "+ get_evidence_only(question: str, seeds: list[str]): EvidenceBundle",
        ],
        x=550, y=280, w=390, color="#1e293b"
    )

    c_answer = UMLBox(
        name="GroundedAnswer",
        stereotype="dataclass",
        attrs=[
            "+ answer_text: str",
            "+ citations: list[dict]",
            "+ candidate_entities: list[str]",
            "+ total_latency_ms: float",
        ],
        methods=[],
        x=550, y=580, w=390, color="#0369a1"
    )

    c_llm = UMLBox(
        name="LLMClient",
        stereotype="Boundary / API Gateway",
        attrs=[
            "- model_name: str = 'gemini-1.5-pro' / 'llama3'",
            "- temperature: float = 0.0",
            "- system_instruction: str",
        ],
        methods=[
            "+ __init__(model_name: str, temperature: float = 0.0)",
            "+ generate_grounded_response(prompt_context: str): str",
        ],
        x=1020, y=280, w=300, color="#334155"
    )

    boxes = [c_part_retriever, c_adapter, c_ev_bundle, c_pipeline, c_answer, c_llm]
    elements_svg = "\n".join(b.render_svg() for b in boxes)

    conns = []
    # Pipeline aggregates Retriever, Adapter, and LLMClient
    conns.append(draw_edge(550, 320, 440, 220, label="1. routes query", marker_start="diamond_filled", marker_end="open_arrow"))
    conns.append(draw_edge(550, 400, 440, 400, label="2. traverses", marker_start="diamond_filled", marker_end="open_arrow"))
    conns.append(draw_edge(940, 350, 1020, 350, label="3. generates response", marker_start="diamond_filled", marker_end="open_arrow"))
    
    # Adapter produces EvidenceBundle
    conns.append(draw_edge(260, 480, 260, 560, label="yields bundle", marker_end="open_arrow", style="dashed"))
    # EvidenceBundle passed to LLMClient
    conns.append(draw_polyline([(440, 620), (500, 620), (500, 460), (1050, 460), (1050, 410)], label="provides structured context", marker_end="open_arrow", style="dashed"))
    # Pipeline produces GroundedAnswer
    conns.append(draw_edge(745, 480, 745, 580, label="produces", marker_end="open_arrow", style="dashed"))

    title_svg = f'''
    <text x="{svg_w/2}" y="48" fill="#0f172a" font-family="Helvetica, Arial, sans-serif" font-size="22" font-weight="bold" text-anchor="middle">Sub-Level 4: End-to-End RAG Pipeline &amp; Grounded Generation Class Diagram</text>
    <text x="{svg_w/2}" y="76" fill="#64748b" font-family="Helvetica, Arial, sans-serif" font-size="13" text-anchor="middle">Orchestration of Partition Routing, Subgraph Traversal, and Hallucination-Free LLM Synthesis</text>
    '''

    svg_content = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w}" height="{svg_h}" viewBox="0 0 {svg_w} {svg_h}">
    {make_defs()}
    <rect width="{svg_w}" height="{svg_h}" fill="#f8fafc"/>
    {title_svg}
    {"\n".join(conns)}
    {elements_svg}
    </svg>'''
    return svg_content


# ==============================================================================
# MAIN RENDERER (SVG -> PNG via sips)
# ==============================================================================
def render_all():
    diagrams = {
        "master_class_diagram": generate_master_class_diagram(),
        "sublevel_1_preprocessing_db": generate_sublevel_1_diagram(),
        "sublevel_2_retrieval_routing": generate_sublevel_2_diagram(),
        "sublevel_3_traversal_evidence": generate_sublevel_3_diagram(),
        "sublevel_4_pipeline_generation": generate_sublevel_4_diagram(),
    }

    for name, svg in diagrams.items():
        svg_path = os.path.join(DIAGRAMS_DIR, f"{name}.svg")
        png_path = os.path.join(DIAGRAMS_DIR, f"{name}.png")

        with open(svg_path, "w", encoding="utf-8") as f:
            f.write(svg)

        print(f"Converting {svg_path} -> {png_path}...")
        cmd = ["sips", "-s", "format", "png", svg_path, "--out", png_path]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"Error converting {name}: {res.stderr}")
        else:
            print(f"Successfully generated {png_path}")

if __name__ == "__main__":
    render_all()
