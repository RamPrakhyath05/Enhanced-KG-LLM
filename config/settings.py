import os
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# Subgraph Partition Artifacts Directory (Loose coupling for swapping partition versions)
DEFAULT_SUBG_DIR = "data/Partitions" if os.path.exists("data/Partitions") else "data/subg"
SUBG_DIR = os.getenv("SUBG_DIR", DEFAULT_SUBG_DIR)

# Ollama LLM Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
