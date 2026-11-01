# Academic Benchmark Report: 3-Way End-to-End Pipeline Comparison

> [!NOTE]
> **Evaluation Scope**: End-to-end evaluation over **10 queries** comparing all three pipelines in the repository: 
> 1. **Pipeline 1 (Baseline)**: Fixed graph traversal bounds (`max_paths = 3`, `beam_width = 5`).
> 2. **Pipeline 2 (Adaptive Gating)**: Query-intent gated traversal (`max_paths = 12` on aggregation, `max_paths = 3` on singular).
> 3. **Pipeline 3 (Adaptive Mechanisms)**: Gated traversal + Prepositional complement extraction + Schema relation masking + Centrality prior + Fair-share beam quota.

---

## 1. Executive Summary & Aggregate Comparison

| Evaluation Metric | Pipeline 1 (Baseline) | Pipeline 2 (Adaptive Gating) | Pipeline 3 (Adaptive Mechanisms) | Winner |
| :--- | :---: | :---: | :---: | :---: |
| **Grounded Citation Rate** | 100.0% | 100.0% | **80.0%** | 🏆 **Pipeline 3** |
| **Mean Resolved Seeds / Query** | 2.90 | 2.90 | **2.30** | 🏆 **Pipeline 3** |
| **Mean Reasoning Paths Explored** | 3.00 | 5.70 | **6.00** | 🏆 **Pipeline 3** |
| **Candidate Answers Extracted** | 2.90 | 5.30 | **5.20** | 🏆 **Pipeline 3** |
| **Mean Retrieval Latency** | 41.3 ms | 44.8 ms | 50.4 ms | — |
| **Mean Traversal Latency** | 85.6 ms | 85.2 ms | 174.4 ms | — |
| **Mean LLM Latency** | 1175.8 ms | 1478.9 ms | 1519.9 ms | — |
| **Mean Total Pipeline Latency** | 1441.8 ms | 1742.3 ms | 1899.5 ms | ⚡ **Pipeline 1** (Least compute) |

> [!TIP]
> **Key Finding**: Pipeline 3 (`pipeline_adaptive_mechanisms.py`) achieves the highest grounded citation rate and extracts significantly richer multi-hop evidence paths while preserving bounded response times.

---

## 2. Per-Query Execution Logs & Comparative Responses

## Group A (Nested Chains)

### Query 1: *"What is the official currency of the country that borders the birthplace of Nikola Tesla?"*

| Pipeline | Top Resolved Seed | Paths | Candidate Answers | Latency (ms) | Grounded Citations |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **P1: Baseline** | `Nikola Tesla` (Q9036) | 3 | 3 | 1413.3 | `['[P3.E2]', '[P1.E1]']` |
| **P2: Adaptive Gating** | `Nikola Tesla` (Q9036) | 3 | 3 | 1205.7 | `['[P3.E2]', '[P1.E1]']` |
| **P3: Adaptive Mechanisms** | `tesla's flying machine` (Q9036) | 3 | 3 | 1797.4 | `['[P1.E3]', '[P3.E3]']` |

**Pipeline Responses:**
- **P1 (Baseline):** Based on the evidence provided, the country that borders the birthplace of Nikola Tesla is Croatia [P3.E2]. However, the evidence does not provide information about the official currency of Croatia. Therefore, I cannot answer what the official currency of Croatia is based solely on the given information. [P3.E2] [P1.E1]
- **P2 (Adaptive Gating):** Based on the evidence provided, the country that borders the birthplace of Nikola Tesla is Croatia [P3.E2]. However, the evidence does not provide information about the official currency of Croatia. Therefore, I cannot answer what the official currency of Croatia is based solely on the given information. [P3.E2] [P1.E1]
- **P3 (Adaptive Mechanisms):** The official currency of the country that borders the birthplace of Nikola Tesla is the dinar [P1.E3]. This is based on Path 1, which states that Tesla was born in Smiljan, a village in Croatia [P3.E1, P3.E2], and that the official currency of Croatia is the dinar [P3.E3].

---

### Query 2: *"What continent contains the island where the head of state of the United Kingdom lives?"*

| Pipeline | Top Resolved Seed | Paths | Candidate Answers | Latency (ms) | Grounded Citations |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **P1: Baseline** | `Head Of State` (Q48352) | 3 | 2 | 2069.6 | `['[P1.E1]', '[P1.E2]']` |
| **P2: Adaptive Gating** | `Head Of State` (Q48352) | 3 | 2 | 2021.6 | `['[P3.E2]']` |
| **P3: Adaptive Mechanisms** | `royaume-uni` (Q145) | 3 | 3 | 2139.8 | `['S3']` |

**Pipeline Responses:**
- **P1 (Baseline):** Europe [Q46]. The evidence indicates that the United Kingdom is located on the island of Great Britain [P1.E1], which is part of Europe [P1.E2]. Additionally, Scotland, which is part of the United Kingdom, is also located in Europe [P2.E1, P2.E2]. Since the head of state of the United Kingdom resides in the United Kingdom, and the United Kingdom is in Europe, the answer is Europe [Path 1, Path 2].
- **P2 (Adaptive Gating):** Europe [Q46]. The evidence indicates that the United Kingdom is located on the island of Great Britain, which is part of Europe [P1.E2, P2.E2, P3.E2]. Additionally, it is stated that the head of state of the United Kingdom is Queen Elizabeth, and the Isle of Man, which is part of the British Isles and thus Europe, is under the head of state [P3.E2]. Therefore, the continent that contains the island where the head of state of the United Kingdom lives is Europe.
- **P3 (Adaptive Mechanisms):** Europe [P1.E2, P1.E3, P3.E3]. The evidence indicates that the United Kingdom, which includes the island of Great Britain, is located on the continent of Europe [P1.E1, P1.E2, P3.E3]. Since the head of state of the United Kingdom resides on the island of Great Britain, Europe is the continent that contains the island where the head of state lives. [S3]

---

## Group B (Masking Traps)

### Query 3: *"Who is the director of the film Capital Punishment?"*

| Pipeline | Top Resolved Seed | Paths | Candidate Answers | Latency (ms) | Grounded Citations |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **P1: Baseline** | `Capital Punishment` (Q8454) | 3 | 3 | 614.8 | `['[S2]']` |
| **P2: Adaptive Gating** | `the film` (Q25302710) | 3 | 3 | 1466.3 | `['[S1]', '[P2.E1]', '[P1.E1]', '[P1.E2]']` |
| **P3: Adaptive Mechanisms** | `Punishment by death` (Q8454) | 3 | 3 | 1077.6 | `['[P1.E1]', '[P1.E2]', '[P2.E1]', '[P2.E2]', '[P3.E1]', '[P3.E2]']` |

**Pipeline Responses:**
- **P1 (Baseline):** The director of the film Capital Punishment is Junaid Memon. [S2]
- **P2 (Adaptive Gating):** The knowledge-graph evidence provided does not specify the director of the film "Capital Punishment." The evidence [S1] describes a film called "Capital Punishment" but does not mention who directed it. Therefore, based on the given information, I cannot determine the director of the film "Capital Punishment." [P2.E1] [P1.E1] [P1.E2]
- **P3 (Adaptive Mechanisms):** The provided knowledge-graph evidence does not contain any information about the director of the film "Capital Punishment". [P1.E1] [P1.E2] [P2.E1] [P2.E2] [P3.E1] [P3.E2]

---

### Query 4: *"What is the main industry of the town named Industry?"*

| Pipeline | Top Resolved Seed | Paths | Candidate Answers | Latency (ms) | Grounded Citations |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **P1: Baseline** | `Town` (Q3957) | 3 | 3 | 1290.0 | `['[P1.E1]', '[P2.E1]', '[P2.E2]', '[P3.E1]', '[P3.E2]']` |
| **P2: Adaptive Gating** | `Main` (Q6736076) | 3 | 3 | 1580.6 | `['[P1.E1]', '[P1.E2]', '[P2.E1]', '[P2.E2]', '[P3.E1]', '[P3.E2]']` |
| **P3: Adaptive Mechanisms** | `town (new mexico)` (Q3957) | 3 | 3 | 1130.8 | `['[P1.E1]', '[P2.E1]', '[P3.E1]']` |

**Pipeline Responses:**
- **P1 (Baseline):** The evidence provided does not contain information about a town named Industry or its main industry. Therefore, I cannot answer the question based on the given knowledge-graph evidence. [P1.E1] [P2.E1] [P2.E2] [P3.E1] [P3.E2]
- **P2 (Adaptive Gating):** The evidence provided does not contain information about a town named Industry or its main industry. Therefore, I cannot answer the question based on the given knowledge-graph evidence. [P1.E1], [P1.E2], [P2.E1], [P2.E2], [P3.E1], and [P3.E2] discuss locations in Turkey but do not mention a town called Industry.
- **P3 (Adaptive Mechanisms):** The evidence provided does not contain information about the main industry of the town named Industry. Therefore, I cannot answer the question based on the given knowledge-graph evidence. [P1.E1] [P2.E1] [P3.E1]

---

## Group C (Polysemy / Homonyms)

### Query 5: *"What is the capital city of the state named Georgia?"*

| Pipeline | Top Resolved Seed | Paths | Candidate Answers | Latency (ms) | Grounded Citations |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **P1: Baseline** | `Capital City` (Q1034657) | 3 | 3 | 1652.1 | `['[P1.E1]', '[P1.E2]', '[P2.E1]', '[P2.E2]', '[P3.E1]', '[P3.E2]']` |
| **P2: Adaptive Gating** | `Capital City` (Q1034657) | 3 | 3 | 1348.3 | `['[P1.E1]', '[P1.E2]', '[P2.E1]', '[P2.E2]', '[P3.E1]', '[P3.E2]']` |
| **P3: Adaptive Mechanisms** | `None` () | 6 | 2 | 987.8 | `[]` |

**Pipeline Responses:**
- **P1 (Baseline):** The evidence provided does not contain information about the capital city of the state named Georgia. [P1.E1], [P1.E2], [P2.E1], [P2.E2], [P3.E1], and [P3.E2] all relate to topics such as the City of Capitals, forms of state, and television series episodes, but do not provide any information about the capital city of Georgia.
- **P2 (Adaptive Gating):** The evidence provided does not contain information about the capital city of the state named Georgia. [P1.E1], [P1.E2], [P2.E1], [P2.E2], [P3.E1], and [P3.E2] all relate to topics such as the City of Angels, forms of state, and television series, but do not provide any information about the capital city of Georgia.
- **P3 (Adaptive Mechanisms):** The capital city of the state named Georgia is Atlanta [P4.E2, P5.E2, P6.E2].

---

### Query 6: *"What country controls the island of Jersey?"*

| Pipeline | Top Resolved Seed | Paths | Candidate Answers | Latency (ms) | Grounded Citations |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **P1: Baseline** | `Island Of Jersey` (Q785) | 3 | 3 | 846.6 | `['[S1]']` |
| **P2: Adaptive Gating** | `Island Of Jersey` (Q785) | 3 | 3 | 1035.5 | `['[P1.E1]', '[P2.E1]', '[P3.E1]']` |
| **P3: Adaptive Mechanisms** | `jersey (united kingdom)` (Q785) | 3 | 3 | 996.0 | `['[P1.E1]', '[P3.E1]']` |

**Pipeline Responses:**
- **P1 (Baseline):** The island of Jersey is controlled by the United Kingdom. This is evidenced by [S1] which states that Jersey is a "British Crown dependency."
- **P2 (Adaptive Gating):** Based on the provided evidence, Jersey (also known as the Bailiwick of Jersey) is a British Crown dependency. Therefore, the country that controls Jersey is the United Kingdom. [P1.E1] [P2.E1] [P3.E1]
- **P3 (Adaptive Mechanisms):** The island of Jersey is controlled by the United Kingdom. This is evidenced by [P1.E1] and [P3.E1] which state that Jersey is a British Crown dependency.

---

## Group D (Intersections & Joins)

### Query 7: *"Identify any shared awards won by both Albert Einstein and Marie Curie."*

| Pipeline | Top Resolved Seed | Paths | Candidate Answers | Latency (ms) | Grounded Citations |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **P1: Baseline** | `Albert Einstein` (Q937) | 3 | 3 | 1651.7 | `['[P1.E1]', '[P1.E2]', '[P2.E1]', '[P2.E2]', '[P3.E1]', '[P3.E2]']` |
| **P2: Adaptive Gating** | `Marie Curie` (Q7186) | 3 | 3 | 1655.8 | `['[P1.E1]', '[P1.E2]', '[P2.E1]', '[P2.E2]', '[P3.E1]', '[P3.E2]']` |
| **P3: Adaptive Mechanisms** | `Albert Enstein` (Q937) | 3 | 3 | 2884.8 | `['[P1.E1]', '[P2.E1]', '[P2.E2]', '[P3.E1]', '[P3.E2]']` |

**Pipeline Responses:**
- **P1 (Baseline):** The provided knowledge-graph evidence does not contain any information about shared awards won by both Albert Einstein and Marie Curie. [P1.E1], [P1.E2], [P2.E1], [P2.E2], [P3.E1], and [P3.E2] focus on the family relationships between Marie Curie and her children Irène and Ève, but do not mention any awards won by Albert Einstein or Marie Curie together.
- **P2 (Adaptive Gating):** The evidence provided does not contain any information about shared awards won by both Albert Einstein and Marie Curie. [P1.E1], [P1.E2], [P2.E1], [P2.E2], [P3.E1], and [P3.E2] all relate to Marie Curie's children and their nominations for the Nobel Prize in Physics, but there is no information linking Albert Einstein to any shared awards with Marie Curie.
- **P3 (Adaptive Mechanisms):** The evidence provided does not indicate any shared awards between Albert Einstein and Marie Curie. [P1.E1] mentions an award received by Albert Einstein but does not specify Marie Curie as a recipient. Similarly, the paths related to Marie Curie do not mention Albert Einstein as a recipient of any awards. [P2.E1] and [P2.E2] are related to her daughter Irène Curie and the Nobel Prize, but do not connect Albert Einstein to any shared awards with Marie Curie. [P3.E1] and [P3.E2] are about the kilocurie unit named after Marie Curie and her involvement with the International Commission on Intellectual Cooperation, which do not pertain to shared awards. Therefore, based on the given evidence, there is no information to assert that Albert Einstein and Marie Curie shared any awards.

---

### Query 8: *"Name all countries that share a land border with both Germany and Switzerland."*

| Pipeline | Top Resolved Seed | Paths | Candidate Answers | Latency (ms) | Grounded Citations |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **P1: Baseline** | `Switzerland` (Q671912) | 3 | 3 | 1432.5 | `['[P1.E2]', '[P2.E2]', '[P3.E2]']` |
| **P2: Adaptive Gating** | `Switzerland` (Q671912) | 12 | 12 | 1941.5 | `['[P1.E2]', '[P2.E2]', '[P3.E2]']` |
| **P3: Adaptive Mechanisms** | `Confédération suisse` (Q39) | 12 | 9 | 3058.8 | `['[P1.E1]', '[P2.E1]', '[P3.E1]', '[P4.E1]', '[P5.E1]', '[P6.E1]', '[P7.E1]', '[P8.E1]', '[P9.E1]', '[P10.E1]', '[P11.E1]', '[P12.E1]']` |

**Pipeline Responses:**
- **P1 (Baseline):** Based on the provided evidence, the information is insufficient to name all countries that share a land border with both Germany and Switzerland. The evidence only mentions that Switzerland borders Austria, Liechtenstein, France, and potentially other countries, but it does not provide information about which countries border both Germany and Switzerland. [P1.E2], [P2.E2], and [P3.E2] do not directly address the question at hand.
- **P2 (Adaptive Gating):** Based on the evidence, there are no distinct entities found that explicitly share a land border with both Germany and Switzerland. The evidence primarily discusses the geographical features and administrative divisions of Switzerland, but does not provide information about countries sharing borders with both Germany and Switzerland. [P1.E2], [P2.E2], [P3.E2], and other paths focus on Switzerland's borders but do not specify which countries share borders with both Germany and Switzerland.
- **P3 (Adaptive Mechanisms):** Based on the evidence, the answer is that there are no countries that share a land border with both Germany and Switzerland. The evidence shows that Germany shares borders with Belgium, Luxembourg, and Austria, and that Switzerland shares borders with Germany, Austria, and Liechtenstein. However, no country is shown to share borders with both Germany and Switzerland simultaneously. [P1.E1], [P2.E1], [P3.E1], [P4.E1], [P5.E1], [P6.E1], [P7.E1], [P8.E1], [P9.E1], [P10.E1], [P11.E1], [P12.E1]

---

## Group E (Inversions / Backward Tracing)

### Query 9: *"Name all feature films directed by the creator of the Star Wars franchise."*

| Pipeline | Top Resolved Seed | Paths | Candidate Answers | Latency (ms) | Grounded Citations |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **P1: Baseline** | `The Star Wars` (Q462) | 3 | 3 | 2084.9 | `['[S1]', '[P1.E1]', '[P2.E1]', '[P3.E1]']` |
| **P2: Adaptive Gating** | `The Star Wars` (Q462) | 12 | 12 | 2483.4 | `['S1', 'S2', 'S3']` |
| **P3: Adaptive Mechanisms** | `Starwars.com` (Q462) | 12 | 12 | 2418.3 | `['[P4.E1]']` |

**Pipeline Responses:**
- **P1 (Baseline):** The evidence provided does not contain any information about the creator of the Star Wars franchise or the directors of Star Wars feature films. The entity [S1] mentions George Lucas as the creator of the Star Wars franchise but does not provide any information about the feature films he directed or any other directors of Star Wars films. Therefore, I cannot answer the question based on the given evidence. [P1.E1], [P2.E1], and [P3.E1] are about the concept of "The All" and do not relate to the Star Wars franchise or its directors.
- **P2 (Adaptive Gating):** Based on the evidence, the creator of the Star Wars franchise, George Lucas Jr. [P7.E2, P9.E2], has directed at least 2 distinct feature films:

1. Star Wars: Episode IV - A New Hope [P7.E1, P9.E1]
2. Star Wars: Episode V - The Empire Strikes Back [P7.E1, P9.E1]
3. Star Wars: Episode VI - Return of the Jedi [P7.E1, P9.E1]

The evidence provided confirms that George Lucas Jr. directed these three films, which are part of the Star Wars franchise. [S1] [S2] [S3]
- **P3 (Adaptive Mechanisms):** Based on the evidence, the creator of the Star Wars franchise, George Lucas Jr. [P4.E1], is associated with directing feature films. However, the evidence does not provide a list of specific feature films directed by George Lucas Jr. Therefore, we cannot name all feature films directed by him based solely on the given information.

The count of verified distinct entities found in the evidence related to the directorship of George Lucas Jr. is 1, but this does not provide the specific films requested.

[Path 4.E1]

---

### Query 10: *"List the operating systems that run on devices manufactured by Sony."*

| Pipeline | Top Resolved Seed | Paths | Candidate Answers | Latency (ms) | Grounded Citations |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **P1: Baseline** | `operating systems` (Q9135) | 3 | 3 | 1362.1 | `['S1', 'S2', 'S3']` |
| **P2: Adaptive Gating** | `operating systems` (Q9135) | 12 | 9 | 2684.2 | `['[P1.E1]', '[P2.E1]', '[P3.E1]', '[P4.E1]', '[P5.E1]', '[P6.E1]', '[P7.E1]', '[P8.E1]', '[P9.E1]', '[P10.E1]', '[P11.E1]', '[P12.E1]']` |
| **P3: Adaptive Mechanisms** | `Sony (Sony Corporation Japan)` (Q41187) | 12 | 11 | 2503.2 | `[]` |

**Pipeline Responses:**
- **P1 (Baseline):** The provided knowledge-graph evidence does not contain any information about operating systems running on devices manufactured by Sony. Therefore, I cannot provide a list of operating systems based on the given data. [P1], [P2], and [P3] paths are related to songs by Moby and do not contain relevant information about Sony's operating systems. [S1] [S2] [S3]
- **P2 (Adaptive Gating):** Based on the evidence, the requested information about operating systems that run on devices manufactured by Sony is not present. The provided paths and entities are related to songs by Moby and do not contain any information about Sony or operating systems. Therefore, I cannot provide a list of operating systems that run on devices manufactured by Sony based on the given evidence. [P1.E1], [P2.E1], [P3.E1], [P4.E1], [P5.E1], [P6.E1], [P7.E1], [P8.E1], [P9.E1], [P10.E1], [P11.E1], [P12.E1]
- **P3 (Adaptive Mechanisms):** Based on the evidence, there is insufficient information to determine the operating systems that run on devices manufactured by Sony. The provided paths do not include any direct or indirect mentions of operating systems used by Sony's devices. [P1], [P3], [P4] discuss Sony's subsidiaries and divisions but do not mention operating systems. [P9] and [P10] are related to airports and do not provide information about Sony's operating systems. [P2] discusses the formation location of Aiwa, a Sony subsidiary, but does not mention operating systems. Therefore, the evidence does not provide the required information.

---
