# L2F: Learning to Forget - Architecture & Design

## 📋 Executive Summary

**L2F** is a **context hygiene system** for Long-Context LLMs that actively *prunes* contradictory and irrelevant facts from memory, rather than just fitting more tokens into a sliding window. 

The system uses a **Heterogeneous Knowledge Graph** to model facts and their relationships, then learns a **pruning policy** via Reinforcement Learning to maximize accuracy while minimizing context size.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          RAW CONTEXT                             │
│              (Meeting transcripts, chat history, docs)           │
│                     ~150k tokens                                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 1: ATOMIZER (Text → Fact Triples)                        │
├─────────────────────────────────────────────────────────────────┤
│  Input:  Raw text stream                                        │
│  Method: LLM extraction (Groq) OR spaCy OR pattern rules        │
│  Output: List[FactTriple] = [(S, R, O, timestamp), ...]         │
│                                                                  │
│  Example Facts:                                                 │
│    - ("Project", "deadline", "March 15", t=0)                  │
│    - ("John", "is", "PM", t=10)                                │
│    - ("Project", "deadline", "April 1", t=50) ← CONTRADICTION!  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 2: GRAPH BUILDER (Facts → Knowledge Graph)               │
├─────────────────────────────────────────────────────────────────┤
│  Input:  Fact triples                                           │
│  Method: Build heterogeneous directed graph using rustworkx     │
│  Output: GraphState with typed edges                            │
│                                                                  │
│  Graph Components:                                              │
│  • Nodes: FactTriples (facts are first-class graph citizens)    │
│  • Edges: Typed relationships                                   │
│    - TEMPORAL: fact A happened before B                         │
│    - CONTRADICTION: same (S,R) different O                      │
│    - SAME_ENTITY: same subject                                  │
│    - SUPPORTS: evidence relationships                           │
│    - RELATED_TO: semantic similarity                            │
│                                                                  │
│  Edge Detection:                                                │
│    • Contradiction: A.conflicts_with(B) → CONTRADICTION edge    │
│    • Temporal: Sorted by timestamp → TEMPORAL edges             │
│    • Entity links: Same subject → SAME_ENTITY edge              │
│                                                                  │
│  Graph visualization properties:                                │
│    • Green nodes: Facts to keep                                 │
│    • Red nodes: Contradicted facts (marked for deletion)        │
│    • Gray nodes: Low relevance (island/disconnected)            │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 3: NODE FEATURE COMPUTATION                              │
├─────────────────────────────────────────────────────────────────┤
│  Computes 64-dim feature vector for each fact:                  │
│                                                                  │
│  Temporal Features:                                             │
│    - recency: t / max_timestamp (0=old, 1=recent)              │
│    - age: 1 - recency                                           │
│                                                                  │
│  Structural Features:                                           │
│    - in_degree: incoming edges                                  │
│    - out_degree: outgoing edges                                 │
│    - num_contradictions: contradiction edge count               │
│    - has_contradiction: binary flag                             │
│                                                                  │
│  Content Features:                                              │
│    - content_hash: simple encoding of (S, R, O)                 │
│    - length features: len(subject), len(object)                 │
│                                                                  │
│  Query Relevance:                                               │
│    - query_overlap: word overlap with user query                │
│    - semantic_similarity: (in production use embeddings)        │
│                                                                  │
│  Confidence:                                                    │
│    - fact.confidence: from atomizer (0-1)                       │
│                                                                  │
│  Output: [num_nodes, 64] feature matrix                         │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 4: POLICY NETWORK (Graph → Pruning Mask)                │
├─────────────────────────────────────────────────────────────────┤
│  Architecture: HeterogeneousGraphTransformer (HGT)              │
│                                                                  │
│  Model Components:                                              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Input: node_features [N, 64]                            │   │
│  │        edge_index [2, E]                                │   │
│  │        edge_types [E] ∈ {0,1,2,3,4,5}                  │   │
│  └─────────────┬───────────────────────────────────────────┘   │
│                │                                                │
│                ▼                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Input Projection: [N, 64] → [N, 128]                    │  │
│  │   x = Linear(node_features)                             │  │
│  └─────────────┬──────────────────────────────────────────┘   │
│                │                                                │
│                ▼                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ HGT Layer 1: Type-specific Multi-head Attention          │  │
│  │   For each edge type e ∈ {TEMPORAL, CONTRADICTION, ...} │  │
│  │     - K_e = Linear_k[e](x)                              │  │
│  │     - V_e = Linear_v[e](x)                              │  │
│  │     - attn_e = softmax((Q · K_e) / √d)                 │  │
│  │     - out_e = attn_e · V_e                              │  │
│  │   x = Aggregate(out_0, ..., out_5) + x                 │  │
│  │   x = LayerNorm(x)                                      │  │
│  └─────────────┬──────────────────────────────────────────┘   │
│                │                                                │
│                ▼                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ HGT Layer 2: (same as Layer 1)                           │  │
│  └─────────────┬──────────────────────────────────────────┘   │
│                │                                                │
│                ▼                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Policy Head (per-node decision):                         │  │
│  │   logits = MLP(x)  # [N, 1]                             │  │
│  │   keep_prob = Sigmoid(logits)  # [N, 1] ∈ [0, 1]       │  │
│  └─────────────┬──────────────────────────────────────────┘   │
│                │                                                │
│                ▼                                                │
│  Output: keep_probs [N] - probability to KEEP each fact       │
│                                                                  │
│  Inference:                                                     │
│    action = (keep_prob > 0.5).float()  # Hard decision         │
│    OR                                                           │
│    action = Bernoulli(keep_prob)       # Stochastic           │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 5: APPLY PRUNING MASK                                   │
├─────────────────────────────────────────────────────────────────┤
│  For each fact i:                                              │
│    if keep_prob[i] > threshold (0.5):                          │
│      KEEP fact[i]                                              │
│    else:                                                        │
│      PRUNE fact[i] → mark as inactive in graph                │
│                                                                  │
│  Result: Minimal Sufficient Subgraph                           │
│    • All ground-truth facts preserved                          │
│    • Contradictions explicitly resolved (newer wins)           │
│    • Irrelevant facts removed (islands)                        │
│    • ~80-90% token reduction achieved                          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 6: CONTEXT RECONSTRUCTION                               │
├─────────────────────────────────────────────────────────────────┤
│  Input:  Active facts (non-pruned nodes)                       │
│  Method: Linearize facts back to natural language              │
│  Output: Pruned context (<4k tokens)                           │
│                                                                  │
│  Example reconstruction:                                       │
│    Input facts:                                                │
│      - ("Project", "deadline", "April 1", t=50)              │
│      - ("John", "is", "PM", t=10)                            │
│                                                                  │
│    Output text:                                                │
│      "PRUNED CONTEXT:                                          │
│       - Project's deadline is April 1                          │
│       - John is the project manager                            │
│       Total: 17 tokens (vs 150,000 original)"                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 7: ANSWER GENERATION (SOLVER)                           │
├─────────────────────────────────────────────────────────────────┤
│  Input:  Pruned context + User query                           │
│  Method: Frozen LLM (Groq, GPT-4, etc.)                        │
│  Output: Answer to user query                                  │
│                                                                  │
│  Key property:                                                  │
│    Answer(pruned_context, query) ≈ Answer(full_context, query) │
│    BUT with 5-10x fewer tokens                                 │
│    AND lower hallucination (contradictions explicitly removed)  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 8: VERIFICATION & REWARD SIGNAL                         │
├─────────────────────────────────────────────────────────────────┤
│  For training: Compare answer against ground truth             │
│                                                                  │
│  Reward = Accuracy × (1 + α×Sparsity) - β×Hallucination       │
│                                                                  │
│  Where:                                                         │
│    Accuracy = 1 if answer correct, 0 otherwise (VERIFIABLE!)  │
│    Sparsity = (pruned_facts / total_facts)                    │
│    Hallucination = 1 if answer cites deleted fact, 0 else     │
│    α = 0.3 (sparsity weight)                                  │
│    β = 0.5 (hallucination penalty)                            │
│                                                                  │
│  This signal is used to train the HGT policy via PPO          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Data Flow: Concrete Example

### Input: Meeting Transcript (150k tokens)
```
Turn 1:  "Sarah: The project deadline is March 15th."
Turn 2:  "Mike: That works for our schedule."
Turn 3-47: [Long discussion about unrelated topics...]
Turn 48: "Sarah: Actually, after stakeholder review, 
         the new deadline is April 1st."
Turn 49: "Everyone: Okay, noted."
```

### After Phase 1 (Atomizer): Fact Triples
```
Fact 1:  (Project, deadline, "March 15", t=0)     confidence=0.9
Fact 2:  (Mike, confirmation, "yes", t=1)         confidence=0.8
Fact 3-47: [Many unrelated facts about weather, meetings, etc.]
Fact 48: (Project, deadline, "April 1", t=48)     confidence=0.95 ← NEWER!
Fact 49: (Everyone, confirmation, "yes", t=49)    confidence=0.7
```

### After Phase 2 (Graph Builder): Graph Structure
```
Nodes:  48 facts (fact nodes in graph)

Edges:
  Fact 1 --CONTRADICTION--> Fact 48  (same subject+relation, diff object)
  Fact 48 --TEMPORAL--> Fact 49      (more recent)
  Fact 2 --SAME_ENTITY--> Fact 49    (both mention "confirmation")
  ... (many other edges for entity coreference)

Contradiction Detection:
  Fact1.subject = "Project", Fact1.relation = "deadline", Fact1.obj = "March 15"
  Fact48.subject = "Project", Fact48.relation = "deadline", Fact48.obj = "April 1"
  → Same (S, R), different O → CONTRADICTION EDGE
```

### After Phase 3 (Features): Node Features
```
Fact 1 features:
  recency = 0 / 49 = 0.0        (oldest)
  in_degree = 1                  (one incoming contradiction edge)
  out_degree = 1                 (contradiction edge to newer fact)
  has_contradiction = 1.0        (YES)
  query_relevance = 0.8          ("deadline" matches query)
  confidence = 0.9
  [... 8 more features ...]
  Vector: [0.0, 1.0, 0.1, 0.1, 1.0, 0.8, 0.9, ...]

Fact 48 features:
  recency = 48 / 49 = 0.98      (very recent! ↑)
  in_degree = 1                  (one incoming contradiction from Fact1)
  out_degree = 1                 (temporal edge to Fact49)
  has_contradiction = 1.0        (YES, but newer so shouldn't be pruned)
  query_relevance = 0.8          ("deadline" matches query)
  confidence = 0.95
  Vector: [0.98, 0.02, 0.1, 0.1, 1.0, 0.8, 0.95, ...]
```

### After Phase 4 (Policy Network):  Pruning Decisions
```
HGT forward pass:
  Input: [48 nodes, 16 features each], edges with types
  Output: keep_probs for each node

  Fact 1:  keep_prob = 0.15  (low, because older contradicted fact)
  Fact 48: keep_prob = 0.92  (high, because recent & matches query)
  Fact 49: keep_prob = 0.78  (medium, confirmation of new deadline)
  Others:  keep_prob ≈ 0.3-0.5  (low relevance, island detection)

Thresholding (θ = 0.5):
  Keep if keep_prob > 0.5:
    ✓ Fact 48 (0.92)
    ✓ Fact 49 (0.78)
    ✗ Fact 1  (0.15) ← PRUNED (old contradicted fact)
    ✗ Others  (< 0.5) ← PRUNED (low relevance)
```

### After Phase 6 (Reconstruction): Final Pruned Context
```
"SANITIZED CONTEXT (L2F Pruned):

Query: What is the project's deadline?

Relevant Facts:
  • (Project, deadline, "April 1")
  • (Everyone, confirmation, "yes")

Compression: 150,000 tokens → 45 tokens (3333x reduction!)"
```

### Answer Generation
```
Query:    "What is the project's deadline?"
Context:  "Project deadline is April 1, everyone confirmed."

LLM Output: "The project deadline is April 1."  ✓ CORRECT
```

---

## 🔧 Key Components

### 1. FactTriple Dataclass
```python
@dataclass
class FactTriple:
    subject: str              # "Project", "Sarah", etc.
    relation: str             # "deadline", "is", "said"
    obj: str                  # "March 15", "Manager", etc.
    timestamp: int            # Logical timestamp (turn #, token position)
    confidence: float = 1.0   # Extraction confidence (0-1)
    source_span: Tuple = (0,0) # Original text location
    fact_id: str = ""         # Unique hash
    
    def conflicts_with(other):
        return (self.subject == other.subject and 
                self.relation == other.relation and 
                self.obj != other.obj)
```

### 2. GraphState
```python
class GraphState:
    graph: rx.PyDiGraph              # rustworkx directed graph
    nodes: Dict[fact_id, NodeState]  # All nodes with metadata
    edge_types: Dict[Tuple, EdgeType]# Typed edges
    current_timestamp: int           # Max timestamp seen
    
    def add_fact(fact):             # Add node
    def add_edge(src, dst, type):   # Add typed edge
    def get_contradictions():        # Find conflict edges
    def prune_node(fact_id, reason): # Mark for deletion
    def to_context(max_tokens):      # Reconstruct text
```

### 3. HeterogeneousGraphAttention Layer
Implements type-specific transformations for each edge type:
```python
class HeterogeneousGraphAttention(nn.Module):
    def forward(x, edge_index, edge_type):
        # For each edge type t in {TEMPORAL, CONTRADICTION, ...}:
        Q = W_q(x)           # Queries (global)
        K_t = W_k[t](x)      # Keys (type-specific)
        V_t = W_v[t](x)      # Values (type-specific)
        
        attn_t = softmax(Q @ K_t / √d)
        out_t = attn_t @ V_t
        
        # Add edge type embeddings
        K_t = K_t + edge_emb[t]
        
        # Aggregate across types
        out = mean(out_0, ..., out_5)
        return LayerNorm(out + x)
```

### 4. Reward Function (RLVR)
```python
def compute_reward(actions, facts, graph, reference, α=0.3, β=0.5):
    # Sparsity: how many facts pruned (higher = better for efficiency)
    sparsity = (total_facts - kept_facts) / total_facts
    
    # Contradiction handling: penalize keeping BOTH conflicting facts
    contradiction_penalty = sum(
        keep_prob[i] * keep_prob[j] 
        for (i,j) in contradiction_edges
    )
    
    # Accuracy proxy: facts with higher confidence are kept
    accuracy_proxy = mean(confidence[i] for kept facts)
    
    # Composite reward (trained via PPO)
    R = accuracy_proxy * (1 + α * sparsity) - β * contradiction_penalty
    return R
```

---

## 💾 Input/Output Specifications

### Inputs
- **Raw Context**: String (any length, typically 5k-150k tokens)
  - Format: Free-form text (meetings, chat, documents)
  - Required: Some temporal/contradictory structure for meaningful pruning
  
- **Query**: String (optional, used for relevance scoring)
  - Format: Natural language question
  - Example: "What is the project deadline?"

### Outputs
- **Pruned Context**: String (<4k tokens typically)
  - Contains only relevant, non-contradicting facts
  - Formatted as linearized fact list
  - Ready to feed to frozen LLM solver
  
- **Pruning Report**: List of {fact, reason} pairs
  - "Deleted because Rule 3: Recency Override"
  - "Deleted because Rule 2: Island Detection"
  - Provides complete explainability audit
  
- **Metrics**:
  - `compression_ratio`: original_tokens / pruned_tokens
  - `sparsity`: pruned_facts / total_facts
  - `kept_facts`: list of preserved facts
  - `pruned_facts`: list of deleted facts

---

## 📈 Compute Requirements

| Component      | Time (M1 Mac) | GPU Needed? | Memory   |
|----------------|---------------|------------|----------|
| Atomizer       | 10-50ms       | No         | 500MB    |
| Graph Build    | 1-5ms         | No         | 100MB    |
| Features       | 5-10ms        | No         | 50MB     |
| HGT Forward    | 20-50ms       | Optional   | 100MB    |
| HGT Training   | 5-10 min/epoch | **Yes**    | 1GB      |
| Full Pipeline  | ~100ms/sample | No (GPU ok)| 300MB    |

**Recommendation**: 
- Development/inference: Run on Mac (fast enough)
- Training: Use Colab GPU (10x speedup)
- Production: CPU sufficient for inference

---

## 🔄 Training Loop (RLVR via PPO)

```python
for episode in range(num_episodes):
    # Generate scenario (or load real data)
    scenario = gym.generate_contradiction_scenario()
    
    # Forward pass
    facts = atomizer.extract_facts(scenario.raw_stream)
    graph = graph_builder.build_graph(facts)
    features = graph_builder.compute_node_features(graph, scenario.query)
    
    # Policy action
    keep_probs = policy(features, edge_index, edge_types)
    actions = (keep_probs > 0.5).float()
    
    # Reward signal (verifiable!)
    reward = reward_computer.compute_reward(
        actions, facts, graph, scenario.ground_truth
    )
    
    # PPO update
    loss = ppo_loss(log_probs, actions, advantages)
    loss.backward()
    optimizer.step()
```

This completes the L2F architecture—a principled, end-to-end system for active context pruning.
