# L2F: Core Logic & Algorithms

## 🧠 Fundamental Insight

**The Core Problem**: Long-context LLMs hit an accuracy ceiling around 4k tokens due to:
1. **Attention dilution**: Model can't focus on relevant facts
2. **Lost-in-the-middle**: Earlier facts fade from attention
3. **Context rot**: Contradictory old facts confuse the model

**The L2F Solution**: Don't just *fit* more—*remove* what's logically obsolete.

Instead of:
```
User Query + [ALL 150k tokens] → Answer
                                  ↑ Noisy, contradictory, attention diluted
```

Do this:
```
User Query + [PRUNED 2k tokens] → Answer
                                  ↑ Clean, coherent, focused
```

---

## 🔍 Contradiction Detection Logic

### The Three Rules of Forgetfulness

#### **Rule 1: Recency Override (Temporal Logic)**
**When**: Two facts have the same (Subject, Relation) but different Objects.
**What**: The newer fact supersedes the older one.

```python
def conflicts_with(fact_a, fact_b):
    return (fact_a.subject == fact_b.subject and 
            fact_a.relation == fact_b.relation and 
            fact_a.obj != fact_b.obj)

# Example
Fact A (t=5):  ("Project", "deadline", "March 15")
Fact B (t=48): ("Project", "deadline", "April 1")

# They conflict! Fact B wins because t=48 > t=5
```

**Implementation**:
```python
for (i, j) in contradiction_edges:
    fact_i, fact_j = facts[i], facts[j]
    if fact_i.timestamp < fact_j.timestamp:
        # Keep j, prune i
        actions[i] = 0  # Don't keep older contradicted fact
        actions[j] = 1  # Keep newer fact
    else:
        actions[j] = 0
        actions[i] = 1
```

**Why it works**: Meeting transcripts naturally have this structure:
- "The deadline is X"
- [discussion]
- "Actually, the deadline is Y"

Rule 1 automatically cleans these up!

---

#### **Rule 2: Island Detection (Graph Connectivity)**
**When**: A fact is disconnected from the query-relevant component.
**What**: Prune isolated subgraphs (noise).

```python
def get_disconnected_subgraphs(graph):
    """Find weakly connected components in graph."""
    components = rx.weakly_connected_components(graph)
    return [set(comp) for comp in components]

# Example
Facts:
  1. (Project, deadline, "April 1")       [RELEVANT]
  2. (Sarah, likes, "coffee")              [NOISE]
  3. (Sarah, breakfast, "toast")           [NOISE]
  4. (Everyone, confirmed, "deadline")     [RELEVANT]

Graph structure:
  Component 1: {Fact1 -- (temporal) -- Fact4}  [MAIN: 2 nodes]
  Component 2: {Fact2 -- (same_entity) -- Fact3} [ISLAND: 2 nodes]

# Component 1 is larger (main query) → keep
# Component 2 is small + low relevance → prune (island)
```

**Implementation**:
```python
components = get_disconnected_subgraphs(graph)
main_component = max(components, key=len)  # Largest component

for component in components:
    if component != main_component and len(component) < 10:
        # Small disconnected component = island
        for fact_id in component:
            if relevance[fact_id] < 0.1:  # Also check relevance
                prune_node(fact_id, "Island Detection")
```

**Why it works**: Irrelevant distractors (weather, random comments) get "stuck" on their own connected component, naturally isolated from the query-relevant facts.

---

#### **Rule 3: Relevance Scoring (Semantic)**
**When**: A fact has very low semantic overlap with the query.
**What**: Lower keep probability.

```python
def compute_relevance(fact, query):
    """Word overlap-based relevance (simple version)."""
    query_words = set(query.lower().split())
    fact_words = set(f"{fact.subject} {fact.obj}".lower().split())
    
    overlap = len(query_words & fact_words)
    relevance = overlap / max(len(query_words), 1)
    
    return relevance

# Example
Query: "What is the project deadline?"
query_words = {"what", "is", "the", "project", "deadline"}

Fact A: ("Project", "deadline", "April 1")
fact_words = {"project", "deadline", "april", "1"}
overlap = {"project", "deadline"}
relevance = 2 / 5 = 0.4 ✓ HIGH

Fact B: ("Sarah", "likes", "coffee")
fact_words = {"sarah", "likes", "coffee"}
overlap = {} (empty)
relevance = 0 / 5 = 0.0 ✗ LOW
```

**In HGT Network**: This becomes a learned feature
```python
relevance_feature = len(query_words & fact_words) / max(len(query_words), 1)
node_features[i] = [..., relevance_feature, ...]

# Policy network learns: "if relevance < 0.1, set keep_prob low"
```

---

## 🎯 Graph-Based Fact Modeling

### Why Graphs, Not Just Lists?

**Naive approach**: Just delete contradicted facts from a list.
```python
facts = [
    ("Project", "deadline", "March 15"),
    ("Project", "deadline", "April 1")
]
# Delete fact[0], keep fact[1]
# Problem: What if there are 3+ versions? Chains of updates?
```

**Graph approach**: Model dependencies explicitly.
```
Fact1 (Mar 15) --CONTRADICTION--> Fact2 (Apr 1) --CONTRADICTION--> Fact3 (May 1)

# Policy learns: "chain of contradictions → keep only the last one"
# This is a PATTERN the network can learn!
```

### Edge Types & Their Semantics

| Edge Type | When | Logic |
|-----------|------|-------|
| **TEMPORAL** | fact_i.timestamp < fact_j.timestamp | Sequential ordering, causality |
| **CONTRADICTION** | Same (S,R), different O | Update/invalidation |
| **SAME_ENTITY** | fact_i.subject == fact_j.subject | Entity coreference, scope |
| **SUPPORTS** | fact_i provides evidence for fact_j | Epistemic justification |
| **RELATED_TO** | Semantic similarity | Cross-fact relevance |

**HGT Mechanism**: The network uses edge-type-specific transformations.
```python
for edge_type in {TEMPORAL, CONTRADICTION, SUPPORTS, ...}:
    # Type-specific attention for this edge type
    K_type = Linear_k[edge_type](x)  # Different for each type
    V_type = Linear_v[edge_type](x)
    
    attn_type = softmax(Q @ K_type)
    out_type = attn_type @ V_type
    
# Each edge type can learn different propagation rules
# → TEMPORAL might propagate "recency signal"
# → CONTRADICTION might propagate "conflict signal"
```

---

## 🤖 Heterogeneous Graph Transformer (HGT) Logic

### Why HGT Instead of Regular GNN?

**Regular GNN**: All edges treated the same way.
```
h_i = σ(W_0 h_i + Σ W_1 h_j  for all neighbors j)
         ^base      ^same for all edge types
```

**Problem**: Can't distinguish "TEMPORAL means newer is better" from "CONTRADICTION means old is worse"

---

### HGT: Type-Aware Aggregation

```python
class HeterogeneousGraphAttention:
    def forward(x, edge_index, edge_types):
        Q = W_q(x)  # Global queries [N, d]
        
        # Type-specific keys and values
        out = torch.zeros_like(x)
        
        for edge_type t in {0, 1, 2, 3, 4, 5}:
            mask = (edge_types == t)
            edges_t = edge_index[:, mask]
            
            if edges_t.shape[1] == 0:
                continue  # No edges of this type
            
            # Type-specific transformation
            K_t = W_k[t](x)        # [N, d]
            V_t = W_v[t](x)        # [N, d]
            
            # Add type embedding (learnable prior)
            K_t = K_t + embed[t]   # [N, d]
            
            # Compute attention on this edge type's edges
            src, dst = edges_t[0], edges_t[1]
            q_i = Q[dst]          # [E_t, d]
            k_j = K_t[src]        # [E_t, d]
            v_j = V_t[src]        # [E_t, d]
            
            # Attention scores for this edge type
            attn = (q_i * k_j).sum(dim=-1) / √d  # [E_t]
            attn = softmax(attn)  # [E_t]
            
            # Aggregate: for each destination node, sum messages from sources
            messages = attn.unsqueeze(-1) * v_j  # [E_t, d]
            out.index_add_(0, dst, messages)     # Add to output
        
        return LayerNorm(out + x)
```

**Key insight**: Different edge types can learn different importance weights.

Example learning dynamics:
```
TEMPORAL edges: Learn "newer is better" → high attention to recent facts
CONTRADICTION: Learn "prefer newer value" → suppress older contradicted facts
SAME_ENTITY: Learn "track entity through mentions" → coreference resolution
```

---

## 🏆 Reward Function: RLVR (Reinforcement Learning with Verifiable Rewards)

### The Three Components of the Reward

```python
R = A × (1 + α×S) - β×H

where:
  A = Accuracy (1 if correct answer to query, 0 otherwise)
  S = Sparsity (pruned_facts / total_facts)
  H = Hallucination rate (did answer cite deleted facts?)
  α = 0.3 (sparsity weight)
  β = 0.5 (hallucination penalty)
```

#### **Component 1: Accuracy (Verifiable)**
```python
def check_accuracy(llm_answer, ground_truth):
    """
    Binary check: did the LLM produce correct answer?
    No proxy metrics—REAL VERIFICATION.
    """
    return 1.0 if ground_truth.lower() in llm_answer.lower() else 0.0
```

**Why it matters**: 
- Most RL agents use proxy rewards (perplexity, loss)
- L2F uses ACTUAL correctness as the reward signal
- This ensures the agent learns the right objective

#### **Component 2: Sparsity (Efficiency)**
```python
def compute_sparsity(actions, total_facts):
    """Higher sparsity = more aggressive pruning = more efficient."""
    kept_facts = actions.sum().item()
    sparsity = 1 - (kept_facts / total_facts)
    return sparsity

# Example
total_facts = 100
kept_facts = 30
pruned_facts = 70
sparsity = 70 / 100 = 0.7  (70% pruned)

# With α=0.3 sparsity bonus:
reward_boost = (1 + 0.3 * 0.7) = 1.21  (21% boost)
```

**The trade-off**: 
- High sparsity = efficient (fewer tokens) but risky (might prune important facts)
- α controls how much we value efficiency vs. accuracy
- α=0.3 means: "We care about compression, but not as much as correctness"

#### **Component 3: Hallucination Penalty (Integrity)**
```python
def detect_hallucination(llm_answer, pruned_facts):
    """
    Check if the answer cites facts that were pruned.
    These are "hallucinations" (invented info, or forgotten info being made up).
    """
    hallucinations = []
    for fact in pruned_facts:
        # Extract key values from fact
        obj_value = fact.split(",")[-1].strip()
        
        if obj_value.lower() in llm_answer.lower():
            hallucinations.append(fact)
    
    hallucination_rate = len(hallucinations) / max(len(pruned_facts), 1)
    return hallucination_rate

# Example
pruned_facts = [
    "Project, deadline, March 15",
    "John, status, inactive",
    "Team, size, 5"
]
llm_answer = "The project deadline was March 15 and team has 5 members."

# "March 15" and "5" were pruned but cited in answer → hallucination!
hallucination_rate = 2 / 3 = 0.67

reward_penalty = β × hallucination_rate = 0.5 × 0.67 = 0.33
```

---

### Complete Reward Calculation Example

```python
Scenario:
  Total facts: 50
  Pruned facts: 40 (sparsity = 0.8)
  Kept facts: 10
  
Policy decision: actions = [0, 0, 1, 0, 1, ...]  # keep_prob > 0.5
  
LLM generates answer: "The project deadline is April 1st, and John confirmed."

Ground truth: "April 1st"

Evaluation:
  A (Accuracy): "April 1st" in answer? YES → A = 1.0
  S (Sparsity): 40 / 50 = 0.8
  H (Hallucination): Answer mentions "John" but John was pruned → H = 0.5
  
Final Reward:
  R = 1.0 × (1 + 0.3 × 0.8) - 0.5 × 0.5
    = 1.0 × 1.24 - 0.25
    = 1.24 - 0.25
    = 0.99  ✓ Excellent!
    
Interpretation:
  - Got the answer right (A=1.0) ✓
  - Aggressively pruned context (S=0.8) ✓
  - But hallucinated about John (H=0.5) ✗
  - Net: Great reward (0.99) but penalized for hallucination
```

---

## 🎓 Training Algorithm: PPO with Graph State

### Proximal Policy Optimization (PPO)

L2F uses **PPO** instead of other RL algorithms because:
1. **Stable training**: Doesn't overshoot policy updates
2. **Sample efficient**: Works well with limited scenarios
3. **On-policy**: Learns from its own rollouts (good for graph tasks)

```python
def ppo_update(policy, optimizer, experiences, config):
    states, actions, old_log_probs, returns, advantages = experiences
    
    for epoch in range(config.num_epochs):
        # Forward pass with new policy
        new_log_probs, new_values = policy(states)
        
        # PPO ratio
        ratio = exp(new_log_probs - old_log_probs)
        
        # Clipped objective
        surr1 = ratio * advantages
        surr2 = clip(ratio, 1-ε, 1+ε) * advantages
        
        policy_loss = -min(surr1, surr2).mean()
        value_loss = (returns - new_values).pow(2).mean()
        entropy = -(pi * log(pi)).mean()
        
        total_loss = (
            policy_loss + 
            config.value_coef * value_loss - 
            config.entropy_coef * entropy
        )
        
        optimizer.zero_grad()
        total_loss.backward()
        clip_grad_norm_(policy.parameters(), config.max_grad_norm)
        optimizer.step()
```

---

### Data Collection: Generate Scenarios

Two sources of training data:

#### **Option 1: Synthetic (ConflictGym)**
```python
class ConflictGym:
    """Generate infinite training scenarios with ground truth."""
    
    def generate_contradiction_scenario(self):
        """
        Scenario:
          "X is Blue" → [noise] → "UPDATE: X is Red"
        
        Ground truth: Keep only "X is Red"
        Reward = 1.0 if X kept AND Red kept AND Blue pruned
        """
        entity = sample(ENTITIES)
        old_value = sample(COLORS)
        new_value = sample([c for c in COLORS if c != old_value])
        
        # Build raw stream
        stream = f"""
        {entity}'s attribute is {old_value}.
        [noise padding]
        UPDATE: {entity}'s attribute is now {new_value}.
        """
        
        return SyntheticScenario(
            raw_stream=stream,
            facts=[
                Fact(entity, "attribute", old_value, t=0),
                Fact(entity, "attribute", new_value, t=100)
            ],
            ground_truth=[new_value],
            query=f"What is {entity}'s attribute?"
        )
```

**Why synthetic is useful**:
- Infinite supply of training data
- Ground truth is 100% certain
- Can control difficulty (add more contradictions, noise)
- Fast iteration during development

#### **Option 2: Real Data (QMSum)**
```python
class QMSumAdapter:
    """Adapt real meeting transcripts for L2F."""
    
    def adapt(qmsum_sample):
        """
        QMSum has real contradictions naturally:
          "The deadline is March 15"
          [discussion]
          "Actually, it's April 1"
        
        Ground truth: References final decision
        """
        transcript = qmsum_sample['input']
        ground_truth = qmsum_sample['output']  # Final summary
        
        # Extract facts from transcript
        facts = atomizer.extract_facts(transcript)
        
        # Which facts are referenced in ground truth?
        relevant_facts = [f for f in facts if f.obj in ground_truth]
        
        return L2FSample(
            raw_stream=transcript,
            facts=facts,
            ground_truth_facts=relevant_facts,  # Proxy for what to keep
            ground_truth_answer=ground_truth,
            query="Summarize key decisions"
        )
```

**Why real data matters**:
- Validates that synthetic patterns transfer
- Tests zero-shot generalization
- Measures actual LLM accuracy improvement

---

## 🔗 End-to-End Training Loop

```python
for episode in range(num_episodes):
    # === SAMPLE GENERATION ===
    if random() < 0.5:
        scenario = gym.generate_batch(1)[0]  # Synthetic 50%
    else:
        scenario = qmsum_adapter.adapt(qmsum_data.sample())  # Real 50%
    
    # === FORWARD PASS ===
    facts = atomizer.extract_facts(scenario.raw_stream)
    graph = graph_builder.build_graph(facts)
    features = compute_node_features(graph, scenario.query)
    
    keep_probs, value = policy(features, edge_index, edge_types)
    actions = Bernoulli(keep_probs).sample()  # Stochastic sampling
    
    # === REWARD SIGNAL ===
    reward = reward_computer.compute_reward(
        actions=actions,
        facts=facts,
        graph=graph,
        reference=scenario.ground_truth_answer
    )
    
    # === STORE EXPERIENCE ===
    experience_buffer.append({
        'features': features,
        'actions': actions,
        'log_probs': log_probs,
        'value': value,
        'reward': reward
    })
    
    # === BATCH UPDATE (every N episodes) ===
    if (episode + 1) % batch_size == 0:
        # Compute advantages using GAE
        advantages = compute_gae(
            experience_buffer,
            gamma=0.99,
            lambda=0.95
        )
        
        # PPO update
        for epoch in range(num_ppo_epochs):
            ppo_update(policy, optimizer, advantages)
        
        # Clear buffer
        experience_buffer.clear()
```

---

## 🎯 Summary of Core Logic

| Concept | Logic | Implementation |
|---------|-------|-----------------|
| **Contradiction** | Same (S,R), diff O | `fact_a.conflicts_with(fact_b)` |
| **Recency** | Newer fact overrides older | Timestamp comparison + edge priority |
| **Island Detection** | Disconnected small components = noise | Graph connectivity analysis |
| **Relevance** | Query word overlap | Set intersection |
| **Graph** | Model fact relationships | rustworkx DiGraph with typed edges |
| **HGT** | Learn edge-type-specific propagation | Type-specific K, V transformations |
| **Reward** | Verifiable + sparse + no hallucination | A × (1 + αS) - βH |
| **Training** | PPO on graph states | sample → forward → compute reward → update |

This is the complete logic behind L2F's ability to actively forget while preserving correctness.
