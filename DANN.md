# Agent Neural Network (ANN 2.0) — Formal Specification

**Version**: 1.0
**Date**: 2026-02-18
**Classification**: Multi-Agent Cognitive System
**Status**: Theoretical Framework + Production Architecture

---

## Table of Contents

1. [Strict Mathematical Model](#1-strict-mathematical-model)
2. [Decentralization and Distribution](#2-decentralization-and-distribution)
3. [Evolutionary Mechanism](#3-evolutionary-mechanism)
4. [Agent Interaction Protocol](#4-agent-interaction-protocol)
5. [Scientific Hypothesis](#5-scientific-hypothesis)
6. [Computational Complexity](#6-computational-complexity)
7. [Production Architecture](#7-production-architecture)

---

# 1. Strict Mathematical Model

## 1.1 System Definition

An **Agent Neural Network** (ANN) is a 9-tuple:

```
ANN = (A, G, Σ, M, Φ, Ψ, Ω, R, C)
```

Where:

| Symbol | Name | Domain | Description |
|--------|------|--------|-------------|
| **A** | Agent set | `{a₁, ..., aₙ}` | Autonomous agent-neurons |
| **G** | Communication graph | `(A, E, W)` | Dynamic weighted directed graph |
| **Σ** | Signal protocol | `Σ: A × A → Msg` | Message passing mechanism |
| **M** | Memory architecture | `M_ε × M_σ × M_π × M_α` | 4-tier distributed memory |
| **Φ** | Input encoder | `Φ: Input → P(A)` | Maps input to agent activation set |
| **Ψ** | Trust function | `Ψ: A × A × T → [0, 1]` | Time-dependent trust/reputation |
| **Ω** | Evolution operator | `Ω: G_t → G_{t+1}` | Topology evolution |
| **R** | Reward function | `R: A × S × Action → ℝ` | Agent utility |
| **C** | Constraint set | `C = {c_energy, c_time, c_complexity}` | Resource constraints |

## 1.2 Agent Formal Model

Each agent `aᵢ ∈ A` is a 7-tuple:

```
aᵢ = (πᵢ, Mᵢ, Tᵢ, Rᵢ, Sᵢ, Lᵢ, κᵢ)
```

Where:

- **πᵢ : Sᵢ × Mᵢ → P(Actionᵢ)** — policy (stochastic mapping from state × memory to action distribution)
- **Mᵢ = (mᵢᵉ, mᵢˢ, mᵢᵖ)** — local memory triple:
  - `mᵢᵉ ∈ ℝᵈ` — episodic (recent context, fast-decaying)
  - `mᵢˢ ∈ ℝᵏ` — semantic (persistent knowledge embeddings)
  - `mᵢᵖ ∈ Graph` — procedural (action patterns as subgraph)
- **Tᵢ ⊆ T_global** — tool subset available to agent
- **Rᵢ : S × Action × S' → ℝ** — local reward function
- **Sᵢ ∈ S** — internal state (belief, intent, plan)
- **Lᵢ ∈ {L₁, L₂, L₃, L₄}** — cognitive level (reactive, cognitive, metacognitive, emergent-hub)
- **κᵢ ∈ [0, 1]** — confidence score

## 1.3 State Space

The global state of ANN at time `t`:

```
S(t) = (G(t), {Sᵢ(t)}ᵢ₌₁ⁿ, {Mᵢ(t)}ᵢ₌₁ⁿ, Mshared(t), Ψ(t))
```

State transition:

```
S(t+1) = F(S(t), {πᵢ}ᵢ₌₁ⁿ, Input(t), Noise(t))
```

Where `F` is the global transition function that composes:
1. Input encoding: `Φ(Input(t)) → activated ⊆ A`
2. Signal propagation: `∀ (i,j) ∈ E: msgᵢⱼ = Σ(aᵢ, aⱼ)`
3. Agent computation: `∀ aᵢ: (Oᵢ, Sᵢ') = πᵢ(Sᵢ, Mᵢ, {msgⱼᵢ})`
4. Memory update: `Mᵢ' = UpdateMem(Mᵢ, Oᵢ, {msgⱼᵢ})`
5. Trust update: `Ψ'(i,j) = UpdateTrust(Ψ(i,j), quality(Oᵢ))`
6. Topology evolution: `G' = Ω(G, Ψ', performance)`

## 1.4 Communication Graph

The graph `G = (A, E, W)` where:

```
E ⊆ A × A                          # directed edges (communication channels)
W : E → ℝ⁺                         # edge weights
W(i,j) = Ψ(aᵢ, aⱼ, t) · ρ(i,j)  # weight = trust × relevance
```

**Adjacency matrix**: `Aᵢⱼ = W(i,j)` if `(i,j) ∈ E`, else `0`

**Graph Laplacian**: `L = D - A`, where `Dᵢᵢ = Σⱼ Aᵢⱼ`

**Consensus dynamics** (for distributed agreement):

```
x(t+1) = (I - εL)x(t)
```

Converges to average consensus when `0 < ε < 1/d_max` and `G` is strongly connected.

## 1.5 Trust Dynamics

Trust between agents follows a dual-level model (inspired by RepuNet):

**Direct trust** (from interaction):

```
Ψ_direct(i,j,t) = (1 - λ)·Ψ_direct(i,j,t-1) + λ·quality(Oⱼ, t)
```

Where `λ ∈ (0, 1)` is learning rate and `quality: Output → [0,1]`.

**Reputation** (from gossip):

```
Ψ_rep(j,t) = Σᵢ wᵢ · Ψ_direct(i,j,t) / Σᵢ wᵢ
```

Where `wᵢ = Ψ_direct(observer, i, t)` — trust-weighted aggregation.

**Combined trust**:

```
Ψ(i,j,t) = α·Ψ_direct(i,j,t) + (1-α)·Ψ_rep(j,t)
```

With `α` increasing as agent `i` has more direct interactions with `j`.

## 1.6 Signal Propagation

Unlike classical NN forward pass, ANN uses **asynchronous message passing**:

```
For each active agent aᵢ:
  1. Receive: inbox_i = {msg_{j→i} | (j,i) ∈ E}
  2. Aggregate: context_i = Aggregate(inbox_i, M_i)
  3. Reason: (decision_i, confidence_i) = π_i(S_i, context_i)
  4. Act:
     if confidence_i ≥ θ_i:
       O_i = Execute(decision_i, T_i)
     else:
       O_i = Escalate(decision_i, parent(i))
  5. Send: ∀ j ∈ neighbors_out(i): send(msg_{i→j} = encode(O_i))
```

**Aggregation function** (attention-weighted):

```
context_i = Σⱼ α_ij · encode(msg_{j→i})

where α_ij = softmax_j(score(S_i, msg_{j→i}))
      score(q, k) = q^T W_score k / √d
```

## 1.7 Global Optimization Objective

The ANN optimizes:

```
max_{π₁,...,πₙ,G}  J(ANN) = Σᵢ₌₁ⁿ E[Σₜ₌₀^∞ γᵗ · Rᵢ(Sᵢ(t), Oᵢ(t))]

subject to:
  Σᵢ cost(aᵢ) ≤ Budget_energy          (energy constraint)
  max_i latency(aᵢ) ≤ T_max            (time constraint)
  |A| ≤ N_max                           (scale constraint)
  ∀ i: Ψ(i, ·) ≥ θ_trust               (trust constraint)
```

## 1.8 Comparison with Classical NN

| Concept | Classical NN | ANN |
|---------|-------------|-----|
| Unit | Neuron: `yᵢ = f(Σⱼ wᵢⱼxⱼ + bᵢ)` | Agent: `Oᵢ = πᵢ(Sᵢ, Mᵢ, {msg})` |
| Weight | `wᵢⱼ ∈ ℝ` | `Ψ(i,j) ∈ [0,1]` (trust) |
| Activation | `f: ℝ → ℝ` (ReLU, sigmoid) | `π: S×M → P(Action)` (reasoning) |
| Forward pass | Synchronous layer-by-layer | Asynchronous message propagation |
| Loss | `L(y, ŷ)` | `J(ANN) = Σ E[R]` |
| Backprop | `∂L/∂w` via chain rule | Trust update + structural evolution |
| Architecture | Fixed DAG | Dynamic graph `G(t)` |
| State | Stateless (between forward passes) | Stateful (persistent memory) |

---

# 2. Decentralization and Distribution

## 2.0 Принцип: никакой центральной точки

> **Axiom D₁**: В ANN нет выделенного центрального узла.
> Любой агент может быть уничтожен — система продолжит работу.
> Любой агент может войти или выйти — система адаптируется.
> Никакой агент не обладает глобальным контролем.

Это НЕ иерархия с оркестратором. Это **самоорганизующийся граф равноправных узлов**,
где структура, роли и координация **эмерджентны** — возникают из локальных взаимодействий.

## 2.1 P2P Overlay Network

Каждый агент — полноправный узел одноранговой сети. Никаких серверов, никаких мастеров.

```
Физическая топология:

  Node₁ (Machine A)          Node₂ (Machine B)          Node₃ (Machine C)
  ┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐
  │  [a₁] ←→ [a₂]  │◄══════►│  [a₄] ←→ [a₅]  │◄══════►│  [a₇] ←→ [a₈]  │
  │    ↕       ↕    │        │    ↕       ↕    │        │    ↕       ↕    │
  │  [a₃]   [a_x]  │        │  [a₆]   [a_y]  │        │  [a₉]   [a_z]  │
  │                 │        │                 │        │                 │
  │  Local Redis    │        │  Local Redis    │        │  Local Redis    │
  │  Local Storage  │        │  Local Storage  │        │  Local Storage  │
  └────────┬────────┘        └────────┬────────┘        └────────┬────────┘
           │                          │                          │
           └──────────────────────────┴──────────────────────────┘
                         P2P Transport Layer (libp2p / NATS mesh)

  Нет центрального сервера. Нет единой базы данных.
  Каждый узел самодостаточен и может работать автономно.
```

## 2.2 Agent Discovery: Distributed Hash Table (DHT)

Агенты находят друг друга без централизованного реестра.

```
Протокол: Kademlia DHT (как в BitTorrent, IPFS, Ethereum)

Идентификация:
  agent_id = SHA-256(public_key)         # 256-bit идентификатор
  distance(a, b) = agent_id(a) XOR agent_id(b)  # XOR-метрика

Таблица маршрутизации (k-buckets):
  Каждый агент хранит:
    bucket[i] = {k ближайших агентов с расстоянием в диапазоне [2^i, 2^(i+1))}
    k = 20 (replication factor)

Обнаружение агента:
  FIND_AGENT(target_id):
    1. Найти α ближайших агентов из своей таблицы
    2. Параллельно запросить у них ещё более близких
    3. Итеративно сужать, пока не найден целевой агент
    4. Сложность: O(log n) хопов для n агентов

Публикация возможностей:
  ANNOUNCE(agent_id, capabilities, domain, trust_score):
    1. Вычислить hash(domain) → DHT-ключ
    2. Сохранить запись на k ближайших к ключу узлах
    3. TTL = 1 час, автоматический refresh
```

**Поиск по компетенции** (не по ID):

```
FIND_BY_CAPABILITY(domain="backend", min_trust=0.7):
  1. key = hash("capability:" + domain)
  2. records = DHT_GET(key)  → список {agent_id, trust, load}
  3. filter: trust ≥ min_trust AND load < 0.8
  4. sort by: trust · (1 - load)  # лучший доступный
  5. return top-k

Это позволяет любому агенту найти нужного партнёра
без обращения к какому-либо центру.
```

## 2.3 Leaderless Consensus: Hashgraph-inspired Protocol

Никакого лидера. Никакого координатора. Агенты достигают согласия
через **виртуальное голосование** без явного обмена голосами.

```
Protocol: AGENT-GRAPH CONSENSUS (AGC)

Основан на Hashgraph (Baird, 2016) + доверительные веса ANN.

Структура данных — Directed Acyclic Graph событий:

  Время →

  a₁: ●────────●────────────●──────────●
       │        │            │          │
  a₂: ●────●───┘────●───────┘──●───────┘
       │    │        │          │
  a₃: ●────┘────────●──────────┘

  Каждая точка (●) = событие = {payload, timestamp, self_parent, other_parent, signature}
  Каждая связь = gossip sync между двумя агентами

Алгоритм:

  1. GOSSIP_SYNC (случайный):
     Агент aᵢ случайно выбирает соседа aⱼ
     aᵢ отправляет aⱼ все события, которых aⱼ не видел
     aⱼ отправляет aᵢ все события, которых aᵢ не видел
     Оба создают новое событие, ссылающееся на последние события обоих

  2. DIVIDE_ROUNDS:
     Событие x принадлежит раунду r, если:
       x может сильно видеть >2n/3 свидетелей раунда r-1
     "Сильно видеть" = путь через >2n/3 различных агентов

  3. VIRTUAL_VOTING (без реальных голосов!):
     Порядок событий определяется математически:
       - Для каждого события вычисляется "медианный timestamp"
       - Учитываемый через сильную видимость через >2n/3 узлов
       - Результат детерминистичен: все честные агенты приходят
         к одному и тому же порядку

  4. TRUST_WEIGHTED_EXTENSION:
     Стандартный Hashgraph: каждый агент = 1 голос
     ANN расширение: голос агента aᵢ взвешен его агрегированным доверием:

       vote_weight(aᵢ) = Ψ_rep(aᵢ) / Σⱼ Ψ_rep(aⱼ)

     Порог: "сильно видеть" ≥ 2W/3, где W = сумма весов

     Это означает: агенты с высоким доверием имеют больше влияния
     на порядок решений, но НИ ОДИН агент не может быть диктатором,
     потому что нужно >2/3 совокупного веса.

Свойства:
  - Asynchronous BFT: работает без синхронных часов
  - Fairness: нельзя цензурировать транзакции
  - Tolerance: f < n/3 Byzantine узлов (по весу доверия)
  - Finality: решение финально за O(log n) раундов
  - NO LEADER: нет выделенного координатора на любом этапе
  - Bandwidth: O(n · txn_size) на раунд через gossip
```

## 2.4 Self-Organizing Topology

Топология возникает снизу вверх. Никто не назначает "кто с кем связан".

```
Принцип: STIGMERGIC TOPOLOGY (стигмергическая топология)

Агенты формируют и разрывают связи на основе ЛОКАЛЬНЫХ сигналов:

  Rule S1 — ATTRACTION (формирование связи):
    IF quality(interactions with aⱼ) > θ_attract
    AND Ψ(i,j) > θ_trust
    AND NOT saturated(neighbors(aᵢ))
    THEN: add_edge(aᵢ → aⱼ)

    Агенты "притягиваются" к тем, с кем полезно работать.

  Rule S2 — REPULSION (разрыв связи):
    IF Ψ(i,j) < θ_repulse for T consecutive interactions
    OR inactive(edge(i,j)) for T_idle
    THEN: remove_edge(aᵢ → aⱼ)

    Агенты "отталкиваются" от бесполезных или ненадёжных.

  Rule S3 — INTRODUCTION (рекомендация):
    IF aᵢ knows aⱼ AND aᵢ knows aₖ
    AND domain_overlap(aⱼ, aₖ) > θ_intro
    AND NOT edge(aⱼ, aₖ)
    THEN: aᵢ sends INTRODUCE(aⱼ, aₖ) to both

    Агенты рекомендуют друг друга — "знакомят".

  Rule S4 — CLUSTER_EMERGENCE (самоорганизация кластеров):
    Кластеры НЕ назначаются сверху.
    Они возникают как плотные подграфы:

    cluster(aᵢ) = connected_component in subgraph where ∀ edges: Ψ > θ_cluster

    Кластер — это группа агентов с высоким взаимным доверием.
    Никто не создаёт кластер. Он ПОЯВЛЯЕТСЯ.

  Rule S5 — ROLE_EMERGENCE (эмерджентные роли):
    Роли не назначаются. Они определяются позицией в графе:

    IF betweenness_centrality(aᵢ) > θ_bridge:
      aᵢ де-факто является BRIDGE агентом (связывает кластеры)

    IF degree(aᵢ) > μ + 2σ AND avg_trust(aᵢ) > θ_hub:
      aᵢ де-факто является HUB (но не контролирует — просто популярен)

    IF aᵢ consistently validates others' output:
      aᵢ де-факто является VALIDATOR

    Роли динамичны. Сегодня hub — завтра нет.
    Сеть постоянно перестраивается.

Аналогия с биологией:
  - Attraction/Repulsion → хемотаксис
  - Cluster emergence → формирование тканей
  - Role emergence → дифференциация клеток
  - Introduction → синаптогенез через промежуточные нейроны
```

**Математика самоорганизации**:

```
Динамика edge weight:
  W(i,j,t+1) = W(i,j,t) + η · [reward(i,j,t) - baseline(i,t)] · eligibility(i,j,t)

Где:
  reward(i,j,t) = quality последнего взаимодействия через это ребро
  baseline(i,t) = скользящее среднее reward для агента i
  eligibility(i,j,t) = exp(-Δt/τ) — следы недавней активности
  η = learning rate

Это REINFORCE для структуры графа:
  Хорошие взаимодействия усиливают связь.
  Плохие — ослабляют.
  Нет обратной связи от центра. Только локальные наблюдения.

Обрезка:
  IF W(i,j,t) < θ_prune: remove edge(i,j)

Ограничение на степень (предотвращает "суперузлы"):
  max_degree(aᵢ) = D_max = 2 · ⌈log₂(|A|)⌉

  Если aᵢ достиг предела, новая связь возможна только
  если она вытесняет самую слабую существующую:
    IF W(i,j_new) > min_k W(i,k): replace(min_k, j_new)
```

## 2.5 Decentralized Memory: CRDTs + Merkle DAG

Никакой "глобальной базы данных". Память распределена между узлами.

```
Архитектура:

  ┌─────────────────────────────────────────────────────────────┐
  │                                                              │
  │   Agent a₁                Agent a₂               Agent a₃   │
  │   ┌──────────┐            ┌──────────┐           ┌────────┐ │
  │   │ Episodic  │            │ Episodic  │           │Episodic│ │
  │   │ (local)   │            │ (local)   │           │(local) │ │
  │   ├──────────┤            ├──────────┤           ├────────┤ │
  │   │ Semantic  │◄═════════►│ Semantic  │◄═════════►│Semantic│ │
  │   │ (replica) │  CRDT     │ (replica) │  CRDT    │(replica│ │
  │   ├──────────┤  sync     ├──────────┤  sync    ├────────┤ │
  │   │Procedural│◄═════════►│Procedural│◄═════════►│Proced. │ │
  │   │ (replica) │  Merkle   │ (replica) │  Merkle  │(replica│ │
  │   ├──────────┤  DAG      ├──────────┤  DAG     ├────────┤ │
  │   │  Audit    │◄═════════►│  Audit    │◄═════════►│ Audit  │ │
  │   │ (append)  │  CL chain │ (append)  │  CL chain│(append)│ │
  │   └──────────┘            └──────────┘           └────────┘ │
  │                                                              │
  │   Каждый агент хранит ВСЮ память локально.                  │
  │   Синхронизация — через gossip без центрального сервера.     │
  └─────────────────────────────────────────────────────────────┘
```

### Episodic Memory (локальная, не реплицируется)

```
Хранение: локальный Redis / in-memory LRU
Видимость: только владелец
Очистка: TTL + LRU eviction
Назначение: контекст текущей задачи, короткие воспоминания
```

### Semantic Memory (CRDT-реплицируемая)

```
Тип CRDT: OR-Set (Observed-Remove Set) для коллекций
           LWW-Register (Last-Writer-Wins) для отдельных записей

Структура:
  semantic_store = OR_Set<SemanticEntry>

  SemanticEntry = {
    id:        UUID,
    vector:    float[d],          # embedding
    content:   string,            # raw text
    author:    AgentID,
    timestamp: HLC,               # Hybrid Logical Clock
    version:   VectorClock,
    ttl:       Duration | ∞
  }

Запись:
  1. Агент aᵢ создаёт запись локально
  2. aᵢ подписывает запись: sig = sign(entry, private_key_i)
  3. aᵢ добавляет в свой OR-Set
  4. Gossip распространяет на соседей: O(log n) раундов до полного spread

Конфликты:
  OR-Set гарантирует: add побеждает concurrent remove
  LWW-Register: при конкурентных записях побеждает с большим HLC

  HLC (Hybrid Logical Clock):
    hlc.send():    l' = max(l, pt); c' = (l'==l ? c+1 : 0); return (l',c')
    hlc.receive(m): l' = max(l, m.l, pt); c' = ...; return (l',c')
    Порядок: (l₁,c₁) < (l₂,c₂) iff l₁<l₂ OR (l₁=l₂ AND c₁<c₂)

    Не требует синхронизации часов. Работает в полностью асинхронной сети.

Валидация (без центрального gate):
  Вместо единого commit_node — распределённая валидация:

  VALIDATE_ENTRY(entry, k=3):
    1. Выбрать k случайных валидаторов из DHT (по домену entry)
    2. Каждый валидатор проверяет: relevance, accuracy, consistency
    3. IF ≥ ⌈k/2⌉+1 approve: entry помечается validated=true
    4. validated=true записи имеют приоритет при поиске

  Любой агент может быть валидатором. Выбор случайный + доверительный.
```

### Procedural Memory (Merkle DAG)

```
Граф знаний хранится как Merkle DAG (как в IPFS/Git):

  Каждый узел графа:
    node_hash = SHA-256(content + Σ child_hashes)

  Каждый агент хранит свой локальный subgraph.
  Синхронизация — через обмен корневыми хэшами:

  SYNC_PROCEDURAL(aᵢ, aⱼ):
    1. aᵢ отправляет root_hash(aᵢ)
    2. aⱼ сравнивает со своим root_hash(aⱼ)
    3. Если различаются — обмен по дельте:
       - Обход дерева сверху вниз
       - Пропуск поддеревьев с совпадающими хэшами
       - Передача только отличающихся узлов
    4. Merge: union с разрешением конфликтов через HLC

  Сложность синхронизации: O(diff_size · log(graph_size))
  В стабильной системе diff_size ≈ 0 → почти бесплатно.

Преимущество: целостность верифицируема.
  Любой агент может проверить, что данные не искажены,
  сравнив хэши с несколькими соседями.
```

### Audit Log (Causal Log Chain)

```
Append-only лог, цепочка причинно-связанных записей:

  entry[t] = {
    data:       AuditEvent,
    prev_hash:  hash(entry[t-1]),         # цепочка
    author:     AgentID,
    signature:  sign(data + prev_hash),
    witnesses:  [AgentID, ...]            # кто видел запись
  }

Нет блокчейна в полном смысле (нет PoW/PoS).
Есть причинный порядок + подписи + свидетели.

Верификация: entry действительна, если:
  1. signature корректна
  2. prev_hash совпадает с предыдущей записью
  3. ≥ w свидетелей подтвердили (w = witness_threshold, default 2)
```

## 2.6 Agent Lifecycle: Join / Live / Leave / Fail

Агенты свободно входят и выходят. Нет permission server.

```
JOIN (новый агент входит в сеть):
  1. aᵢ_new генерирует keypair: (public_key, private_key)
  2. agent_id = SHA-256(public_key)
  3. aᵢ_new подключается к bootstrap nodes (hardcoded или из конфига)
  4. aᵢ_new выполняет DHT_BOOTSTRAP:
     - Находит ближайших соседей по XOR-расстоянию
     - Заполняет свои k-buckets
  5. aᵢ_new публикует свои capabilities в DHT:
     ANNOUNCE(agent_id, {domain, tools, level})
  6. aᵢ_new начинает получать задачи через gossip
  7. Начальное доверие: Ψ(*, aᵢ_new) = Ψ_default = 0.3
     (низкое — нужно заработать)

LIVE (нормальная работа):
  - Периодический heartbeat через gossip (не broadcast!)
  - Участие в DHT maintenance (refresh buckets каждые 15 мин)
  - Обработка входящих сообщений
  - Инициация исходящих взаимодействий
  - Синхронизация памяти с соседями

LEAVE (грациозный выход):
  1. aᵢ отправляет LEAVE_NOTICE соседям
  2. Соседи перераспределяют DHT-записи, которые хранил aᵢ
  3. Незавершённые задачи aᵢ возвращаются в pool
  4. Edge weights к aᵢ обнуляются, рёбра удаляются

FAIL (неожиданный отказ):
  1. Соседи обнаруживают отсутствие heartbeat:
     timeout = 3 × heartbeat_interval (default: 30s)
  2. Сосед aⱼ инициирует FAILURE_SUSPECTED(aᵢ):
     - Опрос k других соседей aᵢ
     - Если ≥ ⌈k/2⌉+1 подтверждают: aᵢ объявляется FAILED
  3. DHT-записи aᵢ переносятся на ближайшие узлы
  4. Задачи aᵢ переназначаются через gossip
  5. Ψ(*, aᵢ) не обнуляется (сохраняется на случай возврата)

Анти-Sybil защита:
  Ψ_default = 0.3 означает: новый агент почти ничего не может.
  Для участия в валидации нужно Ψ > 0.6 — это недели работы.
  Для участия в консенсусе нужно Ψ > 0.5 — нельзя "наводнить" сеть.

  Proof of Useful Work:
    Агент доказывает свою ценность только через РЕЗУЛЬТАТЫ.
    Нет shortcut. Нет регистрации. Только работа и доверие.
```

## 2.7 Partition Tolerance: Split Brain Resolution

```
Сценарий: сеть разделяется на два изолированных сегмента.

  Segment P          │ network    │       Segment Q
  [a₁][a₂][a₃]      │ partition  │       [a₄][a₅][a₆]
                     │            │

Поведение:
  1. Оба сегмента ПРОДОЛЖАЮТ РАБОТАТЬ автономно
     (Availability > Consistency)

  2. Каждый сегмент:
     - Обслуживает задачи со своим подмножеством агентов
     - Продолжает локальный консенсус (с меньшим числом участников)
     - Накапливает изменения в памяти

  3. При восстановлении связи — CRDT merge:

     MERGE_AFTER_PARTITION(P, Q):
       a) Semantic memory: OR-Set merge (автоматический, бесконфликтный)
       b) Procedural memory: Merkle DAG merge (diff + union)
       c) Audit log: каузальное слияние (interleave по HLC)
       d) Trust scores: average с весом = время в сегменте
          Ψ_merged(i,j) = (T_P · Ψ_P(i,j) + T_Q · Ψ_Q(i,j)) / (T_P + T_Q)
       e) Topology: union рёбер из обоих сегментов

     Гарантия: после merge все агенты видят одинаковое состояние
     (CRDT convergence theorem).

  4. Если разделение > T_permanent (default: 24h):
     Сегменты считаются независимыми ANN-сетями.
     Merge возможен, но требует ручного подтверждения.

CAP позиция:
  - Availability: ДА (оба сегмента работают)
  - Partition tolerance: ДА (по определению)
  - Consistency: eventual (после merge через CRDTs)

  Staleness bound (в нормальном режиме):
    ∀ запись w, созданная в момент t:
    ∀ агентов aᵢ: aᵢ увидит w к моменту t + O(diameter(G) · gossip_interval)

    Типично: diameter ≈ log(n), gossip_interval = 5s
    → Staleness ≈ 5·log(n) секунд
    → Для 1000 агентов: ~50 секунд
```

## 2.8 Decentralized Task Routing

Задачи маршрутизируются без центрального диспетчера.

```
Протокол: STIGMERGIC TASK ROUTING (STR)

Когда задача поступает в сеть (от внешнего клиента или другого агента):

  1. ENTRY: задача попадает на ЛЮБОЙ узел сети (точка входа случайна)

  2. LOCAL_EVAL: принимающий агент aᵢ оценивает:
     - Могу ли я решить это сам? → confidence(aᵢ, task) ≥ θ_solo
     - Если да: решаю локально
     - Если нет: переходим к маршрутизации

  3. ROUTE_BY_COMPETENCE:
     aᵢ ищет лучшего исполнителя через DHT:
     candidates = FIND_BY_CAPABILITY(task.domain, min_trust=0.5)

     Scoring:
       score(aⱼ) = w₁·domain_match(aⱼ, task)
                  + w₂·Ψ(aᵢ, aⱼ)
                  + w₃·(1 - load(aⱼ))
                  - w₄·distance(aᵢ, aⱼ)  # сетевые хопы

     Отправить задачу агенту с max score

  4. RECURSIVE_DECOMPOSITION:
     Если задача сложная, получивший агент:
     a) Декомпозирует на подзадачи
     b) Для каждой подзадачи — ROUTE_BY_COMPETENCE
     c) Собирает результаты
     d) Агрегирует финальный ответ

     Никакого центрального планировщика.
     Декомпозиция — решение конкретного агента.

  5. PHEROMONE_TRAILS (стигмергия):
     Успешные маршруты оставляют "след":

     pheromone(task_domain, aⱼ) += reward
     pheromone(task_domain, aⱼ) *= (1 - evaporation_rate)  # decay

     Следующие похожие задачи будут маршрутизироваться
     по усиленным путям — без какого-либо центрального обучения.
     Чистая стигмергия, как у муравьёв.

  6. LOAD_BALANCING (без балансировщика):
     Агент с load > θ_overload:
       - Отказывается принимать новые задачи (BACKPRESSURE)
       - Перенаправляет на следующего по score

     Агент с load < θ_idle:
       - Публикует AVAILABLE через gossip
       - Увеличивает радиус видимости в DHT

  Отказоустойчивость маршрутизации:
    Если aⱼ не отвечает в течение timeout:
      aᵢ пробует следующего кандидата из candidates
      Ψ(aᵢ, aⱼ) -= penalty

    Нет единой точки отказа. Задача ВСЕГДА найдёт исполнителя,
    пока в сети есть хотя бы один живой агент с нужной компетенцией.
```

## 2.9 Federation: Межсетевое взаимодействие

Разные ANN-сети могут объединяться без потери автономии.

```
Модель: FEDERATED ANN (по аналогии с ActivityPub / Fediverse)

  ANN Network α              ANN Network β              ANN Network γ
  ┌─────────────┐            ┌─────────────┐            ┌─────────────┐
  │ [a₁]...[aₙ] │            │ [b₁]...[bₘ] │            │ [c₁]...[cₖ] │
  │              │            │              │            │              │
  │  Gateway Gα  │◄══════════►│  Gateway Gβ  │◄══════════►│  Gateway Gγ  │
  └─────────────┘            └─────────────┘            └─────────────┘

Gateway — НЕ центральный узел. Это РОЛЬ, которую может исполнять
любой агент с betweenness_centrality > θ_gateway.
Роль ротируется автоматически.

Межсетевой протокол:
  1. Discovery: ANN-сети публикуют свои gateway endpoints
  2. Trust: межсетевое доверие начинается с Ψ_inter = 0.1
  3. Tasks: задачи могут маршрутизироваться в другую сеть,
     если локальная компетенция недостаточна
  4. Memory: по умолчанию НЕ реплицируется между сетями
     (только по явному соглашению)
  5. Autonomy: каждая сеть сохраняет полный суверенитет

Пример: ANN_research федерируется с ANN_engineering.
  - Исследовательская задача → маршрутизируется в ANN_research
  - Инженерная задача → маршрутизируется в ANN_engineering
  - Мультидисциплинарная → декомпозируется между обеими
  - Каждая сеть сохраняет свою топологию, доверие, эволюцию.
```

## 2.10 Формальные гарантии децентрализации

```
Теорема D1 (No Single Point of Failure):
  ∀ aᵢ ∈ A: ANN \ {aᵢ} остаётся функциональной,
  при условии |A| ≥ 4 и граф связности k ≥ 2.

  Доказательство: DHT перебалансируется за O(log n) хопов.
  Задачи aᵢ перемаршрутизируются через STR.
  Память aᵢ восстановима из реплик (replication factor k=3). ∎

Теорема D2 (No Single Point of Control):
  ∀ aᵢ ∈ A: aᵢ не может в одностороннем порядке:
  - изменить глобальное состояние (нужен консенсус >2/3 по весу)
  - заблокировать другого агента (P2P, нет центрального gate)
  - подделать чужие записи (криптографические подписи)
  - монополизировать маршрутизацию (DHT = O(log n), нет центра)

Теорема D3 (Convergence under Partition):
  После разрешения сетевого разделения,
  CRDT-состояние всех агентов сходится к единому значению
  за O(diameter(G_merged) · gossip_interval) времени.

  Следствие из свойств OR-Set и LWW-Register. ∎

Теорема D4 (Permissionless Entry):
  Любой агент может войти в сеть, не запрашивая разрешения.
  Ограничение — через доверие: Ψ_default = 0.3.
  Полная функциональность достигается через Proof of Useful Work.

Теорема D5 (Emergent Hierarchy):
  В любой достаточно большой ANN (|A| > 20) с стигмергической топологией
  спонтанно возникает иерархическая структура (power-law degree distribution)
  с hub-агентами, но без назначенных лидеров.

  Следствие из preferential attachment в scale-free networks
  (Barabási–Albert model). ∎
```

---

# 3. Evolutionary Mechanism

## 3.1 Genome Encoding

Each ANN configuration is encoded as a genome `G`:

```
G = (G_topology, G_agents, G_protocol)

G_topology ∈ {0,1,2,3}^L_t    # topology genes (L_t ∈ [32, 512])
G_agents   ∈ {0,1,2,3}^L_a    # agent configuration genes
G_protocol ∈ {0,1,2,3}^L_p    # protocol parameter genes
```

**Topology genes** encode:
- Number of agents: `n = 4 + g[0:2] decode → [4, 64]`
- Connectivity pattern: `g[2:4] → {ring, star, mesh, hierarchical}`
- Edge density: `g[4:6] → [sparse=0.1, dense=0.9]`
- Cluster count: `g[6:8] → [1, 8]`

**Agent genes** encode (per agent):
- Cognitive level: `g[0] → {L1, L2, L3, L4}`
- Specialization: `g[1:3] → domain index`
- Memory capacity: `g[3:5] → [small, medium, large]`
- Tool set: `g[5:8] → bitmask of available tools`
- Confidence threshold: `g[8:10] → [0.3, 0.95]`

**Protocol genes** encode:
- Consensus threshold: `g[0:2] → [0.5, 0.9]`
- Gossip frequency: `g[2:4] → [1, 100] steps`
- Trust decay rate: `g[4:6] → [0.01, 0.1]`
- Escalation threshold: `g[6:8] → [0.3, 0.8]`

## 3.2 Fitness Function

```
F(G) = w₁·Quality(G) + w₂·Efficiency(G) + w₃·Robustness(G) - w₄·Cost(G)

Where:
  Quality(G)    = avg task success rate over evaluation suite
  Efficiency(G) = 1 / avg response time
  Robustness(G) = performance under agent failures (remove 20% randomly)
  Cost(G)       = Σ_i cost(a_i) / budget  (normalized resource usage)

Default weights: w₁=0.4, w₂=0.2, w₃=0.3, w₄=0.1
```

## 3.3 Evolutionary Operators

### Selection: Tournament Selection with Elitism

```
TOURNAMENT_SELECT(population, k=3):
  candidates = random_sample(population, k)
  return argmax(candidates, key=fitness)

ELITISM: top 10% survive unchanged
```

### Crossover: Graph-aware crossover

```
CROSSOVER(parent1, parent2):
  # Topology crossover
  cut_point = random(0, L_t)
  child_topology = parent1.G_topology[:cut_point] + parent2.G_topology[cut_point:]

  # Agent crossover (per-agent gene alignment via innovation numbers)
  For each agent gene pair (a1_i, a2_j) with matching innovation:
    child_agent_i = uniform_crossover(a1_i, a2_j)

  # Protocol crossover (arithmetic mean)
  child_protocol = (parent1.G_protocol + parent2.G_protocol) / 2

  return Genome(child_topology, child_agents, child_protocol)
```

### Mutation: Multi-scale

```
MUTATE(genome, rate=0.05):
  # Point mutation (change single gene)
  For each gene g_i with probability rate:
    g_i = random({0,1,2,3})

  # Structural mutation (add/remove agent, with probability rate/10):
    ADD_AGENT: insert new random agent genes
    REMOVE_AGENT: delete weakest agent's genes
    ADD_EDGE: connect two previously unconnected agents
    REMOVE_EDGE: disconnect two connected agents

  # Role mutation (change agent specialization, with probability rate/5):
    CHANGE_ROLE: reassign agent domain index
    PROMOTE: increase cognitive level (L1→L2→L3)
    DEMOTE: decrease cognitive level (L3→L2→L1)
```

## 3.4 Speciation (NEAT-inspired)

Agents are grouped into species to protect topological innovation:

```
distance(G₁, G₂) = c₁·E/N + c₂·D/N + c₃·W̄

Where:
  E = number of excess genes
  D = number of disjoint genes
  W̄ = average weight difference of matching genes
  N = max(|G₁|, |G₂|)
  c₁, c₂, c₃ = compatibility coefficients

Speciation threshold: δ = 3.0
G₁ and G₂ are same species iff distance(G₁, G₂) < δ
```

## 3.5 Online Evolution (odNEAT-inspired)

The ANN supports **runtime evolution** without stopping the system:

```
ONLINE_EVOLVE(ANN, interval=1000_tasks):
  Every interval tasks:
    1. Evaluate: compute fitness for each agent based on recent performance
    2. Identify weak: W = {aᵢ | fitness(aᵢ) < μ - 2σ}
    3. Identify strong: S = {aᵢ | fitness(aᵢ) > μ + σ}
    4. Replace: For each w ∈ W:
         parent = tournament_select(S)
         w.genes = mutate(parent.genes)
         w.memory = initialize_fresh()
         w.trust = 0.5  (neutral starting trust)
    5. Adapt topology:
         prune edges where Ψ < 0.2
         add edges between agents with correlated success
    6. Log generation metrics to audit memory
```

## 3.6 Structural Evolution Rules

```
EVOLVE_STRUCTURE(G_t) → G_{t+1}:

  Rule 1 (Growth): If avg_load > 0.8:
    spawn new agent in busiest cluster

  Rule 2 (Pruning): If agent.utilization < 0.1 for 100 cycles:
    remove agent, redistribute connections

  Rule 3 (Rewiring): Every K steps:
    For each edge (i,j) where Ψ(i,j) < θ_prune:
      remove edge (i,j)
    For each agent pair (i,j) where correlation(Oᵢ, Oⱼ) > θ_connect:
      add edge (i,j) if not exists

  Rule 4 (Specialization): If agent handles >80% tasks from one domain:
    restrict tool set to domain-specific tools
    increase memory allocation for that domain

  Rule 5 (Generalization): If agent handles <20% tasks from any single domain:
    expand tool set
    balance memory allocation
```

---

# 4. Agent Interaction Protocol

## 4.1 Message Format

```
Message = {
  id:          UUID,
  type:        MessageType,
  sender:      AgentID,
  receiver:    AgentID | BroadcastGroup,
  content:     Payload,
  priority:    Priority,
  confidence:  float ∈ [0, 1],
  timestamp:   Timestamp,
  ttl:         Duration,
  trace_id:    UUID,           # for distributed tracing
  reply_to:    UUID | null,    # for conversation threading
  signature:   Hash            # integrity verification
}

MessageType ∈ {
  SIGNAL,        # data/result transmission
  CONTEXT,       # state sharing
  COMMAND,       # task delegation directive (peer-to-peer)
  QUERY,         # information request
  VOTE,          # consensus participation
  GOSSIP,        # reputation/trust propagation
  HEARTBEAT,     # liveness check
  ESCALATION,    # confidence-triggered delegation upward
  REFLECTION     # metacognitive feedback
}

Priority ∈ {CRITICAL=0, HIGH=1, NORMAL=2, LOW=3, BACKGROUND=4}

Payload = {
  data:       Any,           # structured content
  reasoning:  string,        # explanation of decision
  artifacts:  [Artifact],    # produced outputs
  metadata:   Map            # extensible metadata
}
```

## 4.2 Interaction Patterns

### Pattern 1: Signal Propagation (Feed-Forward)

```
Sender → Receiver

aᵢ sends result to downstream agent aⱼ.
Analogous to activation flow in classical NN.

Trigger: aᵢ completes computation
Message: type=SIGNAL, content=result
Guarantee: at-least-once delivery
Timeout: ttl (default 30s)
```

### Pattern 2: Context Sharing (Lateral)

```
Agent ←→ Peer

Bilateral state exchange between agents in same cluster.
Analogous to lateral connections / skip connections.

Trigger: periodic or event-driven
Message: type=CONTEXT, content=state_summary
Guarantee: best-effort (gossip)
Frequency: every gossip_interval steps
```

### Pattern 3: Task Delegation (Peer-to-Peer)

```
Agent → Best_Peer (via DHT + pheromone routing)

Agent delegates task to more competent peer.
No hierarchy. Peer selected by trust × competence × availability.
Analogous to attention mechanism (routing to most relevant node).

Trigger: agent lacks confidence or domain expertise
Message: type=COMMAND, content=task_assignment
Guarantee: exactly-once with acknowledgment
Fallback: if peer fails, try next candidate from DHT
```

### Pattern 4: Consensus Round (Leaderless)

```
Any Agent → Gossip → All Relevant → Virtual Vote

Leaderless agreement via Agent-Graph Consensus (AGC).
No proposer/leader. Order determined by gossip DAG structure.
Analogous to batch normalization / pooling.

Trigger: decision requiring collective agreement
Protocol: AGC (see Section 2.3)
Messages: VOTE type via gossip propagation
Guarantee: Async BFT (tolerates f < n/3 faults by trust weight)
```

### Pattern 5: Competence Routing (Lateral)

```
Agent → DHT_Lookup → Best_Peer → Execute → Return

Low-confidence agent routes to more competent PEER (not "up").
No hierarchy. Pure competence-based routing.
Analogous to skip connections / routing in mixture-of-experts.

Trigger: confidence_i < θ_routing
Protocol:
  1. Agent queries DHT for capability match
  2. Scores candidates by trust × domain_match × availability
  3. Sends task to top candidate
  4. Deposits pheromone on successful route
Message: type=QUERY → type=SIGNAL (response)
Guarantee: at-least-once, with retry to next candidate on timeout
```

### Pattern 6: Gossip (Emergent)

```
Agent → Random_Subset(Neighbors)

Trust and reputation propagation.
Analogous to dropout / stochastic depth.

Trigger: periodic (gossip_interval)
Message: type=GOSSIP, content={agent_id, trust_score, evidence}
Guarantee: best-effort, eventually consistent
Propagation: epidemic model, O(log n) rounds to full spread
```

### Pattern 7: Adversarial Challenge (Dialectic)

```
Agent ←→ Critic

Agent-Critic pair for quality verification.
Analogous to GAN discriminator feedback.

Trigger: agent produces output
Message: type=SIGNAL (proposal) → type=SIGNAL (critique)
Protocol:
  1. Agent sends proposal
  2. Critic evaluates and sends critique
  3. Agent revises or justifies
  4. Repeat until convergence or max_rounds
```

## 4.3 Protocol State Machine

Each agent's communication follows a state machine:

```
             ┌─────────┐
             │  IDLE    │◄──────────────────┐
             └────┬─────┘                   │
                  │ receive(msg)            │ timeout / complete
                  ▼                         │
           ┌──────────────┐                │
           │  PROCESSING  │                │
           └──────┬───────┘                │
                  │                         │
         ┌────────┼────────┐               │
         ▼        ▼        ▼               │
    ┌────────┐ ┌──────┐ ┌──────────┐      │
    │RESPOND │ │QUERY │ │ESCALATE  │      │
    └───┬────┘ └──┬───┘ └────┬─────┘      │
        │         │           │             │
        ▼         ▼           ▼             │
    ┌────────────────────────────────┐     │
    │         SENDING                │─────┘
    └────────────────────────────────┘
```

## 4.4 Flow Control

**Backpressure**: When agent's inbox exceeds `queue_max`:
```
if inbox.size > queue_max:
  drop(messages where priority > NORMAL and age > ttl/2)
  if still overflow:
    send(BACKPRESSURE_SIGNAL to all senders)
    senders reduce send rate by 50%
```

**Priority queue**: Messages processed in priority order (CRITICAL first):
```
inbox = PriorityQueue(key=lambda m: (m.priority, m.timestamp))
```

**Deduplication**: Messages with same `id` are processed only once:
```
if msg.id in processed_set: drop(msg)
else: processed_set.add(msg.id); process(msg)
```

## 4.5 Interaction Protocol Layers

```
Layer 4: Semantic     — meaning, intent, reasoning
Layer 3: Session      — conversation threading, context
Layer 2: Reliability  — exactly-once, ordering, dedup
Layer 1: Transport    — RabbitMQ / Redis Pub/Sub / gRPC
Layer 0: Network      — TCP/IP, container networking
```

---

# 5. Scientific Hypothesis

## 5.1 Primary Hypothesis

> **H₁**: A network of autonomous reasoning agents connected by trust-weighted
> dynamic edges, subject to evolutionary topology optimization, produces
> emergent collective intelligence that exceeds the sum of individual agent
> capabilities by a factor proportional to the network's effective connectivity.

Formally:

```
Performance(ANN) > Σᵢ Performance(aᵢ) · (1 + β · λ₂(L))

Where:
  λ₂(L) = algebraic connectivity (Fiedler value) of graph Laplacian
  β > 0 = emergence coefficient (to be empirically determined)
```

## 5.2 Supporting Hypotheses

> **H₂ (Cognitive Scaling Law)**: The collective intelligence of an ANN scales
> as O(n · log(n)) with the number of agents n, under optimal topology,
> compared to O(n) for independent agents and O(1) for a monolithic system.

> **H₃ (Trust-Performance Correlation)**: The performance of an ANN on a given
> task class correlates positively with the average trust score along the
> critical path of agent activations for that task class.

```
Corr(Performance(task), avg_path_trust(task)) > 0.7
```

> **H₄ (Evolutionary Convergence)**: Under the evolutionary mechanism (Section 3),
> the ANN topology converges to a near-optimal configuration within
> O(G · |A|² · log(|A|)) generations, where G is the task diversity.

> **H₅ (Resilience Emergence)**: An ANN with trust-based topology evolution
> autonomously develops fault-tolerant structures (redundant paths,
> specialized backup agents) without explicit fault-tolerance programming.

## 5.3 Null Hypotheses (To Be Refuted)

> **H₀₁**: ANN performance does not significantly exceed a single monolithic
> agent with equivalent total computational budget.

> **H₀₂**: Evolutionary topology optimization provides no significant
> improvement over a fixed hierarchical topology.

> **H₀₃**: Trust dynamics do not significantly affect network performance
> compared to uniform edge weights.

## 5.4 Experimental Design

To test these hypotheses:

**Independent Variables**:
- Network size: n ∈ {4, 8, 16, 32, 64}
- Topology type: {random, star, ring, hierarchical, evolved}
- Trust mechanism: {none, static, dynamic, evolutionary}
- Task complexity: {simple, compound, adversarial}

**Dependent Variables**:
- Task success rate
- Response latency (wall clock)
- Resource consumption (LLM tokens, API calls)
- Robustness (performance under 20% agent failure)
- Emergent structure metrics (clustering coefficient, path length)

**Control**:
- Single agent baseline with equivalent total compute
- Fixed topology with equivalent agent count
- Random trust assignment baseline

**Evaluation Suite**: 100 tasks spanning:
- Single-domain factual (baseline)
- Multi-domain synthesis (requires agent collaboration)
- Adversarial (conflicting information)
- Open-ended (requires creative emergence)
- Sequential reasoning (where multi-agent should NOT help, per Science of Scaling)

## 5.5 Falsifiability Criteria

The hypothesis is **falsified** if any of the following hold across 3+ independent runs:

1. `Performance(ANN) ≤ Performance(single_agent)` for >50% of tasks
2. Evolved topology shows no statistically significant improvement (p > 0.05) over fixed topology
3. Trust dynamics show no correlation (|r| < 0.3) with performance
4. System fails to self-recover from 20% agent failure within 10× normal response time

---

# 6. Computational Complexity

## 6.1 Theoretical Complexity Bounds

### Agent-Level Computation

```
Per agent, per step:
  - Receive messages:     O(d_in)        where d_in = in-degree
  - Aggregate context:    O(d_in · d)    where d = embedding dimension
  - Reasoning (LLM call): O(L²)          where L = context length (tokens)
  - Tool execution:       O(T_tool)      variable, depends on tool
  - Send messages:        O(d_out)       where d_out = out-degree

Total per agent per step: O(d_in · d + L²)
```

### Network-Level Computation

**Single propagation wave** (all agents process once):

```
Sequential:  O(n · (d_avg · d + L²))
Parallel:    O(depth(G) · (d_avg · d + L²))    with n/depth(G) parallelism

Where depth(G) = longest path in the activation DAG
```

### Consensus

```
CP-WBFT per round:
  Communication: O(n²)  messages  (standard BFT)
  With clustering: O(K² · (n/K)²) = O(n²/K)  where K = cluster count
  Rounds to converge: O(1) for non-Byzantine, O(f+1) for f Byzantine agents
```

### Trust Update

```
Per agent pair update:  O(1)
Full trust matrix:      O(n²)
With sparsity:          O(|E|)  where |E| = number of edges
Gossip propagation:     O(n · log(n)) steps for global convergence
```

### Evolutionary Step

```
Fitness evaluation:     O(n · T_eval)   where T_eval = evaluation task cost
Selection:              O(P · k)        P = population size, k = tournament size
Crossover:              O(L_genome)
Mutation:               O(L_genome)
Full generation:        O(P · (n · T_eval + L_genome))
Convergence:            O(G · n² · log(n)) generations (hypothesis H₄)
```

## 6.2 Problem Complexity Classification

| Problem | Complexity Class | ANN Approach | Effective Complexity |
|---------|-----------------|--------------|---------------------|
| Cluster-internal coordination | PSPACE-complete | Emergent cluster decomposition | O(n_cluster · PSPACE_local) |
| Fully decentralized | NEXP-complete | Communication-enabled | PSPACE (with Σ protocol) |
| Credit assignment | NP-hard | QMIX decomposition | O(n · |A_i|) per step |
| Optimal topology | NP-hard | Evolutionary search | Approximate in O(gen · pop) |
| Consensus | O(n²) messages | Clustered CP-WBFT | O(n²/K) per round |
| Trust convergence | O(n · log n) | Gossip protocol | O(diameter(G) · log n) |

## 6.3 Scalability Analysis

```
n (agents)  | Messages/step | Memory      | Latency        | LLM Calls/step
------------|---------------|-------------|----------------|----------------
4           | O(16)         | O(4·d)      | O(2·L²)       | 4
16          | O(64)         | O(16·d)     | O(4·L²)       | 16
64          | O(256)        | O(64·d)     | O(6·L²)       | 64
256         | O(1024)*      | O(256·d)    | O(8·L²)       | 256
1024        | O(4096)*      | O(1024·d)   | O(10·L²)      | 1024

* With K=16 clusters, inter-cluster messages: O(K²) = O(256)
  Intra-cluster messages per cluster: O((n/K)²) = O(4096)
  Total: O(K · (n/K)² + K²) = O(n²/K + K²)
```

**Critical scaling limits**:
- **LLM bottleneck**: Each agent requires ≥1 LLM call per step → cost scales as O(n)
- **Memory bottleneck**: Shared memory contention scales as O(n · write_rate)
- **Network bottleneck**: Message volume scales as O(|E|) ≈ O(n · d_avg)

## 6.4 Cost Model

```
Cost_per_step = Σᵢ (cost_LLM(aᵢ) + cost_tools(aᵢ) + cost_memory(aᵢ) + cost_network(aᵢ))

Where:
  cost_LLM(aᵢ)     = tokens_in(aᵢ) · price_in + tokens_out(aᵢ) · price_out
  cost_tools(aᵢ)    = Σ_{t ∈ used_tools} cost(t)
  cost_memory(aᵢ)   = reads(aᵢ) · cost_read + writes(aᵢ) · cost_write
  cost_network(aᵢ)  = messages_sent(aᵢ) · cost_msg

Estimated per-step cost (8 agents, Claude Sonnet):
  8 × (2K tokens × $3/M_in + 500 tokens × $15/M_out) ≈ $0.108/step
  At 10 steps per task: ~$1.08/task

  Optimization target: reduce via caching, routing, selective activation
```

## 6.5 Latency Model

```
Latency_task = Σ_{layers} max_{agents_in_layer}(latency_agent)

latency_agent = t_receive + t_reason + t_act + t_send

Where:
  t_receive = O(d_in · msg_size / bandwidth)      ≈ 1-10ms
  t_reason  = O(LLM_latency)                      ≈ 500-5000ms (dominant)
  t_act     = O(tool_latency)                      ≈ 0-30000ms (variable)
  t_send    = O(d_out · msg_size / bandwidth)      ≈ 1-10ms

For hierarchical 3-layer network:
  Best case:  3 × 500ms = 1.5s
  Typical:    3 × 2000ms = 6s
  Worst case: 3 × 5000ms + tool_time = 15s+
```

---

# 7. Production Architecture

## 7.0 Принцип: каждый узел — полная нода

> **Axiom P₁**: Нет отдельного "серверного" компонента.
> Каждый контейнер агента содержит ВСЁ: runtime, DHT-ноду,
> CRDT-хранилище, gossip-демон, P2P-транспорт.
> Агент = нода сети. Нода = агент.

## 7.1 System Overview

```
                        НЕТ ЦЕНТРА. Только равноправные ноды.

  Node₁ (any machine)         Node₂ (any machine)         Node₃ (any machine)
  ┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
  │ ┌─────────────────┐ │     │ ┌─────────────────┐ │     │ ┌─────────────────┐ │
  │ │  Agent Runtime   │ │     │ │  Agent Runtime   │ │     │ │  Agent Runtime   │ │
  │ │  (LLM + Tools)   │ │     │ │  (LLM + Tools)   │ │     │ │  (LLM + Tools)   │ │
  │ ├─────────────────┤ │     │ ├─────────────────┤ │     │ ├─────────────────┤ │
  │ │  P2P Transport   │◄├─────├►│  P2P Transport   │◄├─────├►│  P2P Transport   │ │
  │ │  (libp2p/QUIC)   │ │     │ │  (libp2p/QUIC)   │ │     │ │  (libp2p/QUIC)   │ │
  │ ├─────────────────┤ │     │ ├─────────────────┤ │     │ ├─────────────────┤ │
  │ │  DHT Node        │◄├─────├►│  DHT Node        │◄├─────├►│  DHT Node        │ │
  │ │  (Kademlia)      │ │     │ │  (Kademlia)      │ │     │ │  (Kademlia)      │ │
  │ ├─────────────────┤ │     │ ├─────────────────┤ │     │ ├─────────────────┤ │
  │ │  Gossip Daemon   │◄├─────├►│  Gossip Daemon   │◄├─────├►│  Gossip Daemon   │ │
  │ │  (protocol)      │ │     │ │  (protocol)      │ │     │ │  (protocol)      │ │
  │ ├─────────────────┤ │     │ ├─────────────────┤ │     │ ├─────────────────┤ │
  │ │  Local Memory    │ │     │ │  Local Memory    │ │     │ │  Local Memory    │ │
  │ │  ┌────┐ ┌─────┐ │ │     │ │  ┌────┐ ┌─────┐ │ │     │ │  ┌────┐ ┌─────┐ │ │
  │ │  │Epis│ │CRDT │ │ │     │ │  │Epis│ │CRDT │ │ │     │ │  │Epis│ │CRDT │ │ │
  │ │  │odic│ │Store│ │ │     │ │  │odic│ │Store│ │ │     │ │  │odic│ │Store│ │ │
  │ │  └────┘ └─────┘ │ │     │ │  └────┘ └─────┘ │ │     │ │  └────┘ └─────┘ │ │
  │ │  ┌──────┐┌────┐ │ │     │ │  ┌──────┐┌────┐ │ │     │ │  ┌──────┐┌────┐ │ │
  │ │  │Merkle││Caus│ │ │     │ │  │Merkle││Caus│ │ │     │ │  │Merkle││Caus│ │ │
  │ │  │DAG   ││Log │ │ │     │ │  │DAG   ││Log │ │ │     │ │  │DAG   ││Log │ │ │
  │ │  └──────┘└────┘ │ │     │ │  └──────┘└────┘ │ │     │ │  └──────┘└────┘ │ │
  │ └─────────────────┘ │     │ └─────────────────┘ │     │ └─────────────────┘ │
  │                     │     │                     │     │                     │
  │  External API (opt) │     │                     │     │  External API (opt) │
  │  :8080              │     │                     │     │  :8080              │
  └─────────────────────┘     └─────────────────────┘     └─────────────────────┘

  Любой узел можно убить — сеть продолжит работать.
  Любой новый узел может присоединиться в любой момент.
  Внешний API — опциональная точка входа, не центр управления.
```

## 7.2 Technology Stack

| Component | Technology | Justification |
|-----------|-----------|---------------|
| Agent Runtime | Python 3.11+ / asyncio | Ecosystem, LLM SDK support |
| P2P Transport | libp2p (py-libp2p) / QUIC | NAT traversal, multiplexing, peer identity |
| DHT | Kademlia (libp2p-kad) | Decentralized discovery, O(log n) lookup |
| Gossip | libp2p-gossipsub | Pub/sub without broker, topic-based |
| CRDT Store | yrs (Yjs CRDT in Rust via PyO3) | Production-grade CRDTs, Yjs compatible |
| Merkle DAG | IPLD (InterPlanetary Linked Data) | Content-addressable, verifiable graphs |
| Episodic Memory | SQLite (per-agent, embedded) | Zero-dependency, fast, local |
| Causal Log | Custom append-only + signatures | Minimal, auditable |
| LLM Backend | Claude API (primary), OpenAI (fallback) | Reasoning quality |
| Cryptography | Ed25519 (signing), X25519 (key exchange) | Fast, compact, standard |
| Container | Docker (single image for all agents) | Uniform deployment |
| Observability | OpenTelemetry (embedded in each node) | Decentralized telemetry push |

**Чего НЕТ в стеке** (и почему):
- ~~RabbitMQ~~ → заменён на libp2p-gossipsub (P2P pub/sub)
- ~~Redis (shared)~~ → заменён на CRDT-store внутри каждой ноды
- ~~PostgreSQL (shared)~~ → заменён на Causal Log + CRDT
- ~~Neo4j (shared)~~ → заменён на Merkle DAG внутри каждой ноды
- ~~Qdrant (shared)~~ → заменён на локальный vector index (hnswlib) + CRDT sync
- ~~Traefik~~ → нет центральной точки входа
- ~~Kubernetes orchestrator~~ → по желанию для ops, но НЕ для координации агентов

## 7.3 Agent Node: Single Binary Architecture

Каждый агент — один Docker-образ с полным стеком:

```dockerfile
FROM python:3.11-slim AS builder

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync --frozen

COPY src/ ./src/

FROM python:3.11-slim AS runtime
COPY --from=builder /app /app
WORKDIR /app

# Автогенерация identity при первом запуске
# или загрузка из volume (persistence)
VOLUME /data/identity   # Ed25519 keypair
VOLUME /data/memory     # SQLite + CRDT + Merkle + Causal Log

ENV AGENT_DOMAIN=general
ENV BOOTSTRAP_PEERS=""
ENV LLM_PROVIDER=anthropic
ENV LLM_MODEL=claude-sonnet-4-6
ENV P2P_PORT=4001
ENV API_PORT=8080
ENV LOG_LEVEL=INFO

HEALTHCHECK CMD python -c "import httpx; httpx.get('http://localhost:8080/health').raise_for_status()"
EXPOSE 4001 8080

# Единая точка входа — полная нода
CMD ["python", "-m", "ann_node"]
```

**Нет разделения на "orchestrator" и "agent"**. Один образ. Одна binary. Каждая нода идентична.

## 7.4 Agent Node Runtime

```python
class AgentNode:
    """
    Полная P2P нода Agent Neural Network.
    Нет зависимости от внешних сервисов. Полностью автономна.
    """

    def __init__(self, config: NodeConfig):
        # Identity (persistent)
        self.identity = Identity.load_or_create(config.identity_path)
        self.agent_id = sha256(self.identity.public_key)

        # P2P Network
        self.p2p = P2PHost(
            identity=self.identity,
            listen_addr=f"/ip4/0.0.0.0/tcp/{config.p2p_port}",
            bootstrap_peers=config.bootstrap_peers,
        )
        self.dht = KademliaDHT(self.p2p)
        self.gossip = GossipSub(self.p2p, topics=[
            "ann/tasks",       # task announcements
            "ann/results",     # task results
            "ann/trust",       # trust gossip
            "ann/evolution",   # structural changes
        ])

        # Local Memory (embedded, no external DB)
        self.episodic = SQLiteMemory(config.data_path / "episodic.db")
        self.semantic = CRDTVectorStore(config.data_path / "semantic.crdt")
        self.procedural = MerkleDAGStore(config.data_path / "procedural.dag")
        self.audit = CausalLog(config.data_path / "audit.log", self.identity)

        # Agent Brain
        self.domain = config.domain
        self.policy = LLMPolicy(config.llm_provider, config.llm_model)
        self.tools = ToolRegistry.discover(config.allowed_tools)
        self.trust_table = TrustTable()
        self.confidence_threshold = 0.6

        # No orchestrator. No supervisor. Just self.
        self.pheromone_table = PheromoneTable()  # stigmergic routing

    async def start(self):
        """Boot the node. Join the network. Start working."""
        # 1. Start P2P transport
        await self.p2p.start()

        # 2. Bootstrap into DHT
        await self.dht.bootstrap()

        # 3. Announce capabilities
        await self.dht.provide(
            key=f"capability:{self.domain}",
            value=AgentCapability(
                agent_id=self.agent_id,
                domain=self.domain,
                tools=list(self.tools.keys()),
                trust=self.trust_table.self_trust,
            )
        )

        # 4. Subscribe to gossip topics
        self.gossip.subscribe("ann/tasks", self._on_task_announced)
        self.gossip.subscribe("ann/trust", self._on_trust_gossip)
        self.gossip.subscribe("ann/evolution", self._on_evolution_event)

        # 5. Start main loops (concurrent)
        await asyncio.gather(
            self._task_loop(),
            self._gossip_loop(),
            self._sync_loop(),
            self._evolution_loop(),
        )

    async def _task_loop(self):
        """Process incoming tasks and self-initiated work."""
        while True:
            msg = await self.inbox.get()

            # Evaluate: can I handle this?
            confidence = await self.policy.evaluate_confidence(
                task=msg.payload,
                memory=await self._recall(msg.payload),
                tools=self.tools,
            )

            if confidence >= self.confidence_threshold:
                # Execute locally
                result = await self._execute(msg.payload)
                await self._publish_result(msg, result)
                await self._update_trust_after_task(msg, result)
            else:
                # Route to more competent peer (no "escalation to supervisor")
                peer = await self._find_best_peer(msg.payload)
                if peer:
                    await self.p2p.send(peer, msg)
                    self.pheromone_table.deposit(msg.payload.domain, peer)
                else:
                    # No one can help. Try partial solution.
                    result = await self._execute_best_effort(msg.payload)
                    await self._publish_result(msg, result, partial=True)

    async def _find_best_peer(self, task) -> PeerID | None:
        """Stigmergic + DHT routing. No central dispatcher."""
        # 1. Check pheromone trails first (cached successful routes)
        pheromone_peer = self.pheromone_table.get_best(task.domain)
        if pheromone_peer and self.trust_table.get(pheromone_peer) > 0.5:
            return pheromone_peer

        # 2. DHT lookup by capability
        candidates = await self.dht.find_providers(f"capability:{task.domain}")

        # 3. Score candidates (all local information, no central query)
        scored = []
        for c in candidates:
            if c.agent_id == self.agent_id:
                continue
            score = (
                0.4 * self.trust_table.get(c.agent_id, 0.3)
                + 0.3 * (1 - c.load)
                + 0.2 * self._domain_match(c.domain, task.domain)
                + 0.1 * self.pheromone_table.get(task.domain, c.agent_id)
            )
            scored.append((c.agent_id, score))

        if scored:
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[0][0]
        return None

    async def _gossip_loop(self):
        """Periodic trust and state gossip. Pure P2P."""
        while True:
            await asyncio.sleep(self.gossip_interval)

            # Pick random neighbor
            peer = self.p2p.random_connected_peer()
            if not peer:
                continue

            # Exchange trust observations
            my_observations = self.trust_table.recent_observations(limit=10)
            peer_observations = await self.p2p.exchange(
                peer, "trust_gossip", my_observations
            )

            # Merge (trust-weighted: I weight their observations by my trust in them)
            peer_trust = self.trust_table.get(peer, 0.3)
            for obs in peer_observations:
                self.trust_table.merge_observation(obs, weight=peer_trust)

    async def _sync_loop(self):
        """CRDT and Merkle DAG synchronization with peers."""
        while True:
            await asyncio.sleep(self.sync_interval)

            peer = self.p2p.random_connected_peer()
            if not peer:
                continue

            # Semantic memory: CRDT state exchange
            peer_crdt_state = await self.p2p.exchange(
                peer, "crdt_sync", self.semantic.get_state_vector()
            )
            self.semantic.merge(peer_crdt_state)

            # Procedural memory: Merkle root comparison + diff
            peer_root = await self.p2p.exchange(
                peer, "merkle_sync", self.procedural.root_hash()
            )
            if peer_root != self.procedural.root_hash():
                diff = await self._merkle_diff(peer)
                self.procedural.merge(diff)

    async def _evolution_loop(self):
        """Decentralized structural evolution. No central evolver."""
        while True:
            await asyncio.sleep(self.evolution_interval)

            # 1. Self-assessment
            my_fitness = self._compute_local_fitness()

            # 2. Compare with neighbors via gossip
            peer_fitnesses = await self._collect_peer_fitness()

            # 3. Am I underperforming?
            avg_fitness = sum(peer_fitnesses.values()) / max(len(peer_fitnesses), 1)
            if my_fitness < avg_fitness * 0.5:
                # Self-improvement: mutate my own parameters
                await self._self_mutate()

            # 4. Topology maintenance (local decisions only)
            await self._topology_maintenance()
```

## 7.5 Deployment: Zero-Infrastructure Mode

```yaml
# docker-compose.ann.yml
# Минимальная конфигурация: ТОЛЬКО агенты. Никакой инфраструктуры.

services:
  # Seed node (первый узел, точка входа для остальных)
  seed:
    image: ann-node:latest
    environment:
      - AGENT_DOMAIN=general
      - P2P_PORT=4001
      - API_PORT=8080
      - BOOTSTRAP_PEERS=  # seed node — сам себе bootstrap
    ports:
      - "4001:4001"  # P2P
      - "8080:8080"  # External API (optional entry point)
    volumes:
      - seed-data:/data

  # Остальные агенты — подключаются к seed
  agent-backend:
    image: ann-node:latest
    environment:
      - AGENT_DOMAIN=backend
      - BOOTSTRAP_PEERS=/dns4/seed/tcp/4001/p2p/${SEED_PEER_ID}
    volumes:
      - backend-data:/data
    deploy:
      replicas: 3

  agent-frontend:
    image: ann-node:latest
    environment:
      - AGENT_DOMAIN=frontend
      - BOOTSTRAP_PEERS=/dns4/seed/tcp/4001/p2p/${SEED_PEER_ID}
    volumes:
      - frontend-data:/data
    deploy:
      replicas: 2

  agent-data:
    image: ann-node:latest
    environment:
      - AGENT_DOMAIN=data
      - BOOTSTRAP_PEERS=/dns4/seed/tcp/4001/p2p/${SEED_PEER_ID}
    volumes:
      - data-data:/data
    deploy:
      replicas: 2

  agent-qa:
    image: ann-node:latest
    environment:
      - AGENT_DOMAIN=qa
      - BOOTSTRAP_PEERS=/dns4/seed/tcp/4001/p2p/${SEED_PEER_ID}
    volumes:
      - qa-data:/data
    deploy:
      replicas: 2

  # Observability (единственный "инфраструктурный" компонент — опциональный)
  otel-collector:
    image: otel/opentelemetry-collector-contrib:latest
    ports: ["4317:4317"]

volumes:
  seed-data:
  backend-data:
  frontend-data:
  data-data:
  qa-data:

# ОБРАТИТЕ ВНИМАНИЕ:
# - НЕТ RabbitMQ
# - НЕТ Redis (shared)
# - НЕТ PostgreSQL
# - НЕТ Neo4j
# - НЕТ Qdrant
# - НЕТ Traefik
# - НЕТ Orchestrator service
#
# Seed node — НЕ центр. Это просто первый узел для bootstrap.
# Если seed упадёт, остальные продолжат работать через уже установленные P2P связи.
# Новые ноды смогут подключиться через любой живой узел.
```

### Multi-Machine Deployment

```bash
# Machine A (Moscow):
docker run -d \
  -e AGENT_DOMAIN=backend \
  -e BOOTSTRAP_PEERS=/ip4/185.x.x.x/tcp/4001/p2p/QmSeedPeerID \
  -p 4001:4001 \
  -v agent-a:/data \
  ann-node:latest

# Machine B (Frankfurt):
docker run -d \
  -e AGENT_DOMAIN=frontend \
  -e BOOTSTRAP_PEERS=/ip4/185.x.x.x/tcp/4001/p2p/QmSeedPeerID \
  -p 4001:4001 \
  -v agent-b:/data \
  ann-node:latest

# Machine C (New York):
docker run -d \
  -e AGENT_DOMAIN=data \
  -e BOOTSTRAP_PEERS=/dns4/node-a.ann.example.com/tcp/4001/p2p/QmPeerA,/dns4/node-b.ann.example.com/tcp/4001/p2p/QmPeerB \
  -p 4001:4001 \
  -v agent-c:/data \
  ann-node:latest

# Все три ноды находят друг друга через DHT.
# Формируют P2P mesh. Синхронизируют память через CRDTs.
# Маршрутизируют задачи через стигмергию.
# Никакого центрального сервера между ними.
```

## 7.6 Observability: Decentralized Monitoring

```
Каждая нода экспортирует метрики локально. Нет центрального мониторинга.

Режим 1 — Pull (Prometheus-compatible):
  Каждая нода → :9090/metrics endpoint
  Любой Prometheus может scrape любое подмножество нод
  Работает в том числе без центрального Prometheus (peer queries)

Режим 2 — Push (OpenTelemetry):
  Каждая нода → push to nearest OTel collector
  Collectors могут быть распределены (один на машину)
  Collectors агрегируют и forwarded в Grafana/Loki/etc.

Режим 3 — Gossip Metrics (zero-infra):
  Метрики распространяются через gossip
  Любая нода может агрегировать сетевую картину из gossip
  Приблизительно, но без внешних зависимостей

Локальные метрики каждой ноды:

Agent-level:
  - ann_node_latency_seconds{domain}
  - ann_node_confidence_avg
  - ann_node_trust_received_avg        # как меня оценивают другие
  - ann_node_trust_given_avg           # как я оцениваю других
  - ann_node_messages_total{direction, type}
  - ann_node_llm_tokens_total{direction}
  - ann_node_tools_used_total{tool}
  - ann_node_tasks_total{status}
  - ann_node_peers_connected

Network-level (агрегируется из gossip):
  - ann_network_estimated_size         # оценка через DHT
  - ann_network_gossip_rounds_total
  - ann_network_crdt_sync_total
  - ann_network_consensus_total{outcome}

Alerts (локальные, на каждой ноде):
  - ann_node_peers_connected < 2       → "Isolation risk: low peer count"
  - ann_node_trust_received_avg < 0.2  → "Reputation crisis"
  - ann_node_crdt_sync_lag > 60s       → "Memory sync falling behind"
  - ann_node_tasks_total{status=failed} > 10  → "High failure rate"
```

## 7.7 Integration with Existing xteam-agents

```
Миграция от текущей централизованной xteam-agents к P2P ANN:

Phase 0 — Compatibility Layer:
  Обернуть каждого из 21 существующих агентов в AgentNode.
  Внутри всё ещё LangGraph + существующий код.
  Снаружи — P2P интерфейс.
  Существующая инфраструктура (Redis, Neo4j, etc.) работает параллельно.

Phase 1 — Dual Mode:
  Агенты работают через ОБОИХ:
  - Старый путь: LangGraph → RabbitMQ → shared DB
  - Новый путь: P2P gossip → CRDT → local memory
  Сравнение результатов. A/B тестирование.

Phase 2 — P2P Primary:
  P2P становится основным каналом.
  Старая инфраструктура — fallback.
  CRDT-store синхронизируется с legacy DB (bridge).

Phase 3 — Legacy Shutdown:
  Отключение RabbitMQ, shared Redis, shared PostgreSQL, Neo4j, Qdrant.
  Полный переход на:
  - libp2p вместо RabbitMQ
  - CRDT Store вместо shared Redis
  - Merkle DAG вместо Neo4j
  - Local hnswlib вместо Qdrant
  - Causal Log вместо PostgreSQL audit

Phase 4 — Federation:
  xteam-agents ANN федерируется с внешними ANN-сетями.
  MAGIC human-AI collaboration сохраняется как отдельный gossip topic.
```

## 7.8 Security Model

```
Identity:
  Каждая нода имеет Ed25519 keypair.
  agent_id = SHA-256(public_key).
  Все сообщения подписаны. Все CRDT-записи подписаны.
  Подделка невозможна без private key.

Transport:
  P2P connections шифрованы через Noise Protocol (libp2p default).
  Perfect Forward Secrecy через X25519 ephemeral key exchange.
  Каждое соединение — отдельный encrypted channel.

Anti-Sybil:
  Новый агент: Ψ_default = 0.3 (почти бесполезен).
  Набор доверия: только через Proof of Useful Work.
  Участие в консенсусе: Ψ > 0.5 (невозможно для Sybil-нод).
  Участие в валидации: Ψ > 0.6.

Anti-Eclipse:
  DHT k-buckets обеспечивают разнообразие peers.
  Ноды поддерживают соединения в разных IP-подсетях.
  Random peer selection в gossip предотвращает изоляцию.

Data Integrity:
  Merkle DAG: любое изменение данных меняет root hash.
  CRDT: операции коммутативны и идемпотентны, невозможно "откатить".
  Causal Log: append-only + подписи + свидетели.

Network Security:
  DoS protection: backpressure + rate limiting на P2P transport.
  Max connections per peer: bounded.
  Gossip rate limiting: max N messages per interval.
```

---

# Appendix A: Notation Reference

| Symbol | Meaning |
|--------|---------|
| `A` | Set of agents |
| `aᵢ` | Agent i |
| `G = (A, E, W)` | Weighted directed communication graph |
| `E` | Edge set (communication channels) |
| `W(i,j)` | Edge weight (trust × relevance) |
| `πᵢ` | Agent i's policy |
| `Mᵢ` | Agent i's memory |
| `Tᵢ` | Agent i's tool set |
| `Sᵢ` | Agent i's internal state |
| `Lᵢ` | Agent i's cognitive level |
| `κᵢ` | Agent i's confidence |
| `Ψ(i,j,t)` | Trust from agent i to agent j at time t |
| `Ω` | Topology evolution operator |
| `Φ` | Input encoder |
| `Σ` | Signal protocol |
| `R` | Reward function |
| `L` | Graph Laplacian |
| `λ₂(L)` | Algebraic connectivity (Fiedler value) |
| `F` | Global state transition function |
| `G` (genome) | Evolutionary genome encoding |
| `F` (fitness) | Evolutionary fitness function |

# Appendix B: Key References

1. Bernstein et al. "The Complexity of Decentralized Control of MDPs" (2002) — NEXP-completeness of Dec-POMDPs
2. Stanley & Miikkulainen "NEAT: Evolving Neural Networks through Augmenting Topologies" (2002)
3. Zhang et al. "G-Designer: Architecting Multi-agent Communication Topologies via GNNs" (2024)
4. Ruan et al. "Towards a Science of Scaling Agent Systems" (2025) — Google Research
5. Chen & Wang "Neurons as Autonomous Agents: A Biologically Inspired Framework" (2025)
6. Li et al. "RepuNet: Reputation System for Generative Multi-agent Systems" (2025)
7. "Rethinking Reliability of MAS: BFT Perspective" (2025) — CP-WBFT
8. "MAC-SPGG: Incentivizing Strategic Cooperation in Multi-LLM Systems" (2025)
9. "Emergent Coordination in Multi-Agent Language Models" (2025)
10. "Revisiting Gossip Protocols for Agentic MAS" (2025)

# Appendix C: Glossary

| Term | Definition |
|------|-----------|
| **Agent-Neuron** | Autonomous AI agent serving as a computational unit in ANN |
| **Cognitive Level** | L1 (reactive) through L4 (emergent-hub) capability classification |
| **Trust Weight** | Dynamic edge weight based on historical interaction quality |
| **Cognitive Shard** | Partition of agents by domain for scalability |
| **Bridge Agent** | Emergent role: agent with high betweenness centrality connecting clusters |
| **Gossip Round** | Periodic reputation propagation step via P2P |
| **Escalation** | Low-confidence agent routing task to more competent peer (NOT upward — lateral) |
| **Adversarial Pair** | Agent-Critic duo for quality verification |
| **Fiedler Value** | Second smallest eigenvalue of Laplacian, measures graph connectivity |
| **AGC** | Agent-Graph Consensus — leaderless BFT protocol based on Hashgraph |
| **DHT** | Distributed Hash Table — decentralized agent discovery (Kademlia) |
| **CRDT** | Conflict-free Replicated Data Type — eventually consistent shared state |
| **Merkle DAG** | Content-addressable directed acyclic graph for procedural memory |
| **Stigmergy** | Indirect coordination through environment traces (pheromone trails) |
| **Pheromone Trail** | Cached successful task routing path, decays over time |
| **Proof of Useful Work** | Trust earned through demonstrated competence, not registration |
| **Federation** | Multiple ANN networks interconnected while maintaining autonomy |
| **Online Evolution** | Runtime topology optimization without system stop |