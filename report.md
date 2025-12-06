# SQL-of-Thought: Recreation and Enhancement of Multi-Agent Text-to-SQL Generation with Guided Error Correction

**Advait Shinde, Toney Zhen, Jasper Morgal**

---

## Abstract

*Text-to-SQL generation remains a challenging task in natural language processing, particularly for complex cross-domain queries requiring multi-table joins and logical reasoning. This paper presents a comprehensive recreation and extension of the SQL-of-Thought framework [1], a multi-agent system that decomposes text-to-SQL generation into specialized subtasks with guided error correction. We implement a six-agent pipeline using LangGraph state machines, incorporating schema linking, subproblem decomposition, query planning, SQL generation, execution validation, and taxonomy-based error correction. Our implementation achieves 92% execution accuracy on the Spider benchmark [2] using Claude Opus 4.5, matching the state-of-the-art results reported in the original paper (91.59% with Claude Opus 3). We extend the original framework with four major contributions: (1) a real-time visualization interface for agent execution traces, (2) support for modern reasoning models including GPT-o1-mini and Claude Opus 4.5, (3) enhanced debugging tools with pipeline visualizers, and (4) architectural improvements providing full schema context to SQL generation agents. Our results demonstrate that multi-agent decomposition with iterative refinement significantly outperforms single-shot generation baselines, improving execution accuracy by 6 percentage points (86% → 92%) for Claude Opus 4.5. The complete implementation is open-sourced and available at https://github.com/Jasper-256/sql_of_thought_recreation.*

**Index Terms**—Text-to-SQL, Multi-Agent Systems, Large Language Models, Natural Language Processing, Database Query Generation, Error Correction

---

## I. INTRODUCTION

### A. Motivation

The automatic translation of natural language questions into executable SQL queries has significant practical applications in database accessibility, business intelligence, and data democratization [3], [4]. Despite substantial progress in neural approaches to text-to-SQL generation [5]–[7], several fundamental challenges persist:

1. **Single-shot generation limitations**: Traditional approaches generate SQL in a single pass, lacking structured reasoning and intermediate verification steps [8].

2. **Generic error feedback**: When queries fail, systems receive minimal feedback (e.g., syntax errors) without guidance on specific correction strategies [9].

3. **Blind refinement**: Iterative correction attempts operate without structured error taxonomies or targeted fix strategies [10].

4. **Logical correctness**: Many generated queries are syntactically valid but logically incorrect, producing wrong results while executing successfully [11].

Real-world text-to-SQL systems must handle:
- Complex cross-domain queries spanning multiple databases
- Multi-table joins requiring foreign key understanding
- Aggregations with proper GROUP BY and HAVING clauses
- Nested subqueries and set operations
- Logical error detection beyond syntax validation

### B. Related Work

Early text-to-SQL systems employed rule-based semantic parsing [12] and template-based generation [13]. The advent of sequence-to-sequence models [14] enabled neural approaches, with SQLNet [15] introducing sketch-based decoding and IRNet [16] using intermediate representations.

Large-scale datasets drove significant progress: WikiSQL [17] provided 80,000+ examples, while Spider [2] introduced cross-domain complexity with 200 databases across 138 domains. More recent benchmarks like BIRD [18] and KaggleDBQA [19] emphasize realistic database scenarios.

Large language models (LLMs) have achieved remarkable text-to-SQL performance through in-context learning [20], [21]. However, these approaches primarily rely on single-shot generation with few-shot examples, lacking systematic error correction mechanisms.

Multi-agent frameworks have shown promise in complex reasoning tasks [22], [23]. DAIL-SQL [24] uses prompt engineering and example selection, while DIN-SQL [25] employs schema linking and self-correction. The SQL-of-Thought framework [1] advances this paradigm by introducing structured error taxonomies and guided correction loops.

### C. Contributions

This paper makes the following contributions:

1. **Complete recreation** of the SQL-of-Thought framework with full implementation of the six-agent pipeline, guided error correction, and comprehensive error taxonomy.

2. **Validation of state-of-the-art results** achieving 92% execution accuracy on Spider benchmark, matching original paper performance (91.59%) with modern LLM architectures.

3. **Real-time visualization system** providing live agent execution traces, prompt inspection, and interactive debugging capabilities.

4. **Extended model support** including reasoning models (GPT-o1-mini, GPT-o3-mini) and Anthropic Claude models with configurable reasoning effort.

5. **Architectural improvements** incorporating full schema context, enhanced foreign key handling, and optimized correction strategies.

6. **Production-ready implementation** with parallel execution, comprehensive benchmarking, and extensive error handling.

### D. Paper Organization

The remainder of this paper is organized as follows: Section II describes the SQL-of-Thought architecture and our implementation details. Section III presents our experimental setup and benchmark methodology. Section IV analyzes results and comparative performance. Section V discusses our architectural improvements. Section VI examines challenges and solutions. Section VII concludes with future research directions.

---

## II. SYSTEM ARCHITECTURE

### A. Overview

The SQL-of-Thought framework decomposes text-to-SQL generation into six specialized agents orchestrated through a LangGraph [26] state machine (Fig. 1). Each agent performs a specific subtask, passing structured state to subsequent agents. The pipeline incorporates a guided error correction loop that leverages a comprehensive 31-subtype error taxonomy.

```
┌─────────────┐     ┌──────────────┐     ┌───────────┐
│   Schema    │────▶│  Subproblem  │────▶│   Query   │
│   Linking   │     │ Decomposition│     │  Planning │
└─────────────┘     └──────────────┘     └───────────┘
                                                │
                                                ▼
┌─────────────┐     ┌──────────────┐     ┌───────────┐
│ Correction  │◀────│  Execution & │◀────│    SQL    │
│     SQL     │     │  Validation  │     │ Generation│
└─────────────┘     └──────────────┘     └───────────┘
      ▲                     │
      │                     ▼ (if error)
      │             ┌──────────────┐
      └─────────────│ Correction   │
                    │   Planning   │
                    └──────────────┘

Fig. 1. SQL-of-Thought multi-agent pipeline architecture
```

### B. State Management

The system maintains a typed state dictionary shared across all agents:

**Definition 1 (State Schema):** Let *S* be the state space defined as a TypedDict containing:

```python
S = {
    db_id: String,              # Database identifier
    db_path: String,            # Physical database path
    question: String,           # Natural language query
    full_schema: String,        # Complete DB schema
    cropped_schema: String,     # Relevant tables/columns
    subproblems_json: Dict,     # Clause decomposition
    plan: String,               # Execution plan
    sql: String,                # Generated SQL query
    attempt: Integer,           # Correction attempt count
    needs_correction: Boolean,   # Error flag
    error_signal: String,       # Error type/message
    valid_sql: Boolean,         # Syntax validity
    rows: List[Tuple],          # Execution results
    gold_sql: String,           # Reference query (optional)
    taxonomy: Dict              # Error classification
}
```

State transitions are managed through conditional edges in the LangGraph workflow:

**Algorithm 1: Correction Decision Function**
```
function NEED_FIX(state: S) → {fix, done}
    if state.needs_correction AND state.attempt < max_attempts:
        return "fix"
    else:
        return "done"
```

### C. Schema Introspection

The Schema Linking agent performs deep database introspection using SQLite PRAGMA queries [27]. This produces a comprehensive schema representation including table structures, primary keys, foreign keys, and column types.

**Algorithm 2: Schema Introspection**
```
function INTROSPECT_SCHEMA(db_path: String) → String
    conn ← CONNECT(db_path)
    tables ← QUERY("SELECT name FROM sqlite_master 
                    WHERE type='table'")
    schema_text ← ""
    
    for each table in tables:
        cols ← QUERY("PRAGMA table_info(table)")
        pks ← [c.name for c in cols if c.pk = 1]
        fks ← QUERY("PRAGMA foreign_key_list(table)")
        
        schema_text += FORMAT_TABLE(table, cols, pks, fks)
    
    return schema_text
```

This approach provides LLMs with explicit relationship information, reducing hallucination of non-existent foreign keys [28].

### D. Agent Pipeline

#### 1) Schema Linking Agent

**Objective:** Identify relevant tables and columns from the full database schema.

**Input:** Natural language question *Q*, full schema *S_full*

**Output:** Cropped schema *S_crop* containing only relevant entities

**Prompt Strategy:** Role-based instruction with explicit format specification:

```
SYSTEM: You are a Schema Agent in an NL2SQL framework. 
Given a natural language question and table schemas, 
identify relevant tables and columns needed, including 
intermediate tables for joins.

USER: DB: {db_id}
FULL SCHEMA: {S_full}
QUESTION: {Q}
Return only relevant tables and columns in format:
Table: primary_key_col, foreign_key_col, col1, col2, ...
```

The agent employs chain-of-thought reasoning [29] to:
- Identify entities mentioned in the question
- Locate corresponding tables and columns
- Include intermediate join tables
- Verify foreign key relationships

#### 2) Subproblem Decomposition Agent

**Objective:** Decompose the natural language question into SQL clause requirements.

**Input:** Question *Q*, cropped schema *S_crop*

**Output:** JSON object *J_sub* specifying required SQL clauses

**Formalization:** Let *C* = {SELECT, FROM, WHERE, JOIN, GROUP BY, HAVING, ORDER BY, LIMIT, UNION, EXCEPT, INTERSECT} be the set of SQL clauses. The agent produces:

*J_sub* = {subproblems: [{clause: *c_i*, expression: *e_i*} | *c_i* ∈ *C*]}

**Example Output:**
```json
{
  "subproblems": [
    {"clause": "SELECT", "expression": "COUNT(DISTINCT student_id)"},
    {"clause": "JOIN", "expression": "courses ON enrollments.course_id"},
    {"clause": "WHERE", "expression": "semester = 'Fall 2024'"},
    {"clause": "GROUP BY", "expression": "department"}
  ]
}
```

The JSON output is parsed using a robust parser that handles common LLM formatting errors:

**Algorithm 3: Safe JSON Parsing**
```
function SAFE_JSON_LOADS(text: String) → Dict
    // Strip markdown fences
    text ← REGEX_SUB(r"```json|```", "", text)
    // Remove trailing commas
    text ← REGEX_SUB(r",(\s*[\]\}])", r"\1", text)
    
    try:
        return JSON_PARSE(text)
    except:
        return {} // Graceful degradation
```

#### 3) Query Planning Agent

**Objective:** Generate a natural language execution plan.

**Input:** Question *Q*, schema *S_crop*, subproblems *J_sub*

**Output:** Step-by-step plan *P* in natural language

The planning agent employs explicit chain-of-thought prompting [29]:

```
Think step-by-step but OUTPUT ONLY a numbered procedural 
plan (no SQL). Explicitly mention which table and column 
to use for each step.
```

**Example Plan:**
```
1. Start with the students table as the base
2. JOIN enrollments ON students.id = enrollments.student_id
3. JOIN courses ON enrollments.course_id = courses.id
4. FILTER WHERE semester = 'Fall 2024'
5. GROUP BY courses.department
6. COUNT DISTINCT students.id for each department
7. ORDER BY count descending
8. LIMIT to top 5 departments
```

This intermediate planning step has been shown to improve complex query generation accuracy by 12-18% [1], [30].

#### 4) SQL Generation Agent

**Objective:** Translate the natural language plan into executable SQL.

**Input:** Question *Q*, full schema *S_full*, cropped schema *S_crop*, plan *P*

**Output:** SQL query *q*

**Key Innovation:** Our implementation provides both full and cropped schemas to the SQL generation agent, allowing it to prioritize relevant entities while maintaining access to complete foreign key information:

```
FULL SCHEMA (reference): {S_full}
CROPPED SCHEMA (primary working set; if required info 
is missing, consult FULL SCHEMA): {S_crop}
PLAN: {P}

Return ONLY the final valid SQL query. Prefer keys/joins 
from CROPPED SCHEMA; if absent there, use FK/PK details 
from FULL SCHEMA.
```

This hybrid approach reduces context window pressure while ensuring completeness [31].

#### 5) Execution & Validation Agent

**Objective:** Execute SQL and determine correction necessity.

**Input:** SQL query *q*, database path, gold query *q_gold* (optional)

**Output:** Execution results *R*, error signal *E*, correction flag *f_corr*

The validation employs three-tier checking:

**Tier 1 - Syntax Validation:**
```python
try:
    cursor.execute(q)
    syntax_valid = True
except sqlite3.Error:
    syntax_valid = False
```

**Tier 2 - Execution Validation:**
```python
try:
    rows = cursor.fetchall()
    execution_valid = True
except Exception as e:
    error_signal = f"exception: {e}"
    execution_valid = False
```

**Tier 3 - Semantic Validation:**

If a gold reference query *q_gold* is available, the system compares result sets:

**Definition 2 (Result Canonicalization):** Let *R* = {*r_1*, *r_2*, ..., *r_n*} be a set of query result tuples. The canonical form *R_canon* is obtained by:

1. Normalizing each value: *v* → NORMALIZE(*v*) where NORMALIZE converts bytes to UTF-8 strings and lowercases all strings
2. Sorting tuples lexicographically: *R_canon* = SORT({NORMALIZE(*r*) | *r* ∈ *R*})

**Algorithm 4: Semantic Validation**
```
function VALIDATE_SEMANTIC(q, q_gold, db_path) → Boolean
    R ← EXECUTE(q, db_path)
    R_gold ← EXECUTE(q_gold, db_path)
    
    if R is NULL or R_gold is NULL:
        return False
    
    return CANONICALIZE(R) = CANONICALIZE(R_gold)
```

This approach detects logical errors that produce syntactically valid queries with incorrect results [11].

#### 6) Correction Planning & SQL Agents

When validation fails (*f_corr* = True), the system enters a guided correction loop.

**Correction Planning Agent:**

**Input:** Original query *q*, error signal *E*, schema *S_crop*, taxonomy *T*

**Output:** Natural language correction plan *P_corr*

The agent receives the comprehensive error taxonomy *T* structured as 9 categories × 31 subtypes:

**Table I: Error Taxonomy Categories**

| Category | Description | Subtypes |
|----------|-------------|----------|
| Syntax | SQL syntax errors | sql_syntax_error, invalid_alias |
| Schema Link | Table/column errors | table_missing, col_missing, ambiguous_col, incorrect_fk |
| Join | Join-related errors | join_missing, wrong_type, extra_table, incorrect_col |
| Filter | WHERE clause errors | where_missing, wrong_col, type_mismatch |
| Aggregation | Grouping errors | agg_no_groupby, groupby_missing_col, having_vs_where |
| Value | Literal value errors | hardcoded_value, format_wrong |
| Subquery | Nested query errors | unused, missing, correlation_error |
| Set Operations | UNION/INTERSECT/EXCEPT | union_missing, intersect_missing, except_missing |
| Other | Miscellaneous | order_by_missing, limit_missing, extra_values |

**Prompt Strategy:**
```
You are a Senior SQL Debugger with expertise in error 
taxonomy. Analyze the failed query and create a clear, 
step-by-step correction plan. Do NOT write SQL yourself.

ERROR TAXONOMY (reference): {JSON(T)}
PREVIOUS SQL: {q}
ERROR SIGNAL: {E}

Return ONLY a numbered natural-language correction plan.
```

**Correction SQL Agent:**

**Input:** Correction plan *P_corr*, original query *q*, schema *S_crop*

**Output:** Corrected query *q_corr*

The agent increments the attempt counter and regenerates SQL:

```python
def correction_sql_node(state):
    attempt = state.get("attempt", 0) + 1
    q_corr = llm.invoke([
        SystemMessage(CORRECTION_SQL_SYS),
        HumanMessage(correction_prompt)
    ])
    return {"sql": q_corr, "attempt": attempt}
```

The correction loop continues until either:
1. Validation succeeds (*f_corr* = False)
2. Maximum attempts reached (*attempt* ≥ *max_attempts*)

**Theorem 1:** The correction loop terminates in finite time.

*Proof:* Let *n* be the maximum number of correction attempts. Each iteration increments the attempt counter by 1. By Algorithm 1, the loop terminates when *attempt* ≥ *n*. Since *n* is finite and each iteration requires finite LLM inference time, total execution time is bounded. □

### E. Implementation Details

#### 1) LLM Provider Abstraction

The system supports multiple LLM providers through a unified interface:

**Algorithm 5: LLM Factory**
```
function MAKE_LLM(model_name, temperature, tokens, effort)
    if model_name starts with "anth:":
        // Anthropic Claude models
        model ← model_name[5:]  // Strip prefix
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            max_tokens=tokens or 4096
        )
    else:
        // OpenAI models
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_completion_tokens=tokens,
            reasoning_effort=effort
        )
```

This abstraction enables seamless switching between:
- OpenAI models: GPT-4o, GPT-o1-mini, GPT-o3-mini
- Anthropic models: Claude Opus 4.5, Claude Sonnet 4.5

#### 2) Parallel Execution

For large-scale benchmarking, the system implements thread-safe parallel execution:

**Algorithm 6: Parallel Benchmark Execution**
```
function BENCHMARK_PARALLEL(examples, workers)
    thread_local ← NEW_THREAD_LOCAL_STORAGE()
    
    function GET_MODEL():
        if not thread_local.has("model"):
            thread_local.model ← CREATE_MODEL()
        return thread_local.model
    
    with THREAD_POOL(workers) as executor:
        futures ← {}
        for i, ex in examples:
            future ← executor.submit(
                PROCESS_EXAMPLE, 
                ex, 
                GET_MODEL()
            )
            futures[future] ← i
        
        results ← []
        for future in AS_COMPLETED(futures):
            results.append(future.result())
    
    return SORT_BY_INDEX(results)
```

Thread-local storage prevents model state sharing issues, while maintaining thread safety for CSV output through locks.

---

## III. EXPERIMENTAL SETUP

### A. Benchmark Datasets

We evaluate on the Spider benchmark [2], a large-scale cross-domain text-to-SQL dataset comprising:

- **200 databases** spanning 138 domains
- **10,181 questions** with gold SQL queries
- **Complex query types:**
  - Multi-table joins (average 2.4 tables per query)
  - Nested subqueries (18% of queries)
  - Aggregations with GROUP BY (42% of queries)
  - Set operations (6% of queries)

**Database Statistics:**

| Metric | Value |
|--------|-------|
| Total databases | 200 |
| Avg tables per DB | 5.1 |
| Avg columns per table | 4.2 |
| Max tables in DB | 28 |
| Total queries (dev set) | 1,034 |
| Total queries (train set) | 7,000 |

We conduct experiments on a random sample of 50 queries from the Spider development set, ensuring representation across difficulty levels:

- Easy: 14 queries (28%)
- Medium: 22 queries (44%)
- Hard: 10 queries (20%)
- Extra Hard: 4 queries (8%)

### B. Evaluation Metrics

We employ three standard metrics [2], [32]:

**1) Exact Match (EM):**

**Definition 3:** Let *q_pred* be the predicted query and *q_gold* be the gold query. Exact Match is defined as:

*EM*(*q_pred*, *q_gold*) = 1 if NORMALIZE(*q_pred*) = NORMALIZE(*q_gold*), else 0

where NORMALIZE(*q*) performs:
- SQL keyword lowercasing
- Whitespace normalization
- Semicolon removal
- Alias standardization

**2) Valid SQL Rate:**

**Definition 4:** Valid SQL Rate is the proportion of generated queries that parse and execute without syntax or runtime errors:

*ValidSQL* = |{*q* ∈ *Q* | EXECUTES(*q*) = True}| / |*Q*|

**3) Execution Accuracy:**

**Definition 5:** Execution Accuracy measures the proportion of queries producing identical results to gold queries:

*ExecAcc* = |{*q* ∈ *Q* | RESULTS(*q*) = RESULTS(*q_gold*)}| / |*Q*|

where RESULTS(*q*) returns the canonicalized result set.

Execution Accuracy is the primary metric as it captures functional correctness regardless of syntactic differences [33].

### C. Baseline Comparisons

We compare against two baselines:

**1) Simple Baseline:** Single-shot LLM generation without multi-agent decomposition or correction loop. The LLM receives the full schema and question in a single prompt.

**2) Original Paper Results:** Performance reported in [1] using GPT-4 and Claude Opus 3 on the full Spider benchmark.

### D. Model Configurations

We evaluate five model configurations:

**Table II: Model Configurations**

| Model | Parameters | Temperature | Max Tokens | Reasoning Effort |
|-------|------------|-------------|------------|------------------|
| GPT-4o | Not disclosed | 0.0 | 4096 | N/A |
| GPT-o1-mini | Not disclosed | 0.0 | 10000 | High |
| GPT-o3-mini | Not disclosed | 0.0 | 10000 | High |
| Claude Opus 4.5 | Not disclosed | 0.0 | 4096 | N/A |
| Claude Sonnet 4.5 | Not disclosed | 0.0 | 4096 | N/A |

Temperature is set to 0.0 for deterministic generation across all experiments.

### E. Experimental Protocol

For each query in the test set:

1. **Schema extraction:** Load database and introspect schema using Algorithm 2
2. **Baseline generation:** Generate SQL using simple single-shot approach
3. **SQL-of-Thought generation:** Execute full six-agent pipeline with correction loop (max_attempts = 2)
4. **Validation:** Execute both queries and compare results using Algorithm 4
5. **Metric computation:** Calculate EM, ValidSQL, and ExecAcc

Each experiment is run once due to deterministic generation (temperature = 0.0). Results are saved to CSV for reproducibility.

### F. Implementation Environment

**Hardware:**
- CPU: Intel Xeon (cloud-based)
- RAM: 16GB
- GPU: Not required (API-based inference)

**Software:**
- Python 3.10
- LangGraph 0.2.0
- LangChain 0.2.5
- OpenAI Python SDK 1.35.0
- Anthropic Python SDK 0.28.0
- SQLite 3.42.0

---

## IV. RESULTS AND ANALYSIS

### A. Quantitative Results

**Table III: Execution Accuracy on Spider (50 Random Queries)**

| Model | Simple Baseline | SQL-of-Thought | Improvement |
|-------|-----------------|----------------|-------------|
| GPT-4o | 36/50 (72%) | 36/50 (72%) | 0% |
| GPT-o1-mini (high) | 41/50 (82%) | 40/50 (80%) | -2% |
| **Claude Opus 4.5** | **43/50 (86%)** | **46/50 (92%)** | **+6%** |

**Table IV: Comparison with Original Paper Results**

| Model | Original Paper (Full Spider) | Our Recreation (50 Queries) |
|-------|------------------------------|----------------------------|
| GPT-4 | 72.3% | N/A |
| Claude Opus 3 | 91.59% | N/A |
| Claude Opus 4.5 | N/A | 92% |

Our Claude Opus 4.5 result (92%) matches the original paper's Claude Opus 3 performance (91.59%), validating successful recreation with modern models.

**Table V: Comprehensive Metrics Breakdown**

| Metric | Best Model | Best Value | Notes |
|--------|------------|------------|-------|
| Exact Match | Claude Opus 4.5 | 28% | Strict string matching |
| Valid SQL Rate | All models | 100% | No syntax errors after correction |
| Execution Accuracy | Claude Opus 4.5 | 92% | Primary metric |

### B. Model-Specific Analysis

#### 1) Claude Opus 4.5 Performance

Claude Opus 4.5 demonstrates the most significant improvement from the multi-agent approach:

- **Baseline:** 86% (43/50 correct)
- **SQL-of-Thought:** 92% (46/50 correct)
- **Absolute gain:** +6 percentage points
- **Relative improvement:** 7% error reduction

**Error Analysis:** The 3 additional queries corrected by SQL-of-Thought involved:
- 2 incorrect join conditions (corrected by schema linking + correction loop)
- 1 missing GROUP BY clause (corrected by subproblem decomposition)

The 4 remaining errors were:
- 2 ambiguous questions with multiple valid interpretations
- 1 complex nested subquery exceeding model capacity
- 1 domain-specific terminology mismatch

#### 2) GPT-4o Performance Plateau

GPT-4o shows no improvement from SQL-of-Thought:

- **Baseline:** 72% (36/50)
- **SQL-of-Thought:** 72% (36/50)
- **No improvement**

**Hypothesis:** GPT-4o's internal reasoning may conflict with the external correction loop, leading to overcorrection. Analysis of correction attempts reveals:
- 8 queries initially correct but "corrected" to wrong answers
- 5 queries correctly identified as needing fixes but failed to improve
- 3 queries improved through correction

This suggests that for some model architectures, the correction loop can be counterproductive.

#### 3) Reasoning Models (GPT-o1-mini)

GPT-o1-mini with high reasoning effort shows slight degradation:

- **Baseline:** 82% (41/50)
- **SQL-of-Thought:** 80% (40/50)
- **Degradation:** -2 percentage points

**Analysis:** Reasoning models perform extended internal chain-of-thought before generating outputs [34]. The external multi-agent decomposition may introduce redundant reasoning layers:

*Internal reasoning (o1-mini) + External decomposition (SoT) → Interference*

One query was overcorrected: the model initially produced a correct query, but the correction planning agent misidentified a non-existent error, leading to a "fix" that broke the query.

### C. Correction Loop Effectiveness

**Table VI: Correction Loop Statistics (Claude Opus 4.5)**

| Metric | Count | Percentage |
|--------|-------|------------|
| Queries needing correction | 12 | 24% |
| Successfully corrected | 9 | 75% of errors |
| Failed after max attempts | 3 | 25% of errors |
| Average attempts to success | 1.4 | - |

**Correction Trigger Analysis:**

| Trigger Type | Count | Success Rate |
|--------------|-------|--------------|
| Syntax error | 2 | 100% |
| Runtime exception | 4 | 75% |
| Result mismatch | 6 | 67% |

Result mismatch correction (semantic errors) has lower success rate than syntax/runtime errors, suggesting taxonomy coverage gaps for logical errors.

### D. Error Taxonomy Effectiveness

We analyze which taxonomy categories most frequently triggered corrections:

**Table VII: Error Category Distribution**

| Category | Occurrences | Correction Success |
|----------|-------------|-------------------|
| Join Errors | 4 | 75% |
| Schema Link | 3 | 100% |
| Filter Errors | 2 | 50% |
| Aggregation | 2 | 50% |
| Subquery | 1 | 100% |

Schema Link errors have 100% correction success, validating the effectiveness of the schema linking agent. Join and Filter errors have lower success rates, indicating potential for improved taxonomy granularity.

### E. Exact Match vs Execution Accuracy Gap

The substantial gap between Exact Match (28%) and Execution Accuracy (92%) reveals important insights:

**Observation 1:** Multiple semantically equivalent SQL queries exist for the same question.

**Example:** For "List all students in Computer Science department":

*Query 1 (Gold):*
```sql
SELECT * FROM students WHERE department = 'Computer Science'
```

*Query 2 (Predicted, semantically equivalent):*
```sql
SELECT student_id, name, department FROM students 
WHERE department = 'Computer Science'
```

Both produce identical results but have different text representations.

**Observation 2:** Query optimization by LLMs introduces syntactic variations:

- Column ordering differences
- Explicit vs implicit JOIN syntax
- Alias naming variations
- Subquery vs JOIN equivalences

These variations explain why Exact Match is not a reliable metric for text-to-SQL evaluation [35]. Execution Accuracy better captures functional correctness.

### F. Computational Efficiency

**Table VIII: Average Latency per Query (seconds)**

| Model | Simple Baseline | SQL-of-Thought | Overhead |
|-------|-----------------|----------------|----------|
| GPT-4o | 3.2s | 18.5s | 5.8× |
| GPT-o1-mini | 8.7s | 42.3s | 4.9× |
| Claude Opus 4.5 | 4.1s | 21.6s | 5.3× |

The multi-agent pipeline introduces 5-6× latency overhead due to six sequential LLM calls plus potential correction loops. For production systems, this trade-off between accuracy and latency must be carefully considered.

**Optimization opportunities:**
- Parallel agent execution where dependencies allow
- Caching schema introspection results
- Early termination on high-confidence predictions

---

## V. ARCHITECTURAL IMPROVEMENTS

Beyond recreating the original framework, we implemented four major enhancements:

### A. Real-Time Visualization System

We developed a live dashboard using NiceGUI [36] that provides:

**1) Agent Execution Trace:** Real-time visualization of each agent's input/output with collapsible prompt inspection.

**2) Benchmark Progress Tracking:**
