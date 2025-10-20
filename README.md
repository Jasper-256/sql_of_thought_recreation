# SQL-of-Thought Implementation

An implementation of the **SQL-of-Thought** multi-agent framework for Text-to-SQL generation, based on the paper ["SQL-of-Thought: Multi-agentic Text-to-SQL with Guided Error Correction"](https://arxiv.org/abs/2509.00581) by Chaturvedi et al.

## Overview

SQL-of-Thought decomposes the Text-to-SQL task into specialized agents working together:

1. **Schema Linking** - Identifies relevant tables and columns
2. **Subproblem Identification** - Breaks down the query into clauses
3. **Query Plan Generation** - Creates a step-by-step execution plan using Chain-of-Thought
4. **SQL Generation** - Produces the executable SQL query
5. **Guided Error Correction** - Fixes errors using a taxonomy of 31 SQL error types

The key innovation is the **taxonomy-guided correction loop** that goes beyond simple execution feedback to systematically diagnose and fix logical errors.

## Quick Start

```bash
# Install dependencies
pip install -U langgraph langchain langchain-openai openai datasets tqdm requests gdown python-dotenv

# Set your API key
export OPENAI_API_KEY=sk-...

# Run baseline (single LLM call)
python3 benchmark_spider.py --mode simple --limit 100 --model gpt-4.1-mini

# Run SQL-of-Thought with correction loop
python3 benchmark_spider.py --mode sot --limit 100 --max_corrections 3 --model gpt-4.1-mini
```

## Features

- **Two modes**: Simple baseline vs. full SQL-of-Thought pipeline
- **Automatic Spider dataset download** and schema introspection
- **Multiple benchmarks**: Spider, Spider-Realistic, Spider-SYN
- **Flexible model support**: Works with any OpenAI-compatible model
- **Comprehensive metrics**: Exact Match, Valid SQL rate, and Execution Accuracy

## Results

The paper reports state-of-the-art results on Spider benchmarks:
- **Spider**: 91.59% Execution Accuracy
- **Spider-Realistic**: 90.16% Execution Accuracy
- **Spider-SYN**: 82.01% Execution Accuracy

## Command-Line Options

```bash
python3 benchmark_spider.py \
  --mode sot \                      # "simple" or "sot"
  --model gpt-4.1-mini \            # Any OpenAI model
  --max_corrections 3 \             # Number of correction attempts
  --limit 50 \                      # Examples per split
  --splits spider spider-realistic  # Which benchmarks to run
```

## Architecture

The framework uses LangGraph to orchestrate multiple specialized agents. The correction loop leverages a comprehensive error taxonomy covering:

- Syntax errors
- Schema linking issues
- Join problems
- Filter conditions
- Aggregation logic
- Value formatting
- Subquery errors
- Set operations
- Other SQL patterns

## Citation

```bibtex
@article{chaturvedi2025sqlofthought,
  title={SQL-of-Thought: Multi-agentic Text-to-SQL with Guided Error Correction},
  author={Chaturvedi, Saumya and Chadha, Aman and Bindschaedler, Laurent},
  journal={arXiv preprint arXiv:2509.00581},
  year={2025}
}
```

## License

This implementation is provided for research and educational purposes.
