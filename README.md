# SQL of Thought Paper Recreation

## How to run

Pure llm: `python3 benchmark_spider.py --mode simple --limit 100 --splits spider --model gpt-5-mini-2025-08-07 --reasoning_effort high`

Using the paper's method: `python3 benchmark_spider.py --mode sot --limit 100 --splits spider --max_corrections 3 --model gpt-5-mini-2025-08-07 --reasoning_effort high`
