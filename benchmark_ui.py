#!/usr/bin/env python3
"""
benchmark_ui.py

A real-time UI for visualizing the SQL-of-Thought agent execution pipeline.
Uses NiceGUI for a modern, live-updating interface that shows all LLM 
inputs and outputs at each step.

Usage:
    python benchmark_ui.py [--port 8080] [--limit 10]
"""

import asyncio
import json
import os
import queue
import sys
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from nicegui import ui, app

# Local imports
from sql_of_thought_with_callbacks import SQLOfThoughtGraphWithCallbacks, StepCallback
from sql_model_graph import summarize_sqlite_schema, strip_sql_fences, SQLGeneratorGraph
from benchmark_spider import (
    load_all_examples, DB_DIR, DATA_DIR, ensure_dir, 
    have_local_spider_assets, download_spider_zip_to,
    normalize_sql, rows_to_canonical, try_exec_sqlite, Example
)

load_dotenv()

# ----------------------------- Data Classes -----------------------------
@dataclass
class StepData:
    """Represents a single step in the pipeline execution."""
    name: str
    system_prompt: str
    user_prompt: str
    output: str
    extra: Optional[Dict[str, Any]] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class BenchmarkItem:
    """Represents a single benchmark example with its execution trace."""
    idx: int
    db_id: str
    question: str
    gold_sql: str
    pred_sql: str = ""
    steps: List[StepData] = field(default_factory=list)
    status: str = "pending"  # pending, running, success, failed
    exact_match: bool = False
    valid_sql: bool = False
    exec_accuracy: bool = False
    error_message: str = ""


# ----------------------------- Global State -----------------------------
class AppState:
    """Global application state."""
    def __init__(self):
        self.items: List[BenchmarkItem] = []
        self.current_item_idx: int = -1
        self.is_running: bool = False
        self.model_name: str = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        self.mode: str = "sot"
        self.limit: int = 10
        self.max_corrections: int = 2
        self.split: str = "spider"
        self.total_examples: int = 0
        self.completed_examples: int = 0
        self.em_hits: int = 0
        self.valid_hits: int = 0
        self.exec_hits: int = 0
        self.needs_refresh: bool = False  # Flag for UI refresh


state = AppState()


# ----------------------------- Step Display Names -----------------------------
STEP_DISPLAY_NAMES = {
    "schema_link": "🔗 Schema Linking",
    "subproblem": "🧩 Subproblem Decomposition",
    "plan": "📋 Query Planning",
    "sql": "💾 SQL Generation",
    "execute": "▶️ Execution",
    "correction_plan": "🔧 Correction Planning",
    "correction_sql": "🔄 Correction SQL",
}

STEP_COLORS = {
    "schema_link": "#8B5CF6",  # violet
    "subproblem": "#EC4899",   # pink
    "plan": "#F59E0B",         # amber
    "sql": "#10B981",          # emerald
    "execute": "#3B82F6",      # blue
    "correction_plan": "#EF4444",  # red
    "correction_sql": "#F97316",   # orange
}


# ----------------------------- UI Components -----------------------------
def create_header():
    """Create the application header."""
    with ui.header().classes('items-center justify-between bg-gradient-to-r from-slate-900 via-purple-900 to-slate-900'):
        with ui.row().classes('items-center gap-4'):
            ui.icon('psychology').classes('text-3xl text-purple-400')
            ui.label('SQL-of-Thought Pipeline Visualizer').classes(
                'text-2xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent'
            )
        with ui.row().classes('items-center gap-2'):
            ui.label().bind_text_from(state, 'model_name', lambda x: f'Model: {x}').classes('text-gray-400 text-sm')


def create_control_panel(on_run, on_stop):
    """Create the control panel for benchmark settings."""
    with ui.card().classes('w-full bg-slate-800/50 backdrop-blur border border-slate-700'):
        ui.label('⚙️ Configuration').classes('text-lg font-semibold text-purple-300 mb-2')
        
        with ui.row().classes('w-full gap-4 flex-wrap'):
            with ui.column().classes('gap-1'):
                ui.label('Model').classes('text-xs text-gray-400')
                model_select = ui.select(
                    options=['gpt-4.1-mini', 'gpt-4o-2024-08-06', 'gpt-4o-mini', 'o3-mini'],
                    value=state.model_name
                ).classes('w-48').props('dark dense outlined')
                model_select.bind_value(state, 'model_name')
            
            with ui.column().classes('gap-1'):
                ui.label('Mode').classes('text-xs text-gray-400')
                mode_select = ui.select(
                    options={'sot': 'SQL-of-Thought', 'simple': 'Simple Baseline'},
                    value=state.mode
                ).classes('w-48').props('dark dense outlined')
                mode_select.bind_value(state, 'mode')
            
            with ui.column().classes('gap-1'):
                ui.label('Dataset').classes('text-xs text-gray-400')
                split_select = ui.select(
                    options=['spider', 'spider-realistic', 'spider-syn'],
                    value=state.split
                ).classes('w-40').props('dark dense outlined')
                split_select.bind_value(state, 'split')
            
            with ui.column().classes('gap-1'):
                ui.label('Limit').classes('text-xs text-gray-400')
                limit_input = ui.number(value=state.limit, min=1, max=100).classes('w-24').props('dark dense outlined')
                limit_input.bind_value(state, 'limit')
            
            with ui.column().classes('gap-1'):
                ui.label('Max Corrections').classes('text-xs text-gray-400')
                corrections_input = ui.number(value=state.max_corrections, min=1, max=5).classes('w-24').props('dark dense outlined')
                corrections_input.bind_value(state, 'max_corrections')
        
        with ui.row().classes('w-full gap-4 mt-4'):
            run_btn = ui.button('▶️ Run Benchmark', on_click=on_run).classes(
                'bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700'
            ).props('no-caps')
            run_btn.bind_enabled_from(state, 'is_running', lambda x: not x)
            
            stop_btn = ui.button('⏹️ Stop', on_click=on_stop).classes(
                'bg-red-600 hover:bg-red-700'
            ).props('no-caps')
            stop_btn.bind_enabled_from(state, 'is_running')


def create_progress_panel():
    """Create the progress panel."""
    with ui.card().classes('w-full bg-slate-800/50 backdrop-blur border border-slate-700'):
        ui.label('📊 Progress').classes('text-lg font-semibold text-purple-300 mb-2')
        
        with ui.row().classes('w-full gap-8'):
            with ui.column().classes('items-center'):
                ui.label().bind_text_from(
                    state, 'completed_examples', 
                    lambda _: f'{state.completed_examples}/{state.total_examples}'
                ).classes('text-3xl font-bold text-white')
                ui.label('Completed').classes('text-xs text-gray-400')
            
            with ui.column().classes('items-center'):
                ui.label().bind_text_from(
                    state, 'em_hits',
                    lambda _: f'{(state.em_hits/state.completed_examples*100):.1f}%' if state.completed_examples > 0 else '0%'
                ).classes('text-3xl font-bold text-emerald-400')
                ui.label('Exact Match').classes('text-xs text-gray-400')
            
            with ui.column().classes('items-center'):
                ui.label().bind_text_from(
                    state, 'valid_hits',
                    lambda _: f'{(state.valid_hits/state.completed_examples*100):.1f}%' if state.completed_examples > 0 else '0%'
                ).classes('text-3xl font-bold text-blue-400')
                ui.label('Valid SQL').classes('text-xs text-gray-400')
            
            with ui.column().classes('items-center'):
                ui.label().bind_text_from(
                    state, 'exec_hits',
                    lambda _: f'{(state.exec_hits/state.completed_examples*100):.1f}%' if state.completed_examples > 0 else '0%'
                ).classes('text-3xl font-bold text-purple-400')
                ui.label('Exec Accuracy').classes('text-xs text-gray-400')
        
        # Progress bar - percentage shown as text, no label on bar
        with ui.row().classes('w-full items-center gap-4 mt-4'):
            ui.linear_progress(size='12px', show_value=False).classes('flex-1').props('rounded color=purple-6').bind_value_from(
                state, 'completed_examples',
                lambda _: state.completed_examples / state.total_examples if state.total_examples > 0 else 0
            )
            ui.label().bind_text_from(
                state, 'completed_examples',
                lambda _: f'{int(state.completed_examples / state.total_examples * 100)}%' if state.total_examples > 0 else '0%'
            ).classes('text-lg font-bold text-purple-400 min-w-[50px] text-right')


def create_item_card(item: BenchmarkItem, on_click) -> ui.card:
    """Create a card for a benchmark item."""
    status_colors = {
        'pending': 'bg-slate-700',
        'running': 'bg-purple-900/50 border-purple-500',
        'success': 'bg-emerald-900/30 border-emerald-500',
        'failed': 'bg-red-900/30 border-red-500',
    }
    
    status_icons = {
        'pending': '⏳',
        'running': '🔄',
        'success': '✅',
        'failed': '❌',
    }
    
    with ui.card().classes(f'w-full cursor-pointer transition-all duration-300 {status_colors.get(item.status, "bg-slate-700")} border') as card:
        with ui.row().classes('w-full items-center justify-between'):
            with ui.row().classes('items-center gap-2'):
                ui.label(f'{status_icons.get(item.status, "⏳")}').classes('text-lg')
                ui.label(f'#{item.idx}').classes('font-mono text-purple-400')
                ui.label(f'[{item.db_id}]').classes('text-xs text-gray-500 font-mono')
            
            with ui.row().classes('gap-2'):
                if item.status in ['success', 'failed']:
                    if item.exact_match:
                        ui.badge('EM', color='green').props('outline')
                    if item.valid_sql:
                        ui.badge('Valid', color='blue').props('outline')
                    if item.exec_accuracy:
                        ui.badge('ExAcc', color='purple').props('outline')
        
        ui.label(item.question).classes('text-sm text-gray-300 mt-2 line-clamp-2')
        
        if item.pred_sql:
            with ui.expansion('SQL Output', icon='code').classes('w-full mt-2'):
                ui.code(item.pred_sql, language='sql').classes('text-xs')
    
    card.on('click', lambda: on_click(item.idx))
    return card


def create_step_card(step: StepData):
    """Create a card for a pipeline step."""
    color = STEP_COLORS.get(step.name, '#6B7280')
    display_name = STEP_DISPLAY_NAMES.get(step.name, step.name)
    
    with ui.card().classes('w-full bg-slate-800/70 backdrop-blur border border-slate-600 mb-3 overflow-hidden'):
        with ui.row().classes('w-full items-center gap-2 mb-2'):
            ui.element('div').classes('w-3 h-3 rounded-full').style(f'background-color: {color}')
            ui.label(display_name).classes('font-semibold text-white')
            ui.label(f'{datetime.fromtimestamp(step.timestamp).strftime("%H:%M:%S")}').classes('text-xs text-gray-500 ml-auto')
        
        # System prompt (collapsible) - full content, no truncation
        if step.system_prompt:
            with ui.expansion('System Prompt', icon='psychology').classes('w-full overflow-hidden'):
                ui.code(step.system_prompt).classes('text-xs w-full')
        
        # User prompt (collapsible) - full content, no truncation
        if step.user_prompt:
            with ui.expansion('User Prompt (Input)', icon='input').classes('w-full overflow-hidden'):
                ui.code(step.user_prompt).classes('text-xs w-full')
        
        # Output (always visible)
        ui.label('Output').classes('text-xs text-gray-400 mt-2')
        if step.name == 'sql' or step.name == 'correction_sql':
            ui.code(step.output, language='sql').classes('text-sm w-full')
        elif step.name == 'subproblem':
            try:
                parsed = json.loads(step.output) if step.output.strip().startswith('{') else {}
                if parsed:
                    ui.code(json.dumps(parsed, indent=2), language='json').classes('text-sm w-full')
                else:
                    ui.code(step.output).classes('text-sm w-full')
            except:
                ui.code(step.output).classes('text-sm w-full')
        else:
            ui.code(step.output).classes('text-sm w-full')
        
        # Extra info
        if step.extra:
            with ui.expansion('Extra Info', icon='info').classes('w-full mt-2 overflow-hidden'):
                ui.code(json.dumps(step.extra, indent=2, default=str), language='json').classes('text-xs w-full')


# ----------------------------- Benchmark Execution -----------------------------
stop_requested = False


# ----------------------------- Main UI -----------------------------
@ui.page('/')
def main_page():
    """Main application page."""
    global stop_requested
    
    # Custom styles
    ui.add_head_html('''
    <style>
        body {
            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
            min-height: 100vh;
        }
        .line-clamp-2 {
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }
        ::-webkit-scrollbar {
            width: 8px;
        }
        ::-webkit-scrollbar-track {
            background: rgba(30, 27, 75, 0.5);
        }
        ::-webkit-scrollbar-thumb {
            background: #6366f1;
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #818cf8;
        }
        /* Fix horizontal scroll in code blocks */
        .nicegui-code, .nicegui-code pre, .nicegui-code code {
            white-space: pre-wrap !important;
            word-wrap: break-word !important;
            word-break: break-word !important;
            overflow-x: hidden !important;
            max-width: 100% !important;
        }
        .overflow-x-hidden {
            overflow-x: hidden !important;
        }
        /* Hide horizontal scrollbar on scroll areas */
        .q-scrollarea__content {
            overflow-x: hidden !important;
        }
    </style>
    ''')
    
    # Create refreshable components
    @ui.refreshable
    def items_list():
        """Refreshable items list."""
        for item in state.items:
            create_item_card(item, on_click=select_item)
        
        if not state.items:
            ui.label('No items yet. Click "Run Benchmark" to start.').classes('text-gray-500 italic')
    
    @ui.refreshable
    def details_panel():
        """Refreshable details panel."""
        if state.current_item_idx < 0 or state.current_item_idx >= len(state.items):
            ui.label('Select an item to view details...').classes('text-gray-500 italic')
            return
        
        item = state.items[state.current_item_idx]
        
        # Item header
        with ui.card().classes('w-full bg-slate-700/50 mb-4'):
            with ui.row().classes('items-center gap-2'):
                ui.label(f'#{item.idx}').classes('text-2xl font-bold text-purple-400')
                ui.label(f'[{item.db_id}]').classes('text-sm text-gray-500 font-mono')
            ui.label(item.question).classes('text-white mt-2')
            
            with ui.row().classes('gap-4 mt-2 flex-wrap'):
                with ui.column():
                    ui.label('Gold SQL').classes('text-xs text-gray-400')
                    ui.code(item.gold_sql, language='sql').classes('text-xs')
                
                if item.pred_sql:
                    with ui.column():
                        ui.label('Predicted SQL').classes('text-xs text-gray-400')
                        ui.code(item.pred_sql, language='sql').classes('text-xs')
        
        # Steps
        if item.steps:
            for step in item.steps:
                create_step_card(step)
        else:
            ui.label('No steps recorded yet...').classes('text-gray-500 italic')
    
    def select_item(idx: int):
        """Handle item selection."""
        state.current_item_idx = idx
        details_panel.refresh()
    
    def refresh_all():
        """Refresh both panels."""
        items_list.refresh()
        details_panel.refresh()
    
    async def stop_benchmark():
        """Stop the currently running benchmark."""
        global stop_requested
        stop_requested = True
        state.is_running = False
    
    async def run_benchmark():
        """Run the benchmark with live UI updates."""
        global stop_requested
        stop_requested = False
        
        # Check API key
        if not os.getenv("OPENAI_API_KEY"):
            ui.notify("OPENAI_API_KEY environment variable is not set!", type='negative')
            return
        
        # Ensure data directory
        ensure_dir(DATA_DIR)
        if not have_local_spider_assets():
            ui.notify("Downloading Spider dataset...", type='info')
            download_spider_zip_to(DATA_DIR)
        
        if not DB_DIR.exists():
            ui.notify("Database directory not found. Some features may be limited.", type='warning')
        
        # Reset state
        state.is_running = True
        state.items = []
        state.current_item_idx = -1
        state.completed_examples = 0
        state.em_hits = 0
        state.valid_hits = 0
        state.exec_hits = 0
        
        refresh_all()
        
        # Load examples
        try:
            work = load_all_examples(splits=[state.split], limit=int(state.limit))
            if not work:
                ui.notify("No examples loaded!", type='warning')
                state.is_running = False
                return
            
            examples = work[0][1]
            state.total_examples = len(examples)
        except Exception as e:
            ui.notify(f"Failed to load examples: {e}", type='negative')
            state.is_running = False
            return
        
        # Create benchmark items
        for i, ex in enumerate(examples):
            state.items.append(BenchmarkItem(
                idx=i,
                db_id=ex.db_id,
                question=ex.question,
                gold_sql=ex.gold_sql,
            ))
        
        refresh_all()
        
        ui.notify(f"Starting benchmark with {len(examples)} examples...", type='positive')
        
        # Run each example
        for i, item in enumerate(state.items):
            if stop_requested:
                ui.notify("Benchmark stopped by user", type='warning')
                break
            
            item.status = 'running'
            
            # Only auto-advance to the new item if user is viewing the previous item (or no item yet)
            # This allows users to stay on an item they clicked on without being jumped away
            previous_idx = i - 1
            if state.current_item_idx == previous_idx or state.current_item_idx == -1:
                state.current_item_idx = i
            
            refresh_all()
            await asyncio.sleep(0.1)  # Allow UI to update
            
            db_path = DB_DIR / item.db_id / f"{item.db_id}.sqlite"
            
            # Create callback to capture steps (thread-safe via flag)
            def make_step_callback(current_item):
                def step_callback(step_name: str, sys_prompt: str, user_prompt: str, output: str, extra: Optional[Dict] = None):
                    step = StepData(
                        name=step_name,
                        system_prompt=sys_prompt,
                        user_prompt=user_prompt,
                        output=output,
                        extra=extra,
                    )
                    current_item.steps.append(step)
                    state.needs_refresh = True  # Signal refresh needed
                return step_callback
            
            step_callback = make_step_callback(item)
            
            try:
                if state.mode == 'sot':
                    # SQL-of-Thought mode with callbacks
                    model = SQLOfThoughtGraphWithCallbacks(
                        model_name=state.model_name,
                        temperature=0.0,
                        max_attempts=int(state.max_corrections),
                        step_callback=step_callback,
                    )
                    pred_sql = await asyncio.to_thread(
                        model.generate_sql_for_db,
                        item.question,
                        str(db_path),
                        item.db_id,
                        gold_sql=item.gold_sql,
                    )
                else:
                    # Simple baseline mode
                    model = SQLGeneratorGraph(
                        model_name=state.model_name,
                        temperature=0.0,
                    )
                    
                    # Capture the single step
                    schema_text = summarize_sqlite_schema(str(db_path)) if db_path.exists() else "DB not found"
                    step = StepData(
                        name="sql",
                        system_prompt="Simple baseline - single LLM call",
                        user_prompt=f"Question: {item.question}\nSchema: {schema_text[:500]}...",
                        output="(See SQL output below)",
                    )
                    item.steps.append(step)
                    
                    pred_sql = await asyncio.to_thread(
                        model.generate_sql_for_db,
                        item.question,
                        str(db_path),
                        item.db_id,
                    )
                
                item.pred_sql = pred_sql
                
                # Calculate metrics
                gold_norm = normalize_sql(item.gold_sql)
                pred_norm = normalize_sql(pred_sql)
                item.exact_match = gold_norm == pred_norm
                
                if db_path.exists():
                    ok_pred, pred_rows = try_exec_sqlite(db_path, pred_sql)
                    item.valid_sql = ok_pred
                    if ok_pred:
                        ok_gold, gold_rows = try_exec_sqlite(db_path, item.gold_sql)
                        if ok_gold:
                            item.exec_accuracy = rows_to_canonical(pred_rows) == rows_to_canonical(gold_rows)
                
                item.status = 'success' if item.exec_accuracy else 'failed'
                
                # Update stats
                if item.exact_match:
                    state.em_hits += 1
                if item.valid_sql:
                    state.valid_hits += 1
                if item.exec_accuracy:
                    state.exec_hits += 1
                
            except Exception as e:
                item.status = 'failed'
                item.error_message = str(e)
                item.steps.append(StepData(
                    name="error",
                    system_prompt="",
                    user_prompt="",
                    output=f"Error: {str(e)}",
                ))
            
            state.completed_examples += 1
            refresh_all()
            await asyncio.sleep(0.1)  # Allow UI to update
        
        state.is_running = False
        ui.notify("Benchmark completed!", type='positive')
    
    # Timer to check for refresh requests from background threads
    def check_refresh():
        if state.needs_refresh:
            state.needs_refresh = False
            details_panel.refresh()
    
    ui.timer(0.3, check_refresh)
    
    create_header()
    
    with ui.column().classes('w-full max-w-7xl mx-auto p-4 gap-4'):
        create_control_panel(on_run=run_benchmark, on_stop=stop_benchmark)
        create_progress_panel()
        
        with ui.row().classes('w-full gap-4 flex-1 min-h-0'):
            # Items list panel
            with ui.card().classes('w-96 bg-slate-800/50 backdrop-blur border border-slate-700 h-[calc(100vh-320px)] overflow-hidden'):
                ui.label('📝 Benchmark Items').classes('text-lg font-semibold text-purple-300 mb-2')
                with ui.scroll_area().classes('h-[calc(100vh-370px)]'):
                    with ui.column().classes('w-full gap-2 pr-2'):
                        items_list()
            
            # Details panel - much taller, takes remaining space
            with ui.card().classes('flex-1 bg-slate-800/50 backdrop-blur border border-slate-700 h-[calc(100vh-320px)] overflow-hidden'):
                ui.label('🔬 Pipeline Execution Details').classes('text-lg font-semibold text-purple-300 mb-2')
                with ui.scroll_area().classes('h-[calc(100vh-370px)]').props('style="overflow-x: hidden"'):
                    with ui.column().classes('w-full gap-2 pr-4 overflow-x-hidden'):
                        details_panel()


# ----------------------------- Entry Point -----------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="SQL-of-Thought Pipeline Visualizer")
    parser.add_argument("--port", type=int, default=8080, help="Port to run the UI server")
    parser.add_argument("--limit", type=int, default=10, help="Default limit for benchmark examples")
    args = parser.parse_args()
    
    state.limit = args.limit
    
    print(f"""
╔═══════════════════════════════════════════════════════════════════╗
║       SQL-of-Thought Pipeline Visualizer                          ║
║                                                                   ║
║  Open http://localhost:{args.port} in your browser                    ║
║                                                                   ║
║  Features:                                                        ║
║  • Live visualization of agent pipeline execution                 ║
║  • View all LLM inputs and outputs at each step                  ║
║  • Track benchmark progress and metrics                          ║
╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    ui.run(
        port=args.port,
        title="SQL-of-Thought Visualizer",
        favicon="🧠",
        dark=True,
        reload=False,
    )


if __name__ in {"__main__", "__mp_main__"}:
    main()
