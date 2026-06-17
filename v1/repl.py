"""REPL environment for RLM deep-dive searches.

Adapted from rlm-minimal (https://github.com/alexzhang13/rlm).
Uses Hermes's call_llm(task="rlm") for sub-queries instead of
a direct OpenAI client, so it automatically picks up the auxiliary model
configured in auxiliary.rlm (or falls back to auxiliary.compression).
"""

import io
import json
import os
import re
import sys
import tempfile
import threading
import time
from datetime import datetime


_RLM_LOG = "/tmp/rlm_repl.log"
def _log(msg: str):
    """Temporary debug logger — writes to /tmp/rlm_repl.log."""
    try:
        with open(_RLM_LOG, "a") as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
    except Exception:
        pass


import ast
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# REPL Environment
# ---------------------------------------------------------------------------

@dataclass
class REPLResult:
    """Result from executing code in the REPL sandbox."""
    stdout: str
    stderr: str
    locals: dict
    execution_time: float = 0.0

    def __str__(self):
        return f"REPLResult(stdout={self.stdout[:200]}, stderr={self.stderr[:200]})"


class REPLEnv:
    """Sandboxed Python REPL with llm_query() and context access.

    Adapted from rlm-minimal's REPLEnv. Key changes:
    - llm_query() uses Hermes's call_llm(task="rlm") instead of
      a direct OpenAI client, so it picks up auxiliary.compression.model.
    - Logging stripped (no repl_env_logger).
    - Context loaded from a string (FTS5 search results).
    """

    def __init__(
        self,
        context_str: Optional[str] = None,
        max_llm_tokens: int = 1024,
        hermes_home: Optional[str] = None,
        session_ids: Optional[List[str]] = None,
    ):
        self.original_cwd = os.getcwd()
        self.temp_dir = tempfile.mkdtemp(prefix="rlm_repl_")
        self.max_llm_tokens = max_llm_tokens
        self._hermes_home = hermes_home
        self._session_ids = session_ids

        # Sandboxed globals — allow useful builtins, block dangerous ones
        self.globals = {
            '__builtins__': {
                'print': print, 'len': len, 'str': str, 'int': int, 'float': float,
                'list': list, 'dict': dict, 'set': set, 'tuple': tuple, 'bool': bool,
                'type': type, 'isinstance': isinstance, 'enumerate': enumerate,
                'zip': zip, 'map': map, 'filter': filter, 'sorted': sorted,
                'min': min, 'max': max, 'sum': sum, 'abs': abs, 'round': round,
                'range': range, 'reversed': reversed, 'slice': slice,
                'iter': iter, 'next': next, 'any': any, 'all': all,
                'hasattr': hasattr, 'getattr': getattr, 'dir': dir, 'vars': vars,
                'repr': repr, 'format': format,
                '__import__': __import__,
                'open': open,
                'Exception': Exception, 'ValueError': ValueError, 'TypeError': TypeError,
                'KeyError': KeyError, 'IndexError': IndexError,
                'AttributeError': AttributeError, 'RuntimeError': RuntimeError,
                'FileNotFoundError': FileNotFoundError, 'OSError': OSError,
                'NameError': NameError, 'ImportError': ImportError,
                'input': None, 'eval': None, 'exec': None,  # blocked
                'compile': None, 'globals': None, 'locals': None,
            }
        }
        self.locals: Dict[str, Any] = {}
        self._lock = threading.Lock()

        # Load context into the REPL
        self._load_context(context_str)

        # Expose llm_query — uses Hermes's auxiliary.rlm model config
        def llm_query(prompt: str) -> str:
            """Query the RLM model via auxiliary.rlm config."""
            from agent.auxiliary_client import call_llm
            try:
                response = call_llm(
                    task="rlm",
                    main_runtime={},
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_llm_tokens,
                )
                content = response.choices[0].message.content
                return content.strip() if isinstance(content, str) else str(content or "")
            except Exception as e:
                return f"Error calling LLM: {e}"

        self.globals['llm_query'] = llm_query

        # Expose search_context — FTS5 full-text search over the session DB
        # Scope is fixed at construction time based on the tool call's scope param.
        # The REPL model cannot override it.
        _repl_session_ids = set(self._session_ids) if self._session_ids else set()
        _repl_scoped = bool(_repl_session_ids)  # True = lineage-only, False = all

        def search_context(query_str: str, limit: int = 10) -> str:
            """Search archived messages using FTS5. Returns matching message snippets.

            Scope is fixed by the caller — this function always searches within
            the pre-configured session set.
            """
            try:
                from hermes_state import SessionDB, DEFAULT_DB_PATH
                from pathlib import Path
                db_path = Path(self._hermes_home) / "state.db" if self._hermes_home else DEFAULT_DB_PATH
                _log(f"search_context: query={query_str!r}, limit={limit}, db_path={db_path}, exists={db_path.exists()}, scoped={_repl_scoped}")
                if not db_path.exists():
                    return "Error: session database not found"
                db = SessionDB(db_path)
                # Fetch more than needed so post-filtering by lineage still yields enough
                fetch_limit = limit * 5 if _repl_scoped else limit
                results = db.search_messages(query_str, limit=fetch_limit)
                _log(f"search_context: raw results={len(results)}")
                if not results:
                    return f"No results for: {query_str}"
                # Post-filter by session lineage if scoped
                if _repl_scoped:
                    results = [r for r in results if r.get("session_id") in _repl_session_ids]
                    results = results[:limit]
                    _log(f"search_context: after lineage filter={len(results)}")
                    if not results:
                        return f"No results for: {query_str}"
                parts = []
                for r in results:
                    role = r.get("role", "?")
                    snippet = r.get("snippet", r.get("content", "")[:300])
                    sid = r.get("session_id", "?")
                    parts.append(f"[session:{sid} role:{role}] {snippet}")
                return "\n\n".join(parts)
            except Exception as e:
                _log(f"search_context ERROR: {e}")
                return f"Search error: {e}"

        self.globals['search_context'] = search_context

        # Expose FINAL_VAR — signal completion with variable value
        def final_var(variable_name: str) -> str:
            variable_name = variable_name.strip().strip('"').strip("'").strip('\n').strip('\r')
            if variable_name in self.locals:
                return str(self.locals[variable_name])
            return f"Error: Variable '{variable_name}' not found"

        self.globals['FINAL_VAR'] = final_var

    def _load_context(self, context_str: Optional[str]):
        """Write context to a temp file and load it into the REPL namespace."""
        if context_str is None:
            return
        context_path = os.path.join(self.temp_dir, "context.txt")
        with open(context_path, "w") as f:
            f.write(context_str)
        setup_code = (
            f"import os\n"
            f"with open(r'{context_path}', 'r') as f:\n"
            f"    context = f.read()\n"
        )
        self.code_execution(setup_code)

    def __del__(self):
        try:
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception:
            pass

    @contextmanager
    def _capture_output(self):
        """Thread-safe stdout/stderr capture."""
        with self._lock:
            old_stdout, old_stderr = sys.stdout, sys.stderr
            stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
            try:
                sys.stdout, sys.stderr = stdout_buf, stderr_buf
                yield stdout_buf, stderr_buf
            finally:
                sys.stdout, sys.stderr = old_stdout, old_stderr

    @contextmanager
    def _temp_working_directory(self):
        old_cwd = os.getcwd()
        try:
            os.chdir(self.temp_dir)
            yield
        finally:
            os.chdir(old_cwd)

    def code_execution(self, code: str) -> REPLResult:
        """Execute Python code in the sandboxed REPL."""
        start = time.time()
        with self._capture_output() as (stdout_buf, stderr_buf):
            with self._temp_working_directory():
                try:
                    lines = code.split('\n')
                    import_lines = [l for l in lines if l.startswith(('import ', 'from ')) and not l.startswith('#')]
                    other_lines = [l for l in lines if l not in import_lines]

                    if import_lines:
                        exec('\n'.join(import_lines), self.globals, self.globals)

                    if other_lines:
                        other_code = '\n'.join(other_lines)
                        ns = {**self.globals, **self.locals}

                        # Use AST to check if the last statement is a bare
                        # expression (function call, literal, comprehension, etc.)
                        # vs. an assignment, import, control flow, etc.
                        try:
                            tree = ast.parse(other_code)
                            last_node = tree.body[-1] if tree.body else None
                            is_expr = isinstance(last_node, ast.Expr)
                        except SyntaxError:
                            is_expr = False

                        if is_expr and len(tree.body) > 1:
                            # Execute everything except the last expression,
                            # then eval the last expression and print its value.
                            all_lines = other_code.split('\n')
                            # Find the line range of the last statement
                            last_lineno = last_node.lineno
                            # Lines are 1-indexed in AST
                            prefix_lines = all_lines[:last_lineno - 1]
                            suffix_lines = all_lines[last_lineno - 1:]
                            if prefix_lines:
                                exec('\n'.join(prefix_lines), ns, ns)
                            expr_code = '\n'.join(suffix_lines)
                            result = eval(compile(expr_code, '<repl>', 'eval'), ns, ns)
                            if result is not None:
                                print(repr(result))
                        elif is_expr:
                            result = eval(compile(other_code, '<repl>', 'eval'), ns, ns)
                            if result is not None:
                                print(repr(result))
                        else:
                            exec(other_code, ns, ns)

                        for k, v in ns.items():
                            if k not in self.globals:
                                self.locals[k] = v

                    stdout_content = stdout_buf.getvalue()
                    stderr_content = stderr_buf.getvalue()
                except Exception as e:
                    stderr_content = stderr_buf.getvalue() + str(e)
                    stdout_content = stdout_buf.getvalue()

        self.locals['_stdout'] = stdout_content
        self.locals['_stderr'] = stderr_content
        return REPLResult(stdout_content, stderr_content, self.locals.copy(), time.time() - start)


# ---------------------------------------------------------------------------
# Parsing utilities (from rlm-minimal utils.py)
# ---------------------------------------------------------------------------

def find_code_blocks(text: str) -> List[str]:
    """Find ```repl ... ``` code blocks in model output."""
    pattern = r'```repl\s*\n(.*?)\n```'
    return [m.group(1).strip() for m in re.finditer(pattern, text, re.DOTALL)]


def find_final_answer(text: str) -> Optional[Tuple[str, str]]:
    """Find FINAL(...) or FINAL_VAR(...) in model output."""
    m = re.search(r'^\s*FINAL_VAR\((.*?)\)', text, re.MULTILINE)
    if m:
        return ('FINAL_VAR', m.group(1).strip())
    m = re.search(r'^\s*FINAL\((.+)\)', text, re.MULTILINE)
    if m:
        return ('FINAL', m.group(1).strip())
    return None


# ---------------------------------------------------------------------------
# Prompts (adapted from rlm-minimal prompts.py)
# ---------------------------------------------------------------------------

REPL_SYSTEM_PROMPT = """You are tasked with answering a query using archived conversation messages. You have access to a REPL environment where you can write Python code to process the data and query a sub-LLM.

Your context is a string with {context_total_length} total characters, containing {context_message_count} messages across {context_session_count} sessions.

Context format — each message looks like:
[session:SESSION_ID role:ROLE] message content...

Sample of the first few messages:
{context_preview}

The REPL environment has:
1. A `context` variable that contains the archived messages. You should check its content to understand what you are working with. Make sure you look through it sufficiently as you answer your query.
2. A `llm_query(prompt)` function that calls an LLM inside your REPL environment. Use it to analyze chunks of context — it can handle large inputs.
3. A `search_context(query, limit=10)` function that performs FTS5 full-text search over the archived messages. Returns matching message snippets. Use this FIRST to find relevant messages before reading the full context.
4. `print()` to see output. IMPORTANT: print() output is truncated to metadata (prefix + length) in the conversation history, but the actual values stored in variables are NEVER truncated. If you print a variable and see "... (N chars total)", the full value is still in the variable — do NOT re-query llm_query to "get the full text". Use print() only to check types and previews; use llm_query() on variables to analyze their content.

STRATEGY: Start with search_context() to find relevant messages quickly. If that gives you enough, answer directly. If you need more detail, use llm_query() on specific chunks. The context variable has everything but search_context is faster for targeted queries.

IMPORTANT — BATCH YOUR OPERATIONS: Each code block is one iteration. Write multi-step code in a single block: search, analyze, branch, and print in one go. Do NOT write one-liner blocks — that wastes iterations.

When you want to execute Python code, wrap it in triple backticks with 'repl':
```repl
# GOOD — batch search, analyze, and branch in one block
results = search_context("topic keywords", limit=15)
if results:
    analysis = llm_query(f"What are the key points about X in: {{results}}")
    print(analysis)
else:
    # Fallback: search the raw context
    chunks = context.split('\\n\\n')
    for chunk in chunks[:10]:
        if 'keyword' in chunk.lower():
            print(chunk[:500])
            break
```

CRITICAL RULES:
- Execute code immediately, don't just describe what you'd do.
- Use llm_query liberally — it's cheap and fast.
- You MUST end with FINAL(your complete answer on a single line). This is the ONLY way to return your answer.
- FINAL() must be on its own line. Do NOT put code blocks after FINAL().
- Answer the original query directly and completely in your FINAL().
- If you have enough information, output FINAL() immediately — do not run unnecessary code."""


def build_system_prompt(
    context_total_length: int = 0,
    context_message_count: int = 0,
    context_session_count: int = 0,
    context_preview: str = "",
) -> List[Dict[str, str]]:
    content = REPL_SYSTEM_PROMPT.format(
        context_total_length=context_total_length,
        context_message_count=context_message_count,
        context_session_count=context_session_count,
        context_preview=context_preview,
    )
    return [{"role": "system", "content": content}]


def next_action_prompt(query: str, iteration: int = 0) -> Dict[str, str]:
    base = f'Think step-by-step. Use the REPL environment (which has `context`) to answer: "{query}".\n\n'
    if iteration == 0:
        return {"role": "user", "content": "You have not interacted with the REPL yet. Read the context first.\n\n" + base + "Your next action:"}
    return {"role": "user", "content": "Your previous REPL interactions are above. " + base + "Your next action:"}


def final_answer_prompt(query: str) -> Dict[str, str]:
    return {"role": "user", "content": (
        "STOP. Do NOT write any more code. Based on everything you have learned "
        "from the REPL, provide your FINAL answer now.\n\n"
        "You MUST output: FINAL(your complete answer here)\n\n"
        f"Question: {query}"
    )}


# ---------------------------------------------------------------------------
# RLM REPL loop
# ---------------------------------------------------------------------------

def run_rlm_repl(
    context: str,
    query: str,
    max_iterations: int = 12,
    max_llm_tokens: int = 2048,
    hermes_home: Optional[str] = None,
    session_ids: Optional[List[str]] = None,
) -> str:
    """Run the full RLM REPL loop.

    1. Initialize REPL with context
    2. Iterate: model writes ```repl``` code → sandbox executes → results fed back
    3. Return final answer (via FINAL() or max iterations)

    This is the core RLM pattern from arXiv:2512.24601v1, adapted to use
    Hermes's call_llm for sub-queries.
    """
    from agent.auxiliary_client import call_llm

    # Wipe the log at the start of each run
    try:
        with open(_RLM_LOG, "w") as f:
            pass
    except Exception:
        pass

    # Count messages and sessions for metadata
    import re as _re
    context_message_count = len(_re.split(r'(?=\[session:)', context.strip()))
    context_total_length = len(context)

    # Extract unique session IDs and build a preview of the first few messages
    _session_ids_in_context = set()
    _preview_lines = []
    _msg_count = 0
    for _m in _re.finditer(r'\[session:(\S+)\s+role:(\w+)\]\s*(.*?)(?=\[session:|$)', context, _re.DOTALL):
        _sid, _role, _body = _m.group(1), _m.group(2), _m.group(3).strip()
        _session_ids_in_context.add(_sid)
        if _msg_count < 5:
            _preview_lines.append(f"[session:{_sid} role:{_role}] {_body[:200]}")
        _msg_count += 1
    context_session_count = len(_session_ids_in_context)
    context_preview = "\n".join(_preview_lines) if _preview_lines else "(empty context)"

    env = REPLEnv(context_str=context, max_llm_tokens=max_llm_tokens, hermes_home=hermes_home, session_ids=session_ids)
    messages = build_system_prompt(
        context_total_length=context_total_length,
        context_message_count=context_message_count,
        context_session_count=context_session_count,
        context_preview=context_preview,
    )

    _log(f"starting loop — {len(messages)} messages in history, {context_total_length} chars context, {context_message_count} messages")
    for i in range(max_iterations):
        _log(f"=== iteration {i} ===")
        # Ask the model for its next action
        messages.append(next_action_prompt(query, i))

        response = call_llm(
            task="rlm",
            main_runtime={},
            messages=messages,
            max_tokens=2048,
        )
        content = response.choices[0].message.content or ""
        content = content.strip()
        _log(f"model response ({len(content)} chars):\n{content[:2000]}")

        # Check for FINAL answer
        final = find_final_answer(content)
        if final:
            _log(f"FINAL detected — type={final[0]}, content={final[1][:500]}")
            answer_type, answer_content = final
            if answer_type == 'FINAL':
                return answer_content
            elif answer_type == 'FINAL_VAR':
                var_name = answer_content.strip().strip('"').strip("'")
                if var_name in env.locals:
                    return str(env.locals[var_name])
                # Variable not found — tell the model
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": f"Variable '{var_name}' not found. Use FINAL(answer) instead."})
                continue

        # Execute any ```repl``` code blocks
        code_blocks = find_code_blocks(content)
        if code_blocks:
            _log(f"found {len(code_blocks)} code block(s)")
            messages.append({"role": "assistant", "content": content})
            for code in code_blocks:
                _log(f"executing code:\n{code[:1000]}")
                result = env.code_execution(code)
                _log(f"execution result — stdout={len(result.stdout or '')} chars, stderr={len(result.stderr or '')} chars")
                if result.stdout:
                    _log(f"stdout (full):\n{result.stdout[:3000]}")
                if result.stderr:
                    _log(f"stderr:\n{result.stderr[:1000]}")
                output = result.stdout or ""
                if result.stderr:
                    output += f"\n[stderr] {result.stderr}"
                # Return metadata only — per the paper (Algorithm 1, line 8):
                # "hist ← hist ∥ code ∥ Metadata(stdout)"
                # Only constant-size metadata goes back into history.
                # The actual data stays in REPL variables.
                output_len = len(output)
                if output_len > 200:
                    output = output[:200] + f"\n... ({output_len} chars total, use llm_query on variables to inspect)"
                messages.append({
                    "role": "user",
                    "content": f"Code executed:\n```python\n{code}\n```\n\nOutput:\n{output}",
                })
        else:
            _log("no code blocks — model reasoning only")
            # No code blocks — model is just reasoning, let it continue
            messages.append({"role": "assistant", "content": content})

    # Max iterations — synthesize from accumulated findings
    # Collect all REPL output from the conversation
    findings = []
    for msg in messages:
        if msg.get("role") == "user" and "Code executed:" in msg.get("content", ""):
            # Extract output from execution results
            content = msg["content"]
            output_start = content.find("Output:\n")
            if output_start >= 0:
                findings.append(content[output_start + 8:][:3000])

    if findings:
        synthesis_prompt = (
            f"Based on these findings from searching archived messages, "
            f"answer this question concisely: {query}\n\n"
            f"Findings:\n" + "\n---\n".join(findings[-6:])  # last 6 chunks
        )
    else:
        synthesis_prompt = final_answer_prompt(query)

    messages.append({"role": "user", "content": synthesis_prompt})
    response = call_llm(
        task="rlm",
        main_runtime={},
        messages=messages,
        max_tokens=2048,
    )
    content = response.choices[0].message.content or ""
    final = find_final_answer(content)
    if final:
        return final[1]
    # Graceful fallback: strip code blocks, return prose
    stripped = re.sub(r'```repl\s*\n.*?\n```', '', content, flags=re.DOTALL).strip()
    stripped = re.sub(r'```\w*\s*\n.*?\n```', '', stripped, flags=re.DOTALL).strip()
    if stripped and len(stripped) > 20:
        return stripped
    return content.strip()
