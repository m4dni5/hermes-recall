"""REPL environment for RLM deep-dive searches (v2: JSON archive + FTS5 index).

Adapted from rlm-minimal (https://github.com/alexzhang13/rlm).
Uses Hermes's call_llm(task="rlm") for sub-queries instead of
a direct OpenAI client, so it automatically picks up the auxiliary model
configured in auxiliary.rlm (or falls back to auxiliary.compression).

v2: Context is a JSON array of messages (not a flat string).
    search_context() returns message indices via FTS5.
    Model queries the JSON array programmatically.
"""

import ast
import io
import json
import os
import re
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


_RLM_LOG = "/tmp/rlm_repl.log"
def _log(msg: str):
    """Temporary debug logger — writes to /tmp/rlm_repl.log."""
    try:
        with open(_RLM_LOG, "a") as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
    except Exception:
        pass


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
    """Sandboxed Python REPL with messages variable, llm_query(), search_context().

    v2: Context is loaded as a JSON array into a `messages` variable.
    search_context() returns message indices via FTS5 for fast lookup.
    """

    def __init__(
        self,
        messages_json: Optional[List[dict]] = None,
        max_llm_tokens: int = 1024,
        hermes_home: Optional[str] = None,
        session_ids: Optional[List[str]] = None,
    ):
        self.original_cwd = os.getcwd()
        self.temp_dir = tempfile.mkdtemp(prefix="rlm_repl_")
        self.max_llm_tokens = max_llm_tokens
        self._hermes_home = hermes_home
        self._session_ids = session_ids
        self._messages = messages_json or []

        # Sandboxed globals
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
                'input': None, 'eval': None, 'exec': None,
                'compile': None, 'globals': None, 'locals': None,
            }
        }
        self.locals: Dict[str, Any] = {}
        self._lock = threading.Lock()

        # Load messages JSON into the REPL namespace
        self._load_messages(messages_json)

        # Expose llm_query
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

        # Expose search_context — FTS5 search returning message indices
        _repl_session_ids = set(self._session_ids) if self._session_ids else set()
        _repl_scoped = bool(_repl_session_ids)

        def search_context(query_str: str, limit: int = 10) -> str:
            """FTS5 search returning message indices + snippets.

            Returns indices into the messages array so the model can do:
                hits = search_context("topic")
                for idx in parse_indices(hits):
                    print(messages[idx]["content"])
            """
            try:
                from hermes_state import SessionDB, DEFAULT_DB_PATH
                from pathlib import Path
                db_path = Path(self._hermes_home) / "state.db" if self._hermes_home else DEFAULT_DB_PATH
                _log(f"search_context: query={query_str!r}, limit={limit}, db_path={db_path}, exists={db_path.exists()}, scoped={_repl_scoped}")
                if not db_path.exists():
                    return "Error: session database not found"
                db = SessionDB(db_path)
                fetch_limit = limit * 5 if _repl_scoped else limit
                results = db.search_messages(query_str, limit=fetch_limit)
                _log(f"search_context: raw results={len(results)}")
                if not results:
                    return f"No results for: {query_str}"
                if _repl_scoped:
                    results = [r for r in results if r.get("session_id") in _repl_session_ids]
                    results = results[:limit]
                    _log(f"search_context: after lineage filter={len(results)}")
                    if not results:
                        return f"No results for: {query_str}"
                # Find matching indices in the messages array
                result_sids = {(r.get("session_id"), r.get("role")): r for r in results}
                matched_indices = []
                for i, msg in enumerate(self._messages):
                    key = (msg.get("sid"), msg.get("role"))
                    if key in result_sids:
                        matched_indices.append(i)
                        if len(matched_indices) >= limit:
                            break
                if not matched_indices:
                    # Fallback: content matching
                    result_contents = [r.get("content", "")[:100] for r in results]
                    for i, msg in enumerate(self._messages):
                        mc = msg.get("content", "")[:100]
                        if any(rc in mc or mc in rc for rc in result_contents if rc):
                            matched_indices.append(i)
                            if len(matched_indices) >= limit:
                                break
                return json.dumps(matched_indices)
            except Exception as e:
                _log(f"search_context ERROR: {e}")
                return f"Search error: {e}"

        self.globals['search_context'] = search_context

        # Expose FINAL_VAR
        def final_var(variable_name: str) -> str:
            variable_name = variable_name.strip().strip('"').strip("'").strip('\n').strip('\r')
            if variable_name in self.locals:
                return str(self.locals[variable_name])
            return f"Error: Variable '{variable_name}' not found"

        self.globals['FINAL_VAR'] = final_var

    def _load_messages(self, messages_json: Optional[List[dict]]):
        """Load messages JSON array into the REPL namespace."""
        if not messages_json:
            self.globals['messages'] = []
            return
        # Write to temp file and load (same pattern as v1)
        json_path = os.path.join(self.temp_dir, "messages.json")
        with open(json_path, "w") as f:
            json.dump(messages_json, f)
        setup_code = (
            f"import json\n"
            f"with open(r'{json_path}', 'r') as f:\n"
            f"    messages = json.load(f)\n"
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
        """Execute Python code in the sandboxed REPL (AST-based)."""
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

                        # AST-based expression detection
                        try:
                            tree = ast.parse(other_code)
                            last_node = tree.body[-1] if tree.body else None
                            is_expr = isinstance(last_node, ast.Expr)
                        except SyntaxError:
                            is_expr = False

                        if is_expr and len(tree.body) > 1:
                            all_lines = other_code.split('\n')
                            last_lineno = last_node.lineno
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
# Parsing utilities
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
# Prompts
# ---------------------------------------------------------------------------

REPL_SYSTEM_PROMPT = """You are tasked with answering a query using archived conversation messages. You have access to a REPL environment where you can write Python code to process the data and query a sub-LLM.

Your context is a JSON array of {message_count} messages across {session_count} sessions ({total_chars} total characters), stored in a `messages` variable.

Each message is a dict: {{"i": index, "sid": "session_id", "role": "user"|"assistant"|"tool", "content": "..."}}

Sample of the first few messages:
{preview}

{search_hints}
The REPL environment has:
1. A `messages` variable — a JSON array of all archived messages. Query it with list comprehensions, filters, etc.
2. A `search_context(query, limit=10)` function — FTS5 full-text search that returns JSON array of message indices. Use this FIRST to find relevant messages quickly, then read full content from `messages[idx]`.
3. A `llm_query(prompt)` function — calls a sub-LLM for semantic analysis. Use it to extract meaning from message content.
4. `print()` to see output. IMPORTANT: print() output is truncated to metadata in the conversation history, but variables are NEVER truncated. If you see "... (N chars total)", the full value is still in the variable.

STRATEGY: Start with search_context() to find relevant message indices. Then read full messages from the messages array. Use llm_query() for semantic analysis when needed.

IMPORTANT — BATCH YOUR OPERATIONS: Each code block is one iteration. Write multi-step code in a single block.

```repl
# GOOD — search, filter, analyze in one block
hits = json.loads(search_context("topic keywords", limit=15))
if hits:
    relevant = [messages[i] for i in hits if messages[i]["role"] == "tool"]
    for msg in relevant[:3]:
        print(f"[{{msg['sid']}}] {{msg['content'][:200]}}")
    analysis = llm_query(f"Summarize key points about X from: {{relevant[0]['content'][:5000]}}")
    print(analysis)
else:
    # Fallback: scan the full array
    matches = [m for m in messages if "keyword" in m["content"].lower()]
    print(f"Found {{len(matches)}} matches via scan")
```

CRITICAL RULES:
- Execute code immediately, don't just describe what you'd do.
- Use llm_query liberally — it's cheap and fast.
- You MUST end with FINAL(your complete answer on a single line). This is the ONLY way to return your answer.
- FINAL() must be on its own line. Do NOT put code blocks after FINAL().
- Answer the original query directly and completely in your FINAL().
- If you have enough information, output FINAL() immediately — do not run unnecessary code."""


def build_system_prompt(
    message_count: int = 0,
    session_count: int = 0,
    total_chars: int = 0,
    preview: str = "",
    search_hints: str = "",
) -> List[Dict[str, str]]:
    content = REPL_SYSTEM_PROMPT.format(
        message_count=message_count,
        session_count=session_count,
        total_chars=total_chars,
        preview=preview,
        search_hints=search_hints,
    )
    return [{"role": "system", "content": content}]


def next_action_prompt(query: str, iteration: int = 0) -> Dict[str, str]:
    base = f'Think step-by-step. Use the REPL environment (which has `messages` and `search_context`) to answer: "{query}".\n\n'
    if iteration == 0:
        return {"role": "user", "content": "You have not interacted with the REPL yet. Start with search_context().\n\n" + base + "Your next action:"}
    return {"role": "user", "content": "Your previous REPL interactions are above. " + base + "Your next action:"}


def final_answer_prompt(query: str) -> Dict[str, str]:
    return {"role": "user", "content": (
        "STOP. Do NOT write any more code. Based on everything you have learned "
        "from the REPL, provide your FINAL answer now.\n\n"
        "You MUST output: FINAL(your complete answer here)\n\n"
        f"Question: {query}"
    )}


# ---------------------------------------------------------------------------
# REPL loop
# ---------------------------------------------------------------------------

def run_rlm_repl(
    messages_json: List[dict],
    query: str,
    max_iterations: int = 12,
    max_llm_tokens: int = 2048,
    hermes_home: Optional[str] = None,
    session_ids: Optional[List[str]] = None,
    fts_hints: Optional[List[int]] = None,
) -> str:
    """Run the RLM REPL loop over a JSON messages array.

    1. Initialize REPL with messages JSON
    2. Iterate: model writes ```repl``` code → sandbox executes → results fed back
    3. Return final answer (via FINAL() or max iterations)
    """
    from agent.auxiliary_client import call_llm

    # Wipe the log
    try:
        with open(_RLM_LOG, "w") as f:
            pass
    except Exception:
        pass

    # Compute metadata
    total_chars = sum(len(m.get("content", "")) for m in messages_json)
    session_ids_in_data = set(m.get("sid", "") for m in messages_json)
    session_count = len(session_ids_in_data)

    # Build preview
    preview_lines = []
    for msg in messages_json[:5]:
        sid = msg.get("sid", "?")
        role = msg.get("role", "?")
        content = msg.get("content", "")[:200]
        preview_lines.append(f'[session:{sid} role:{role}] {content}')
    preview = "\n".join(preview_lines) if preview_lines else "(empty)"

    # Build search hints from pre-computed FTS5 hits
    search_hints = ""
    if fts_hints:
        hint_previews = []
        for idx in fts_hints[:5]:
            if idx < len(messages_json):
                msg = messages_json[idx]
                hint_previews.append(f"  messages[{idx}] ({msg['role']}): {msg['content'][:150]}...")
        search_hints = (
            f"PRE-SEARCH RESULTS: FTS5 found {len(fts_hints)} relevant messages at indices {fts_hints}. "
            f"Start by reading these messages directly — no need to call search_context() first.\n"
            + "\n".join(hint_previews) + "\n"
        )

    env = REPLEnv(
        messages_json=messages_json,
        max_llm_tokens=max_llm_tokens,
        hermes_home=hermes_home,
        session_ids=session_ids,
    )
    messages = build_system_prompt(
        message_count=len(messages_json),
        session_count=session_count,
        total_chars=total_chars,
        preview=preview,
        search_hints=search_hints,
    )

    _log(f"starting loop — {len(messages_json)} messages, {total_chars} chars, {session_count} sessions")

    for i in range(max_iterations):
        _log(f"=== iteration {i} ===")
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
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": f"Variable '{var_name}' not found. Use FINAL(answer) instead."})
                continue

        # Execute code blocks
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
                output_len = len(output)
                if output_len > 200:
                    output = output[:200] + f"\n... ({output_len} chars total, use llm_query on variables to inspect)"
                messages.append({
                    "role": "user",
                    "content": f"Code executed:\n```python\n{code}\n```\n\nOutput:\n{output}",
                })
        else:
            _log("no code blocks — model reasoning only")
            messages.append({"role": "assistant", "content": content})

    # Max iterations — synthesize from accumulated findings
    findings = []
    for msg in messages:
        if msg.get("role") == "user" and "Code executed:" in msg.get("content", ""):
            c = msg["content"]
            output_start = c.find("Output:\n")
            if output_start >= 0:
                findings.append(c[output_start + 8:][:3000])

    if findings:
        synthesis_prompt = (
            f"Based on these findings from searching archived messages, "
            f"answer this question concisely: {query}\n\n"
            f"Findings:\n" + "\n---\n".join(findings[-6:])
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
    stripped = re.sub(r'```repl\s*\n.*?\n```', '', content, flags=re.DOTALL).strip()
    stripped = re.sub(r'```\w*\s*\n.*?\n```', '', stripped, flags=re.DOTALL).strip()
    if stripped and len(stripped) > 20:
        return stripped
    return content.strip()
