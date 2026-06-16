"""REPL environment for RLM deep-dive searches.

Adapted from rlm-minimal (https://github.com/alexzhang13/rlm).
Uses Hermes's call_llm(task="compression") for sub-queries instead of
a direct OpenAI client, so it automatically picks up the auxiliary model
configured in auxiliary.compression.model.
"""

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
    - llm_query() uses Hermes's call_llm(task="compression") instead of
      a direct OpenAI client, so it picks up auxiliary.compression.model.
    - Logging stripped (no repl_env_logger).
    - Context loaded from a string (FTS5 search results).
    """

    def __init__(
        self,
        context_str: Optional[str] = None,
        max_llm_tokens: int = 1024,
    ):
        self.original_cwd = os.getcwd()
        self.temp_dir = tempfile.mkdtemp(prefix="rlm_repl_")
        self.max_llm_tokens = max_llm_tokens

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

        # Expose llm_query — uses Hermes's auxiliary model routing
        def llm_query(prompt: str) -> str:
            """Query the cheap model via Hermes's auxiliary compression config."""
            from agent.auxiliary_client import call_llm
            try:
                response = call_llm(
                    task="compression",
                    main_runtime={},
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_llm_tokens,
                )
                content = response.choices[0].message.content
                return content.strip() if isinstance(content, str) else str(content or "")
            except Exception as e:
                return f"Error calling LLM: {e}"

        self.globals['llm_query'] = llm_query

        # Expose FINAL_VAR — signal completion with a variable value
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

                        # Try to eval the last expression (like a REPL)
                        non_comment = [l for l in other_lines if l and not l.startswith('#')]
                        if non_comment:
                            last = non_comment[-1]
                            is_expr = (
                                not last.startswith(('import ', 'from ', 'def ', 'class ', 'if ', 'for ', 'while ', 'try:', 'with ', 'return ', 'yield ', 'break', 'continue', 'pass'))
                                and '=' not in last.split('#')[0]
                                and not last.endswith(':')
                                and not last.startswith('print(')
                            )
                            if is_expr and len(non_comment) > 1:
                                exec('\n'.join(other_lines[:other_lines.index(last)]), ns, ns)
                                result = eval(last, ns, ns)
                                if result is not None:
                                    print(repr(result))
                            elif is_expr:
                                result = eval(last, ns, ns)
                                if result is not None:
                                    print(repr(result))
                            else:
                                exec(other_code, ns, ns)
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
    m = re.search(r'^\s*FINAL_VAR\((.*?)\)', text, re.MULTILINE | re.DOTALL)
    if m:
        return ('FINAL_VAR', m.group(1).strip())
    m = re.search(r'^\s*FINAL\((.*?)\)', text, re.MULTILINE | re.DOTALL)
    if m:
        return ('FINAL', m.group(1).strip())
    return None


# ---------------------------------------------------------------------------
# Prompts (adapted from rlm-minimal prompts.py)
# ---------------------------------------------------------------------------

REPL_SYSTEM_PROMPT = """You are tasked with answering a query using archived conversation messages. You have access to a REPL environment where you can write Python code to process the data and query a sub-LLM.

The REPL environment has:
1. A `context` string containing archived messages (search results from a conversation database). Read it first to understand what's available.
2. A `llm_query(prompt)` function that calls a cheap LLM. Use it to analyze chunks of context — it can handle large inputs.
3. `print()` to see output. `FINAL(answer)` or `FINAL_VAR(variable)` to return your answer.

STRATEGY: The context may be large. Read it, chunk it by message boundaries (look for [session:... role:...] markers), and call llm_query on each chunk to extract relevant information. Then synthesize.

When you want to execute Python code, wrap it in triple backticks with 'repl':
```repl
# Read the context
print(context[:500])
```

```repl
# Chunk and query
chunks = context.split('\\n\\n')
results = []
for chunk in chunks:
    answer = llm_query(f"Extract info relevant to the question from this chunk: {chunk}")
    if 'NO_RELEVANT_INFO' not in answer.upper():
        results.append(answer)
final = llm_query(f"Synthesize these findings: {results}")
```

IMPORTANT:
- Execute code immediately, don't just describe what you'd do.
- Use llm_query liberally — it's cheap and fast.
- When done, output FINAL(your answer) or FINAL_VAR(variable_name).
- Answer the original query directly in your final answer."""


def build_system_prompt() -> List[Dict[str, str]]:
    return [{"role": "system", "content": REPL_SYSTEM_PROMPT}]


def next_action_prompt(query: str, iteration: int = 0) -> Dict[str, str]:
    base = f'Think step-by-step. Use the REPL environment (which has `context`) to answer: "{query}".\n\n'
    if iteration == 0:
        return {"role": "user", "content": "You have not interacted with the REPL yet. Read the context first.\n\n" + base + "Your next action:"}
    return {"role": "user", "content": "Your previous REPL interactions are above. " + base + "Your next action:"}


def final_answer_prompt(query: str) -> Dict[str, str]:
    return {"role": "user", "content": "Based on all the information you have, provide a final answer to: " + query}


# ---------------------------------------------------------------------------
# RLM REPL loop
# ---------------------------------------------------------------------------

def run_rlm_repl(
    context: str,
    query: str,
    max_iterations: int = 8,
    max_llm_tokens: int = 1024,
) -> str:
    """Run the full RLM REPL loop.

    1. Initialize REPL with context
    2. Iterate: model writes ```repl``` code → sandbox executes → results fed back
    3. Return final answer (via FINAL() or max iterations)

    This is the core RLM pattern from arXiv:2512.24601v1, adapted to use
    Hermes's call_llm for sub-queries.
    """
    from agent.auxiliary_client import call_llm

    env = REPLEnv(context_str=context, max_llm_tokens=max_llm_tokens)
    messages = build_system_prompt()

    for i in range(max_iterations):
        # Ask the model for its next action
        messages.append(next_action_prompt(query, i))

        response = call_llm(
            task="compression",
            main_runtime={},
            messages=messages,
            max_tokens=2048,
        )
        content = response.choices[0].message.content or ""
        content = content.strip()

        # Check for FINAL answer
        final = find_final_answer(content)
        if final:
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
            messages.append({"role": "assistant", "content": content})
            for code in code_blocks:
                result = env.code_execution(code)
                output = result.stdout or ""
                if result.stderr:
                    output += f"\n[stderr] {result.stderr}"
                # Truncate large outputs
                if len(output) > 10000:
                    output = output[:10000] + f"\n... (truncated, {len(output)} chars total)"
                messages.append({
                    "role": "user",
                    "content": f"Code executed:\n```python\n{code}\n```\n\nOutput:\n{output}",
                })
        else:
            # No code blocks — model is just reasoning, let it continue
            messages.append({"role": "assistant", "content": content})

    # Max iterations — force a final answer
    messages.append(final_answer_prompt(query))
    response = call_llm(
        task="compression",
        main_runtime={},
        messages=messages,
        max_tokens=2048,
    )
    content = response.choices[0].message.content or ""
    final = find_final_answer(content)
    if final:
        return final[1]
    return content.strip()
