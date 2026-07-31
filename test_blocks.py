"""Golden anti-drift harness for Blocks-py.

The interpreter (`Block.__call__`) and the compiler (`Block.compile`) are two
backends over the *same* AST. This suite builds a set of representative
programs and asserts, for each, that interpreting and compiling produce
identical results. If the two backends ever diverge, a test here fails.

Run with plain CPython, no dependencies:

    python test_blocks.py
"""

from __future__ import annotations
import json
import shutil
import subprocess
import tempfile
from typing import Any, Callable, Dict, List, Optional, Tuple

from blocks import Block, parse_python, parse_js


# Each case: (name, build_fn, list_of_argstates, js_safe).
# build_fn receives a fresh Block and populates it; we then run every argstate
# through both the interpreter and the compiled function and compare states.
# js_safe marks programs that are portable to JavaScript (no Python-specific
# exception types, method names, or empty-container truthiness) and so can be
# cross-checked against the JS backend via node.
Case = Tuple[str, Callable[[Block], None], List[Dict[str, Any]], bool]


def _sum_1_to_n(b: Block) -> None:
	b.total = 0
	with b.for_(b.range(1, b.n + 1), var="i"):
		b.total = b.total + b.i
	b.return_(b.total)


def _augmented_assign(b: Block) -> None:
	# += must desugar to state[x] = state[x] + 1 via operator overloading
	b.acc = 0
	with b.for_(b.range(0, b.n), var="i"):
		b.acc += b.i
	b.return_(b.acc)


def _if_elif_else(b: Block) -> None:
	with b.if_(b.x < 0):
		b.sign = -1
	with b.if_(b.x == 0):
		b.sign = 0
	with b.if_(b.x > 0):
		b.sign = 1
	b.return_(b.sign)


def _short_circuit(b: Block) -> None:
	# and/or must short-circuit and return the operand value, not a bool
	b.r = b.a.and_(b.b)
	b.s = b.a.or_(b.b)
	b.return_([b.r, b.s])


def _while_break_continue(b: Block) -> None:
	b.i = 0
	b.acc = 0
	loop = b.while_(b.i < b.n)
	with loop:
		b.i = b.i + 1
		with b.if_(b.i == 3):
			loop.continue_()
		with b.if_(b.i == 6):
			loop.break_()
		b.acc = b.acc + b.i
	b.return_(b.acc)


def _try_except_finally(b: Block) -> None:
	t = b.try_()
	with t:
		b.y = b.x / b.d
	with t.except_(ZeroDivisionError):
		b.y = -1
	with t.finally_():
		b.ran_finally = True
	b.return_(b.y)


def _subscript_and_attr(b: Block) -> None:
	# index into a list from state, and call a method on a value
	b.first = b.items[0]
	b.upper = b.word.attr("upper").call()  # word.upper()
	b.return_([b.first, b.upper])


def _list_dict_literals(b: Block) -> None:
	b.pair = [b.x, b.x + 1]
	b.mapping = {"lo": b.x, "hi": b.x * 10}
	b.return_([b.pair, b.mapping])


def _nested_expr_chaining(b: Block) -> None:
	# (x + 1) * 2 - only composes if operators live on Expr itself
	b.r = (b.x + 1) * 2 - b.x
	b.return_(b.r)


def _empty_bodies(b: Block) -> None:
	# empty block bodies must still generate valid target code
	b.r = 0
	with b.if_(b.x > 0):
		pass
	loop = b.while_(b.x > 100)
	with loop:
		pass
	b.return_(b.r)


def _return_in_loop_runs_finally(b: Block) -> None:
	t = b.try_()
	with t:
		with b.for_(b.range(0, b.n), var="i"):
			with b.if_(b.i == b.stop):
				b.return_(b.i)
	with t.finally_():
		b.cleaned = True


CASES: List[Case] = [
	("sum_1_to_n", _sum_1_to_n, [{"n": 5}, {"n": 10}, {"n": 0}], True),
	("augmented_assign", _augmented_assign, [{"n": 4}, {"n": 0}], True),
	("if_elif_else", _if_elif_else, [{"x": -3}, {"x": 0}, {"x": 7}], True),
	("short_circuit", _short_circuit, [{"a": 0, "b": 9}, {"a": 5, "b": 0}, {"a": "", "b": "x"}], True),
	("while_break_continue", _while_break_continue, [{"n": 10}, {"n": 2}], True),
	# division-by-zero throws in Python but yields Infinity in JS, and the
	# exception type is Python-specific -> not cross-checked against node.
	("try_except_finally", _try_except_finally, [{"x": 10, "d": 2}, {"x": 10, "d": 0}], False),
	# str.upper() has no JS equivalent by that name -> not portable.
	("subscript_and_attr", _subscript_and_attr, [{"items": [42, 1], "word": "hi"}], False),
	("list_dict_literals", _list_dict_literals, [{"x": 3}], True),
	("nested_expr_chaining", _nested_expr_chaining, [{"x": 4}, {"x": -2}], True),
	("empty_bodies", _empty_bodies, [{"x": 5}, {"x": -1}], True),
	("return_in_loop_runs_finally", _return_in_loop_runs_finally, [{"n": 5, "stop": 2}, {"n": 5, "stop": 99}], True),
]


# ---------------------------------------------------------------------------
# Source-level cases: parse REAL Python -> Block, then check that interpreting,
# compiling to Python, running the original Python, and (when portable) running
# the JavaScript backend all return the same value.
# Each case: (name, source, argstates, js_safe). Source is a single `def`.
# ---------------------------------------------------------------------------
SourceCase = Tuple[str, str, List[Dict[str, Any]], bool]

SOURCE_CASES: List[SourceCase] = [
	("sum_even_odd", """
def sum_even_odd(n, base=0):
    total = base
    for i in range(1, n + 1):
        if i % 2 == 0:
            total += i * 2
        else:
            total += i
    return total
""", [{"n": 5}, {"n": 6, "base": 100}, {"n": 0}], True),

	("classify", """
def classify(x, y):
    if x < 0 and y < 0:
        return -1
    elif x == 0 or y == 0:
        return 0
    else:
        return 1
""", [{"x": -2, "y": -3}, {"x": 0, "y": 9}, {"x": 4, "y": 5}], True),

	("countdown", """
def countdown(n):
    acc = 0
    i = 0
    while i < n:
        i += 1
        if i == 3:
            continue
        if i == 7:
            break
        acc += i
    return acc
""", [{"n": 10}, {"n": 2}], True),

	("pow_mod", """
def pow_mod(a, b, m):
    return (a ** b) % m
""", [{"a": 3, "b": 4, "m": 7}, {"a": 2, "b": 10, "m": 100}], True),

	# try/except with a Python-specific exception type; division by zero throws
	# in Python but not JS -> not cross-checked against node.
	("safe_div", """
def safe_div(x, d):
    try:
        r = x / d
    except ZeroDivisionError:
        r = -1
    finally:
        pass
    return r
""", [{"x": 10, "d": 2}, {"x": 10, "d": 0}], False),

	("build_struct", """
def build_struct(x):
    pair = [x, x + 1]
    mapping = {"lo": x, "hi": x * 10}
    return mapping["hi"] + pair[0]
""", [{"x": 3}, {"x": 0}], True),

	# v2: assignment to subscript targets, including augmented subscript assign.
	("mutate_struct", """
def mutate_struct(x):
    arr = [0, 0, 0]
    arr[1] = x * 10
    d = {}
    d["k"] = x + 1
    arr[2] += x
    return arr[1] + d["k"] + arr[2]
""", [{"x": 5}, {"x": 0}], True),

	# and/or must short-circuit and return the operand (not a bool) in every
	# backend — exercises the JS &&/|| and the Lua _and/_or helpers.
	("logic", """
def logic(a, b):
    x = a and b
    y = a or b
    return [x, y]
""", [{"a": 0, "b": 9}, {"a": 5, "b": 0}, {"a": "", "b": "x"}], True),
]


def _comparable(state: Dict[str, Any]) -> Dict[str, Any]:
	"""Normalize a result state for comparison.

	state['error'] holds an exception instance (never equal across runs), so
	compare by (type, args) instead of identity.
	"""
	out = dict(state)
	if "error" in out and isinstance(out["error"], BaseException):
		err = out["error"]
		out["error"] = (type(err).__name__, err.args)
	return out


def _json_norm(state: Dict[str, Any]) -> Any:
	"""Normalize a state for cross-language comparison via a JSON round-trip."""
	return json.loads(json.dumps(state, sort_keys=True))


def _run_js(source: str, fn_name: str, argstate: Dict[str, Any]) -> Any:
	"""Run compiled JS under node and return the resulting state (parsed)."""
	driver = (
		source
		+ "\nconst __arg = JSON.parse(process.argv[2]);"
		+ f"\nprocess.stdout.write(JSON.stringify({fn_name}(__arg)));\n"
	)
	with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
		f.write(driver)
		tmp = f.name
	proc = subprocess.run(
		["node", tmp, json.dumps(argstate)],
		capture_output=True, text=True, check=True,
	)
	return json.loads(proc.stdout)


def run() -> int:
	failures = 0
	node = shutil.which("node")
	if node is None:
		print("note: `node` not found — skipping JavaScript cross-checks")
	js_checked = 0

	for name, build, argstates, js_safe in CASES:
		# Build once per backend (a Block is single-use as a builder target).
		b_interp = Block()
		build(b_interp)

		b_compile = Block()
		build(b_compile)
		compiled, source = b_compile.compile(f"cf_{name}")

		b_js = Block()
		build(b_js)
		js_fn = f"cf_{name}"
		js_source = b_js.compile_js(js_fn) if (node and js_safe) else None

		for argstate in argstates:
			interp_state = _comparable(b_interp(dict(argstate)))
			compiled_state = _comparable(compiled(dict(argstate)))
			if interp_state != compiled_state:
				failures += 1
				print(f"FAIL {name} on {argstate} [python interp vs compile]")
				print(f"  interpret: {interp_state}")
				print(f"  compile:   {compiled_state}")
				print("  --- generated source ---")
				print(source)
				continue

			if js_source is not None:
				js_state = _run_js(js_source, js_fn, argstate)
				if _json_norm(b_interp(dict(argstate))) != _json_norm(js_state):
					failures += 1
					print(f"FAIL {name} on {argstate} [python vs javascript]")
					print(f"  python: {_json_norm(b_interp(dict(argstate)))}")
					print(f"  js:     {_json_norm(js_state)}")
					print("  --- generated js ---")
					print(js_source)
					continue
				js_checked += 1
				print(f"ok   {name} {argstate} (py==compile==js)")
			else:
				suffix = "" if js_safe else " (js: n/a — non-portable)"
				print(f"ok   {name} {argstate} (py==compile){suffix}")

	total = sum(len(c[2]) for c in CASES)
	if failures:
		print(f"\n{failures} divergence(s) detected")
	else:
		print(f"\nall {total} checks passed across {len(CASES)} programs — "
			  f"interpreter and Python compiler agree; "
			  f"{js_checked} also verified against the JavaScript backend")
	return failures


def _run_js_return(source: str, fn_name: str, argstate: Dict[str, Any]) -> Any:
	"""Run compiled JS and return only state['return']."""
	return _run_js(source, fn_name, argstate).get("return")


# Minimal JSON encoder appended to the Lua driver (Lua has no built-in JSON).
# 0-indexed contiguous tables are emitted as arrays so list returns round-trip.
_LUA_DUMP = r'''
local function _is_array(t)
  local n = 0
  for _ in pairs(t) do n = n + 1 end
  for i = 0, n - 1 do if t[i] == nil then return false, n end end
  return true, n
end
local function _dump(v)
  local ty = type(v)
  if v == nil then return "null" end
  if ty == "boolean" then return tostring(v) end
  if ty == "number" then return tostring(v) end
  if ty == "string" then return '"' .. v:gsub('\\', '\\\\'):gsub('"', '\\"'):gsub('\n', '\\n') .. '"' end
  if ty == "table" then
    local arr, n = _is_array(v)
    if arr then
      local parts = {}
      for i = 0, n - 1 do parts[#parts + 1] = _dump(v[i]) end
      return "[" .. table.concat(parts, ",") .. "]"
    end
    local parts = {}
    for k, val in pairs(v) do parts[#parts + 1] = '"' .. tostring(k) .. '":' .. _dump(val) end
    return "{" .. table.concat(parts, ",") .. "}"
  end
  return "null"
end
'''


def _run_lua_return(source: str, fn_name: str, argstate: Dict[str, Any]) -> Any:
	"""Run compiled Lua under `lua` and return the parsed state['return']."""
	from blocks import _lua_value  # reuse the value->Lua-literal renderer
	driver = (source + _LUA_DUMP
			  + f"\nlocal __r = {fn_name}({_lua_value(argstate)})"
			  + '\nio.write(_dump(__r["return"]))\n')
	with tempfile.NamedTemporaryFile("w", suffix=".lua", delete=False) as f:
		f.write(driver)
		tmp = f.name
	proc = subprocess.run(["lua", tmp], capture_output=True, text=True, check=True)
	return json.loads(proc.stdout)


# ---------------------------------------------------------------------------
# JS-source cases: parse REAL JavaScript -> Block, then check that interpreting
# and compiling to Python match the ORIGINAL JavaScript run in node. This is the
# JavaScript -> Python direction. Each case: (name, js_source, argstates).
# Args are passed positionally (in key order) to the original JS function.
# ---------------------------------------------------------------------------
JS_SOURCE_CASES: List[Tuple[str, str, List[Dict[str, Any]]]] = [
	("sumEvenOdd", """
function sumEvenOdd(n, base = 0) {
  let total = base;
  for (let i = 1; i <= n; i++) {
    if (i % 2 === 0) { total += i * 2; } else { total += i; }
  }
  return total;
}
""", [{"n": 5}, {"n": 6, "base": 100}, {"n": 0}]),

	("countdown", """
function countdown(n) {
  let acc = 0, i = 0;
  while (i < n) {
    i += 1;
    if (i === 3) { continue; }
    if (i === 7) { break; }
    acc += i;
  }
  return acc;
}
""", [{"n": 10}, {"n": 2}]),

	("forOf", """
function forOf(arr) {
  let s = 0;
  for (const x of arr) { s += x * 2; }
  return s;
}
""", [{"arr": [1, 2, 3]}, {"arr": []}]),

	("dictAccess", """
function dictAccess(x) {
  const d = {lo: x, hi: x * 10};
  d["mid"] = x + 5;
  return d.lo + d.hi + d["mid"];
}
""", [{"x": 3}, {"x": 0}]),

	("logic", """
function logic(a, b) {
  return [a && b, a || b, !a];
}
""", [{"a": 0, "b": 9}, {"a": 5, "b": 0}]),

	("power", "function power(a, b) { return a ** b; }", [{"a": 2, "b": 10}, {"a": 3, "b": 4}]),
]


def _run_node_reference(source: str, fn_name: str, argstate: Dict[str, Any]) -> Any:
	"""Run the ORIGINAL JavaScript function in node with positional args."""
	driver = (source
			  + "\nconst __a = JSON.parse(process.argv[2]);"
			  + f"\nconst __keys = Object.keys(__a);"
			  + f"\nprocess.stdout.write(JSON.stringify({fn_name}(...__keys.map(k => __a[k]))));")
	with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
		f.write(driver)
		tmp = f.name
	proc = subprocess.run(["node", tmp, json.dumps(argstate)], capture_output=True, text=True, check=True)
	return json.loads(proc.stdout)


def _deep_eq(a: Any, b: Any) -> bool:
	"""Value equality that treats ints and floats as equal (Lua `^` yields floats)."""
	if isinstance(a, bool) or isinstance(b, bool):
		return a is b
	if isinstance(a, (int, float)) and isinstance(b, (int, float)):
		return a == b
	if isinstance(a, list) and isinstance(b, list):
		return len(a) == len(b) and all(_deep_eq(x, y) for x, y in zip(a, b))
	if isinstance(a, dict) and isinstance(b, dict):
		return a.keys() == b.keys() and all(_deep_eq(a[k], b[k]) for k in a)
	return a == b


def run_source() -> int:
	"""Validate the Python front-end: real Python source through every backend."""
	failures = 0
	node = shutil.which("node")
	lua = shutil.which("lua")
	js_checked = 0
	lua_checked = 0

	for name, source, argstates, js_safe in SOURCE_CASES:
		block = parse_python(source)
		compiled, _ = block.compile(f"cf_{name}")
		js_source = block.compile_js(f"cf_{name}") if (node and js_safe) else None
		lua_source = block.compile_lua(f"cf_{name}") if (lua and js_safe) else None

		# reference: the original Python, executed for real
		real_ns: Dict[str, Any] = {}
		exec(source, real_ns)
		real_fn = real_ns[name]

		for argstate in argstates:
			expected = real_fn(**argstate)
			results = {
				"interpret": block(dict(argstate)).get("return"),
				"compile": compiled(dict(argstate)).get("return"),
			}
			if js_source is not None:
				results["js"] = _run_js_return(js_source, f"cf_{name}", argstate)
			if lua_source is not None:
				results["lua"] = _run_lua_return(lua_source, f"cf_{name}", argstate)

			bad = {k: v for k, v in results.items() if not _deep_eq(v, expected)}
			if bad:
				failures += 1
				print(f"FAIL {name} on {argstate}")
				print(f"  {'real':10}: {expected}")
				for k, v in results.items():
					print(f"  {k:10}: {v}")
			else:
				tag = "==".join(["real", *results.keys()])
				print(f"ok   {name} {argstate} ({tag})")
				if js_source is not None:
					js_checked += 1
				if lua_source is not None:
					lua_checked += 1

	total = sum(len(c[2]) for c in SOURCE_CASES)
	if failures:
		print(f"\n{failures} source-level divergence(s) detected")
	else:
		print(f"\nall {total} source checks passed across {len(SOURCE_CASES)} "
			  f"Python programs — parse -> interpret/compile matches real Python; "
			  f"{js_checked} verified against JavaScript, {lua_checked} against Lua")
	return failures


def run_js_source() -> int:
	"""Validate the JS front-end: real JavaScript -> Block matches node, Python."""
	node = shutil.which("node")
	if node is None:
		print("note: `node` not found — skipping JavaScript front-end checks")
		return 0

	failures = 0
	for name, source, argstates in JS_SOURCE_CASES:
		block = parse_js(source)
		compiled, _ = block.compile(f"cf_{name}")
		for argstate in argstates:
			expected = _run_node_reference(source, name, argstate)
			results = {
				"interpret": block(dict(argstate)).get("return"),
				"py_compile": compiled(dict(argstate)).get("return"),
			}
			bad = {k: v for k, v in results.items() if not _deep_eq(v, expected)}
			if bad:
				failures += 1
				print(f"FAIL {name} on {argstate}")
				print(f"  {'node(js)':12}: {expected}")
				for k, v in results.items():
					print(f"  {k:12}: {v}")
			else:
				print(f"ok   {name} {argstate} (node(js)==interpret==py_compile)")

	total = sum(len(c[2]) for c in JS_SOURCE_CASES)
	if failures:
		print(f"\n{failures} JS-source divergence(s) detected")
	else:
		print(f"\nall {total} JS-source checks passed across {len(JS_SOURCE_CASES)} "
			  f"JavaScript programs — parse_js -> interpret/compile matches node")
	return failures


if __name__ == "__main__":
	import sys
	print("=== AST-level: builder vs backends ===")
	f1 = run()
	print("\n=== Source-level: real Python -> backends ===")
	f2 = run_source()
	print("\n=== JS front-end: real JavaScript -> Python ===")
	f3 = run_js_source()
	sys.exit(1 if (f1 + f2 + f3) else 0)
