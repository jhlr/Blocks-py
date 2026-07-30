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

from blocks import Block


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
	("return_in_loop_runs_finally", _return_in_loop_runs_finally, [{"n": 5, "stop": 2}, {"n": 5, "stop": 99}], True),
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


if __name__ == "__main__":
	import sys
	sys.exit(1 if run() else 0)
