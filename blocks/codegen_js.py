from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import json
from .operators import BINARY_OPS, SHORTCIRCUIT_OPS, UNARY_OPS
from .nodes import (
	AssignNode, AttrExpr, BinOpExpr, BlockNode, BreakNode, CallExpr, ConcatExpr, CondExpr, ContinueNode, DictExpr, Expr, ExprNode, ForNode, IfNode, IndexExpr, ListExpr, LiteralExpr, Node, RangeExpr, ReturnNode, SetAttrNode, SetIndexNode, SliceExpr, TryNode, UnaryExpr, VarExpr, WhileNode,
)
from .codegen_base import _CodegenContext




###############################################################################
# Code generation — JavaScript backend
#
# Mirrors the Python code generator over the *same* AST, reading operator
# symbols from the shared operator tables (single source of truth). JavaScript
# is dynamically typed like Python, so the mapping is close; the deliberate,
# documented divergences are:
#   - Truthiness of [] and {} (truthy in JS, falsy in Python). Conditions and
#     and/or map straight to JS, so programs relying on empty-container
#     falsiness will differ. 0, "" and null match.
#   - == / != map to === / !== (identity for objects, not deep equality).
#   - Missing variables read as null (via _get), matching Python's None.
#   - Exception *types* are language-specific; try/except is emitted as a
#     structural try/catch/finally matching on the error's name/constructor,
#     which is meaningful for JS-thrown errors, not Python ones.
#   - Attribute/method names (b.x.attr("upper")) and keyword call arguments are
#     Python-specific and not portable; the JS backend raises on kwargs.
###############################################################################

_JS_PRELUDE = """\
const _get = (state, k) => (k in state ? state[k] : null);
const _range = (start, stop, step) => {
  if (step === undefined) step = 1;
  const out = [];
  if (step > 0) { for (let i = start; i < stop; i += step) out.push(i); }
  else { for (let i = start; i > stop; i += step) out.push(i); }
  return out;
};
const _matches = (e, name) => !!e && (e.name === name || (e.constructor && e.constructor.name === name));
const _slice = (seq, start, stop, step) => {
  const isStr = typeof seq === "string";
  const n = seq.length;
  step = (step === null || step === undefined) ? 1 : step;
  let lo, hi;
  if (step > 0) {
    lo = (start === null || start === undefined) ? 0 : (start < 0 ? Math.max(0, n + start) : Math.min(n, start));
    hi = (stop === null || stop === undefined) ? n : (stop < 0 ? Math.max(0, n + stop) : Math.min(n, stop));
  } else {
    lo = (start === null || start === undefined) ? n - 1 : (start < 0 ? Math.max(-1, n + start) : Math.min(n - 1, start));
    hi = (stop === null || stop === undefined) ? -1 : (stop < 0 ? Math.max(-1, n + stop) : Math.min(n - 1, stop));
  }
  const out = [];
  if (step > 0) { for (let i = lo; i < hi; i += step) out.push(seq[i]); }
  else { for (let i = lo; i > hi; i += step) out.push(seq[i]); }
  return isStr ? out.join("") : out;
};
"""




def _concat_part_to_js(part: Expr) -> str:
	if isinstance(part, LiteralExpr) and isinstance(part.value, str):
		return _js_literal(part.value)
	return f"String({_expr_to_js(part)})"




def _js_literal(value: Any) -> str:
	"""Render a constant Python value as JavaScript source.

	Uses JSON, so only JSON-compatible values are supported (None->null,
	True->true, numbers, strings, and lists/dicts thereof).
	"""
	try:
		return json.dumps(value)
	except (TypeError, ValueError) as exc:
		raise ValueError(
			f"Cannot emit non-JSON literal for JavaScript: {value!r}"
		) from exc




def _exc_type_to_js_cond(t: Union[type, Tuple[type, ...]], e_name: str) -> str:
	"""Build a JS boolean expression testing whether error `e_name` matches."""
	types = t if isinstance(t, tuple) else (t,)
	checks = [f"_matches({e_name}, {json.dumps(sub.__name__)})" for sub in types]
	return "(" + " || ".join(checks) + ")"




def _expr_to_js(expr: Expr) -> str:
	if isinstance(expr, LiteralExpr):
		return _js_literal(expr.value)
	if isinstance(expr, VarExpr):
		return f"_get(state, {json.dumps(expr.name)})"
	if isinstance(expr, UnaryExpr):
		unary = UNARY_OPS.get(expr.op)
		if unary is None:
			raise ValueError(f"Unknown unary op: {expr.op}")
		return f"({unary[2]}{_expr_to_js(expr.operand)})"
	if isinstance(expr, BinOpExpr):
		if expr.op in BINARY_OPS:
			symbol = BINARY_OPS[expr.op][2]
		elif expr.op in SHORTCIRCUIT_OPS:
			symbol = SHORTCIRCUIT_OPS[expr.op][1]
		else:
			raise ValueError(f"Unknown binary op: {expr.op}")
		return f"({_expr_to_js(expr.left)} {symbol} {_expr_to_js(expr.right)})"
	if isinstance(expr, IndexExpr):
		if isinstance(expr.index, SliceExpr):
			sl = expr.index
			lo = _expr_to_js(sl.lower) if sl.lower is not None else "null"
			up = _expr_to_js(sl.upper) if sl.upper is not None else "null"
			st = _expr_to_js(sl.step) if sl.step is not None else "null"
			return f"_slice({_expr_to_js(expr.obj)}, {lo}, {up}, {st})"
		return f"({_expr_to_js(expr.obj)})[{_expr_to_js(expr.index)}]"
	if isinstance(expr, CondExpr):
		# raw JS truthiness, matching this backend's if/while (documented divergence)
		return (f"({_expr_to_js(expr.cond)} ? {_expr_to_js(expr.then)}"
				f" : {_expr_to_js(expr.orelse)})")
	if isinstance(expr, ConcatExpr):
		return "(" + " + ".join(_concat_part_to_js(p) for p in expr.parts) + ")"
	if isinstance(expr, AttrExpr):
		obj_src = _expr_to_js(expr.obj)
		if expr.name.isidentifier():
			return f"({obj_src}).{expr.name}"
		return f"({obj_src})[{json.dumps(expr.name)}]"
	if isinstance(expr, ListExpr):
		return "[" + ", ".join(_expr_to_js(e) for e in expr.elts) + "]"
	if isinstance(expr, DictExpr):
		items = ", ".join(f"[{_expr_to_js(k)}]: {_expr_to_js(v)}"
						  for k, v in zip(expr.keys, expr.values))
		return "{" + items + "}"
	if isinstance(expr, CallExpr):
		if expr.kwargs:
			raise ValueError("Keyword arguments are not portable to JavaScript")
		func_src = _expr_to_js(expr.func)
		args_src = ", ".join(_expr_to_js(a) for a in expr.args)
		return f"({func_src})({args_src})"
	if isinstance(expr, RangeExpr):
		start = _expr_to_js(expr.start)
		stop = _expr_to_js(expr.stop)
		if expr.step is None:
			return f"_range({start}, {stop})"
		return f"_range({start}, {stop}, {_expr_to_js(expr.step)})"
	raise TypeError(f"Unknown Expr type in JS codegen: {expr!r}")




def _gen_js(nodes: List[Node], ctx: _CodegenContext) -> None:
	for node in nodes:
		if isinstance(node, AssignNode):
			ctx.add_line(f"state[{json.dumps(node.name)}] = {_expr_to_js(node.expr)};")

		elif isinstance(node, SetIndexNode):
			ctx.add_line(f"({_expr_to_js(node.obj)})[{_expr_to_js(node.index)}]"
						 f" = {_expr_to_js(node.value)};")

		elif isinstance(node, SetAttrNode):
			obj_src = _expr_to_js(node.obj)
			val_src = _expr_to_js(node.value)
			if node.name.isidentifier():
				ctx.add_line(f"({obj_src}).{node.name} = {val_src};")
			else:
				ctx.add_line(f"({obj_src})[{json.dumps(node.name)}] = {val_src};")

		elif isinstance(node, ExprNode):
			ctx.add_line(f"{_expr_to_js(node.expr)};")

		elif isinstance(node, ReturnNode):
			ctx.add_line(f'state["return"] = {_expr_to_js(node.expr)};')
			ctx.add_line("return state;")

		elif isinstance(node, BreakNode):
			ctx.add_line("break;")

		elif isinstance(node, ContinueNode):
			ctx.add_line("continue;")

		elif isinstance(node, IfNode):
			ctx.add_line(f"if ({_expr_to_js(node.cond)}) {{")
			ctx.indent += 1
			_gen_js(node.body, ctx)
			ctx.indent -= 1
			for e in node.elifs:
				ctx.add_line(f"}} else if ({_expr_to_js(e.cond)}) {{")
				ctx.indent += 1
				_gen_js(e.body, ctx)
				ctx.indent -= 1
			if node.else_body is not None:
				ctx.add_line("} else {")
				ctx.indent += 1
				_gen_js(node.else_body, ctx)
				ctx.indent -= 1
			ctx.add_line("}")

		elif isinstance(node, ForNode):
			item_var = ctx.next_tmp("__item")
			ctx.add_line(f"for (const {item_var} of {_expr_to_js(node.iterable)}) {{")
			ctx.indent += 1
			ctx.add_line(f"state[{json.dumps(node.var)}] = {item_var};")
			_gen_js(node.body, ctx)
			ctx.indent -= 1
			ctx.add_line("}")

		elif isinstance(node, WhileNode):
			ctx.add_line(f"while ({_expr_to_js(node.cond)}) {{")
			ctx.indent += 1
			_gen_js(node.body, ctx)
			ctx.indent -= 1
			ctx.add_line("}")

		elif isinstance(node, TryNode):
			# JS finally runs natively on return/break/continue/throw, so we
			# do not need the sentinel-exception dance the Python backend uses.
			ctx.add_line("try {")
			ctx.indent += 1
			_gen_js(node.body, ctx)
			ctx.indent -= 1
			if node.excepts:
				e_name = ctx.next_tmp("__e")
				ctx.add_line(f"}} catch ({e_name}) {{")
				ctx.indent += 1
				for i, ex in enumerate(node.excepts):
					cond = _exc_type_to_js_cond(ex.exc_type, e_name)
					keyword = "if" if i == 0 else "} else if"
					ctx.add_line(f"{keyword} ({cond}) {{")
					ctx.indent += 1
					if ex.name is not None:
						ctx.add_line(f"state[{json.dumps(ex.name)}] = {e_name};")
					_gen_js(ex.body, ctx)
					ctx.indent -= 1
				ctx.add_line("} else {")
				ctx.indent += 1
				ctx.add_line(f"throw {e_name};")
				ctx.indent -= 1
				ctx.add_line("}")
				ctx.indent -= 1
			if node.finally_body is not None:
				ctx.add_line("} finally {")
				ctx.indent += 1
				_gen_js(node.finally_body, ctx)
				ctx.indent -= 1
			ctx.add_line("}")

		else:
			raise TypeError(f"Unknown Node type in JS codegen: {node!r}")




def _generate_js_function_source(root: BlockNode, fn_name: str, prestate: Dict[str, Any]) -> str:
	ctx = _CodegenContext()
	ctx.add_line(f"const PRESTATE = {_js_literal(prestate)};")
	ctx.add_line("")
	ctx.add_line(f"function {fn_name}(argstate) {{")
	ctx.indent += 1
	ctx.add_line("const state = Object.assign({}, PRESTATE);")
	ctx.add_line("if (argstate) Object.assign(state, argstate);")
	ctx.add_line('delete state["return"]; delete state["error"];')
	ctx.add_line("try {")
	ctx.indent += 1
	_gen_js(root.body, ctx)
	ctx.indent -= 1
	ctx.add_line("} catch (__e_top) {")
	ctx.indent += 1
	ctx.add_line('state["error"] = __e_top;')
	ctx.indent -= 1
	ctx.add_line("}")
	ctx.add_line("return state;")
	ctx.indent -= 1
	ctx.add_line("}")
	return "\n".join(ctx.lines)
