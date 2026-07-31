from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from .operators import BINARY_OPS
from .nodes import (
	AssignNode, AttrExpr, BinOpExpr, BlockNode, BreakNode, CallExpr, ConcatExpr, CondExpr, ContinueNode, DictExpr, Expr, ExprNode, ForNode, IfNode, IndexExpr, ListExpr, LiteralExpr, Node, RangeExpr, ReturnNode, SetAttrNode, SetIndexNode, SliceExpr, TryNode, UnaryExpr, VarExpr, WhileNode,
)
from .codegen_base import _CodegenContext, _lua_escapes_flow




###############################################################################
# Code generation — Lua backend
#
# Lua is dynamically typed, but its truthiness and control flow differ enough
# from Python that the mapping leans on helpers to stay faithful:
#   - Truthiness: Lua treats 0, "", [] and {} as truthy. To match Python, all
#     conditions and and/or go through _truthy (empty tables are read as falsy),
#     so this backend is actually MORE faithful than the JS one on truthiness.
#   - Sequences are 0-indexed tables ({[0]=.., [1]=..}); iteration uses _seq.
#     Lua's native 1-indexing is bypassed so Python indices (x[0]) hold.
#   - `and`/`or` become _and/_or with a thunk for the right side (lazy + operand
#     return); `not` becomes _not; `**` becomes `^`; `!=` becomes `~=`.
#   - `return` is wrapped in `do ... end` (Lua's return must end its block).
#   - `continue` uses a `goto` to a per-loop label.
#   - try/except uses pcall. Because a pcall'd closure can't carry break/
#     continue/return across its boundary, a try body containing control flow
#     that escapes the try raises LuaUnsupportedError (documented, not silently
#     mistranslated). Exception *types* and division-by-zero remain
#     language-specific, as in the JS backend.
###############################################################################

class LuaUnsupportedError(Exception):
	"""A construct that cannot be faithfully compiled to Lua."""




_LUA_PRELUDE = """\
local function _get(state, k) return state[k] end
local function _truthy(v)
  if v == nil or v == false or v == 0 or v == "" then return false end
  if type(v) == "table" then
    for _ in pairs(v) do return true end
    return false
  end
  return true
end
local function _not(v) return not _truthy(v) end
local function _and(a, bfn) if _truthy(a) then return bfn() else return a end end
local function _or(a, bfn) if _truthy(a) then return a else return bfn() end end
local function _range(start, stop, step)
  step = step or 1
  local out, k = {}, 0
  local i = start
  if step > 0 then
    while i < stop do out[k] = i; k = k + 1; i = i + step end
  else
    while i > stop do out[k] = i; k = k + 1; i = i + step end
  end
  return out
end
local function _seq(t)
  local i = -1
  return function()
    i = i + 1
    return t[i]
  end
end
local function _matches(e, name)
  return type(e) == "table" and e.name == name
end
local function _cond(c, afn, bfn)
  if _truthy(c) then return afn() else return bfn() end
end
local function _len(seq)
  if type(seq) == "string" then return #seq end
  local n = 0
  while seq[n] ~= nil do n = n + 1 end
  return n
end
local function _slice(seq, start, stop, step)
  local isStr = type(seq) == "string"
  local n = _len(seq)
  if step == nil then step = 1 end
  local lo, hi
  if step > 0 then
    if start == nil then lo = 0 elseif start < 0 then lo = math.max(0, n + start) else lo = math.min(n, start) end
    if stop == nil then hi = n elseif stop < 0 then hi = math.max(0, n + stop) else hi = math.min(n, stop) end
  else
    if start == nil then lo = n - 1 elseif start < 0 then lo = math.max(-1, n + start) else lo = math.min(n - 1, start) end
    if stop == nil then hi = -1 elseif stop < 0 then hi = math.max(-1, n + stop) else hi = math.min(n - 1, stop) end
  end
  local i = lo
  if isStr then
    local parts = {}
    if step > 0 then while i < hi do parts[#parts + 1] = seq:sub(i + 1, i + 1); i = i + step end
    else while i > hi do parts[#parts + 1] = seq:sub(i + 1, i + 1); i = i + step end end
    return table.concat(parts)
  end
  local out, k = {}, 0
  if step > 0 then while i < hi do out[k] = seq[i]; k = k + 1; i = i + step end
  else while i > hi do out[k] = seq[i]; k = k + 1; i = i + step end end
  return out
end
"""




def _concat_part_to_lua(part: Expr) -> str:
	if isinstance(part, LiteralExpr) and isinstance(part.value, str):
		return _lua_literal(part.value)
	return f"tostring({_expr_to_lua(part)})"




def _lua_literal(value: Any) -> str:
	if value is None:
		return "nil"
	if value is True:
		return "true"
	if value is False:
		return "false"
	if isinstance(value, str):
		s = (value.replace("\\", "\\\\").replace('"', '\\"')
			 .replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t"))
		return f'"{s}"'
	if isinstance(value, (int, float)):
		return repr(value)
	raise ValueError(f"Cannot emit Lua literal for {value!r}")




def _lua_value(value: Any) -> str:
	"""Render an arbitrary JSON-ish Python value as a Lua expression (for PRESTATE)."""
	if isinstance(value, dict):
		items = ", ".join(f"[{_lua_literal(k)}] = {_lua_value(v)}" for k, v in value.items())
		return "{" + items + "}"
	if isinstance(value, (list, tuple)):
		items = ", ".join(f"[{i}] = {_lua_value(v)}" for i, v in enumerate(value))
		return "{" + items + "}"
	return _lua_literal(value)




def _expr_to_lua(expr: Expr) -> str:
	if isinstance(expr, LiteralExpr):
		return _lua_literal(expr.value)
	if isinstance(expr, VarExpr):
		return f"_get(state, {_lua_literal(expr.name)})"
	if isinstance(expr, UnaryExpr):
		inner = _expr_to_lua(expr.operand)
		if expr.op == "neg":
			return f"(-{inner})"
		if expr.op == "pos":
			return f"({inner})"
		if expr.op == "not":
			return f"_not({inner})"
		raise ValueError(f"Unknown unary op: {expr.op}")
	if isinstance(expr, BinOpExpr):
		if expr.op in BINARY_OPS:
			return f"({_expr_to_lua(expr.left)} {BINARY_OPS[expr.op][3]} {_expr_to_lua(expr.right)})"
		if expr.op == "and":
			return f"_and({_expr_to_lua(expr.left)}, function() return {_expr_to_lua(expr.right)} end)"
		if expr.op == "or":
			return f"_or({_expr_to_lua(expr.left)}, function() return {_expr_to_lua(expr.right)} end)"
		raise ValueError(f"Unknown binary op: {expr.op}")
	if isinstance(expr, IndexExpr):
		if isinstance(expr.index, SliceExpr):
			sl = expr.index
			lo = _expr_to_lua(sl.lower) if sl.lower is not None else "nil"
			up = _expr_to_lua(sl.upper) if sl.upper is not None else "nil"
			st = _expr_to_lua(sl.step) if sl.step is not None else "nil"
			return f"_slice({_expr_to_lua(expr.obj)}, {lo}, {up}, {st})"
		return f"({_expr_to_lua(expr.obj)})[{_expr_to_lua(expr.index)}]"
	if isinstance(expr, CondExpr):
		return (f"_cond({_expr_to_lua(expr.cond)},"
				f" function() return {_expr_to_lua(expr.then)} end,"
				f" function() return {_expr_to_lua(expr.orelse)} end)")
	if isinstance(expr, ConcatExpr):
		return "(" + " .. ".join(_concat_part_to_lua(p) for p in expr.parts) + ")"
	if isinstance(expr, AttrExpr):
		obj_src = _expr_to_lua(expr.obj)
		if expr.name.isidentifier():
			return f"({obj_src}).{expr.name}"
		return f"({obj_src})[{_lua_literal(expr.name)}]"
	if isinstance(expr, ListExpr):
		items = ", ".join(f"[{i}] = {_expr_to_lua(e)}" for i, e in enumerate(expr.elts))
		return "{" + items + "}"
	if isinstance(expr, DictExpr):
		items = ", ".join(f"[{_expr_to_lua(k)}] = {_expr_to_lua(v)}"
						  for k, v in zip(expr.keys, expr.values))
		return "{" + items + "}"
	if isinstance(expr, CallExpr):
		if expr.kwargs:
			raise LuaUnsupportedError("keyword arguments are not portable to Lua")
		func_src = _expr_to_lua(expr.func)
		args_src = ", ".join(_expr_to_lua(a) for a in expr.args)
		return f"({func_src})({args_src})"
	if isinstance(expr, RangeExpr):
		start = _expr_to_lua(expr.start)
		stop = _expr_to_lua(expr.stop)
		if expr.step is None:
			return f"_range({start}, {stop})"
		return f"_range({start}, {stop}, {_expr_to_lua(expr.step)})"
	raise TypeError(f"Unknown Expr type in Lua codegen: {expr!r}")




def _gen_lua(nodes: List[Node], ctx: _CodegenContext) -> None:
	for node in nodes:
		if isinstance(node, AssignNode):
			ctx.add_line(f"state[{_lua_literal(node.name)}] = {_expr_to_lua(node.expr)}")

		elif isinstance(node, SetIndexNode):
			# Leading `;` disambiguates a statement starting with `(` from a call
			# on the previous line (Lua's `x = {} \n (y)[i]=..` gotcha).
			ctx.add_line(f";({_expr_to_lua(node.obj)})[{_expr_to_lua(node.index)}]"
						 f" = {_expr_to_lua(node.value)}")

		elif isinstance(node, SetAttrNode):
			obj_src = _expr_to_lua(node.obj)
			val_src = _expr_to_lua(node.value)
			if node.name.isidentifier():
				ctx.add_line(f";({obj_src}).{node.name} = {val_src}")
			else:
				ctx.add_line(f";({obj_src})[{_lua_literal(node.name)}] = {val_src}")

		elif isinstance(node, ExprNode):
			# A bare call is a valid Lua statement; anything else is assigned to a
			# global throwaway (a `local` here would break `goto`-based continue).
			# Leading `;` avoids the `(` statement/call ambiguity.
			if isinstance(node.expr, CallExpr):
				ctx.add_line(";" + _expr_to_lua(node.expr))
			else:
				ctx.add_line(f"_ = {_expr_to_lua(node.expr)}")

		elif isinstance(node, ReturnNode):
			ctx.add_line(f"state[\"return\"] = {_expr_to_lua(node.expr)}")
			ctx.add_line("do return state end")

		elif isinstance(node, BreakNode):
			ctx.add_line("break")

		elif isinstance(node, ContinueNode):
			if not ctx.loop_labels:
				raise LuaUnsupportedError("continue outside a loop")
			ctx.add_line(f"goto {ctx.loop_labels[-1]}")

		elif isinstance(node, IfNode):
			ctx.add_line(f"if _truthy({_expr_to_lua(node.cond)}) then")
			ctx.indent += 1
			_gen_lua(node.body, ctx)
			ctx.indent -= 1
			for e in node.elifs:
				ctx.add_line(f"elseif _truthy({_expr_to_lua(e.cond)}) then")
				ctx.indent += 1
				_gen_lua(e.body, ctx)
				ctx.indent -= 1
			if node.else_body is not None:
				ctx.add_line("else")
				ctx.indent += 1
				_gen_lua(node.else_body, ctx)
				ctx.indent -= 1
			ctx.add_line("end")

		elif isinstance(node, (ForNode, WhileNode)):
			label = ctx.next_tmp("__cont")
			ctx.loop_labels.append(label)
			if isinstance(node, ForNode):
				item = ctx.next_tmp("__item")
				ctx.add_line(f"for {item} in _seq({_expr_to_lua(node.iterable)}) do")
				ctx.indent += 1
				ctx.add_line(f"state[{_lua_literal(node.var)}] = {item}")
			else:
				ctx.add_line(f"while _truthy({_expr_to_lua(node.cond)}) do")
				ctx.indent += 1
			_gen_lua(node.body, ctx)
			ctx.add_line(f"::{label}::")
			ctx.indent -= 1
			ctx.add_line("end")
			ctx.loop_labels.pop()

		elif isinstance(node, TryNode):
			if _lua_escapes_flow(node.body):
				raise LuaUnsupportedError(
					"try body with break/continue/return that escapes the try "
					"cannot be compiled to Lua (pcall boundary)")
			ok = ctx.next_tmp("__ok")
			err = ctx.next_tmp("__err")
			ctx.add_line(f"local {ok}, {err} = pcall(function()")
			ctx.indent += 1
			_gen_lua(node.body, ctx)
			ctx.indent -= 1
			ctx.add_line("end)")
			handled = ctx.next_tmp("__handled")
			ctx.add_line(f"local {handled} = true")
			if node.excepts:
				ctx.add_line(f"if not {ok} then")
				ctx.indent += 1
				ctx.add_line(f"{handled} = false")
				for i, ex in enumerate(node.excepts):
					names = ex.exc_type if isinstance(ex.exc_type, tuple) else (ex.exc_type,)
					cond = " or ".join(f"_matches({err}, {_lua_literal(t.__name__)})" for t in names)
					keyword = "if" if i == 0 else "elseif"
					ctx.add_line(f"{keyword} {cond} then")
					ctx.indent += 1
					if ex.name is not None:
						ctx.add_line(f"state[{_lua_literal(ex.name)}] = {err}")
					_gen_lua(ex.body, ctx)
					ctx.add_line(f"{handled} = true")
					ctx.indent -= 1
				ctx.add_line("end")
				ctx.indent -= 1
				ctx.add_line("end")
			if node.finally_body is not None:
				_gen_lua(node.finally_body, ctx)
			ctx.add_line(f"if not {ok} and not {handled} then error({err}) end")

		else:
			raise TypeError(f"Unknown Node type in Lua codegen: {node!r}")

	if not nodes:
		ctx.add_line("-- (empty)")




def _generate_lua_function_source(root: BlockNode, fn_name: str, prestate: Dict[str, Any]) -> str:
	ctx = _CodegenContext()
	ctx.add_line(f"local PRESTATE = {_lua_value(prestate)}")
	ctx.add_line("")
	ctx.add_line(f"local function {fn_name}(argstate)")
	ctx.indent += 1
	ctx.add_line("local state = {}")
	ctx.add_line("for k, v in pairs(PRESTATE) do state[k] = v end")
	ctx.add_line("if argstate then for k, v in pairs(argstate) do state[k] = v end end")
	ctx.add_line('state["return"] = nil')
	ctx.add_line('state["error"] = nil')
	ok = ctx.next_tmp("__ok")
	err = ctx.next_tmp("__err")
	ctx.add_line(f"local {ok}, {err} = pcall(function()")
	ctx.indent += 1
	_gen_lua(root.body, ctx)
	ctx.indent -= 1
	ctx.add_line("end)")
	ctx.add_line(f"if not {ok} then state[\"error\"] = {err} end")
	ctx.add_line("return state")
	ctx.indent -= 1
	ctx.add_line("end")
	return "\n".join(ctx.lines)
