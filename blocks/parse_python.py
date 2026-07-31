from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import ast as _ast
import builtins as _builtins
from .nodes import (
	AssignNode, AttrExpr, BinOpExpr, BreakNode, CallExpr, ConcatExpr, CondExpr, ContinueNode, DictExpr, ElifNode, ExceptNode, Expr, ExprNode, ForNode, IfNode, IndexExpr, ListExpr, LiteralExpr, Node, RangeExpr, ReturnNode, SetAttrNode, SetIndexNode, SliceExpr, TryNode, UnaryExpr, VarExpr, WhileNode,
)
from .builder import Block




###############################################################################
# Front-end — parse real Python source into the Blocks AST
#
# Uses the stdlib `ast` module (Python parses itself) and lowers a well-defined
# *subset* of Python into the same AST the builder produces. The result is a
# Block, so every backend (interpret / compile / compile_js / export*) works on
# it unchanged — this is what makes real Python -> JavaScript translation
# possible for the supported subset.
#
# Design rule: anything outside the subset raises UnsupportedSyntaxError with a
# line number. The parser never silently mistranslates.
#
# Supported: module-level statements or a single top-level `def` (its body
# becomes the block; parameter defaults seed the prestate). Statements: assign
# to a simple name, augmented assign, annotated assign, expression, return,
# pass, break, continue, if/elif/else, for (simple-name target), while,
# try/except/finally. Expressions: constants, names, unary (+ - not), binary
# (+ - * / % **), boolean and/or (n-ary), comparisons (incl. chained when the
# middle operands are side-effect-free), calls (positional + keyword), range()
# sugar, attribute access, subscript (index, no slices), list/tuple/dict
# literals.
###############################################################################

class UnsupportedSyntaxError(Exception):
	"""A Python construct outside the subset that parse_python understands."""

	def __init__(self, node: Any, detail: str = ""):
		loc = ""
		if isinstance(node, _ast.AST) and hasattr(node, "lineno"):
			loc = f" (line {node.lineno})"
		kind = type(node).__name__ if isinstance(node, _ast.AST) else str(node)
		msg = f"Unsupported Python construct: {kind}{loc}"
		if detail:
			msg += f" — {detail}"
		super().__init__(msg)
		self.node = node




_AST_BINOP = {
	_ast.Add: "add", _ast.Sub: "sub", _ast.Mult: "mul", _ast.Div: "div",
	_ast.Mod: "mod", _ast.Pow: "pow",
}


_AST_CMP = {
	_ast.Lt: "lt", _ast.LtE: "le", _ast.Gt: "gt", _ast.GtE: "ge",
	_ast.Eq: "eq", _ast.NotEq: "ne",
}


_AST_UNARY = {
	_ast.USub: "neg", _ast.UAdd: "pos", _ast.Not: "not",
}




def _lookup_exc(name: str, node: _ast.AST) -> type:
	obj = getattr(_builtins, name, None)
	if isinstance(obj, type) and issubclass(obj, BaseException):
		return obj
	raise UnsupportedSyntaxError(node, f"unknown or non-builtin exception type: {name}")




def _resolve_exc_type(t: Optional[_ast.expr]) -> Union[type, Tuple[type, ...]]:
	if t is None:  # bare `except:` catches Exception (control-flow uses BaseException)
		return Exception
	if isinstance(t, _ast.Name):
		return _lookup_exc(t.id, t)
	if isinstance(t, _ast.Tuple):
		types = []
		for elt in t.elts:
			if not isinstance(elt, _ast.Name):
				raise UnsupportedSyntaxError(elt, "exception type must be a name")
			types.append(_lookup_exc(elt.id, elt))
		return tuple(types)
	raise UnsupportedSyntaxError(t, "exception type must be a name or tuple of names")




def _is_side_effect_free(node: _ast.expr) -> bool:
	"""Conservative: names, constants, and attribute/index chains over them."""
	if isinstance(node, (_ast.Name, _ast.Constant)):
		return True
	if isinstance(node, _ast.Attribute):
		return _is_side_effect_free(node.value)
	if isinstance(node, _ast.Subscript) and not isinstance(node.slice, _ast.Slice):
		return _is_side_effect_free(node.value) and _is_side_effect_free(node.slice)
	return False




def _lower_expr(node: _ast.expr) -> Expr:
	if isinstance(node, _ast.Constant):
		return LiteralExpr(node.value)

	if isinstance(node, _ast.Name):
		return VarExpr(node.id)

	if isinstance(node, _ast.UnaryOp):
		op = _AST_UNARY.get(type(node.op))
		if op is None:
			raise UnsupportedSyntaxError(node.op)
		return UnaryExpr(op, _lower_expr(node.operand))

	if isinstance(node, _ast.BinOp):
		op = _AST_BINOP.get(type(node.op))
		if op is None:
			raise UnsupportedSyntaxError(node.op, "arithmetic operator not supported")
		return BinOpExpr(op, _lower_expr(node.left), _lower_expr(node.right))

	if isinstance(node, _ast.BoolOp):
		op = "and" if isinstance(node.op, _ast.And) else "or"
		vals = [_lower_expr(v) for v in node.values]
		acc = vals[0]
		for v in vals[1:]:
			acc = BinOpExpr(op, acc, v)
		return acc

	if isinstance(node, _ast.Compare):
		# a OP b -> BinOpExpr; chained a < b < c -> (a<b) and (b<c). The middle
		# operand is evaluated twice, so require it to be side-effect-free.
		for mid in node.comparators[:-1]:
			if not _is_side_effect_free(mid):
				raise UnsupportedSyntaxError(
					node, "chained comparison with a side-effecting middle operand")
		operands = [node.left, *node.comparators]
		parts = []
		for i, op_ast in enumerate(node.ops):
			op = _AST_CMP.get(type(op_ast))
			if op is None:
				raise UnsupportedSyntaxError(op_ast)
			parts.append(BinOpExpr(op, _lower_expr(operands[i]), _lower_expr(operands[i + 1])))
		acc = parts[0]
		for p in parts[1:]:
			acc = BinOpExpr("and", acc, p)
		return acc

	if isinstance(node, _ast.Call):
		# range(...) is sugar for RangeExpr so it works across all backends
		if isinstance(node.func, _ast.Name) and node.func.id == "range":
			if node.keywords:
				raise UnsupportedSyntaxError(node, "range() with keyword arguments")
			args = [_lower_expr(a) for a in node.args]
			if len(args) == 1:
				return RangeExpr(LiteralExpr(0), args[0])
			if len(args) == 2:
				return RangeExpr(args[0], args[1])
			if len(args) == 3:
				return RangeExpr(args[0], args[1], args[2])
			raise UnsupportedSyntaxError(node, "range() takes 1-3 arguments")
		if any(isinstance(a, _ast.Starred) for a in node.args):
			raise UnsupportedSyntaxError(node, "*args are not supported")
		func = _lower_expr(node.func)
		args = [_lower_expr(a) for a in node.args]
		kwargs: Dict[str, Expr] = {}
		for kw in node.keywords:
			if kw.arg is None:
				raise UnsupportedSyntaxError(node, "**kwargs are not supported")
			kwargs[kw.arg] = _lower_expr(kw.value)
		return CallExpr(func, args, kwargs)

	if isinstance(node, _ast.Attribute):
		return AttrExpr(_lower_expr(node.value), node.attr)

	if isinstance(node, _ast.Subscript):
		if isinstance(node.slice, _ast.Slice):
			sl = node.slice
			return IndexExpr(_lower_expr(node.value), SliceExpr(
				_lower_expr(sl.lower) if sl.lower is not None else None,
				_lower_expr(sl.upper) if sl.upper is not None else None,
				_lower_expr(sl.step) if sl.step is not None else None))
		return IndexExpr(_lower_expr(node.value), _lower_expr(node.slice))

	if isinstance(node, (_ast.List, _ast.Tuple)):
		return ListExpr([_lower_expr(e) for e in node.elts])

	if isinstance(node, _ast.Dict):
		if any(k is None for k in node.keys):
			raise UnsupportedSyntaxError(node, "dict unpacking (**) is not supported")
		return DictExpr([_lower_expr(k) for k in node.keys],
						[_lower_expr(v) for v in node.values])

	if isinstance(node, _ast.IfExp):
		return CondExpr(_lower_expr(node.test), _lower_expr(node.body),
						_lower_expr(node.orelse))

	if isinstance(node, _ast.JoinedStr):  # f-string
		parts: List[Expr] = []
		for v in node.values:
			if isinstance(v, _ast.Constant):
				parts.append(LiteralExpr(v.value))
			elif isinstance(v, _ast.FormattedValue):
				if v.conversion not in (-1, ord("s")) or v.format_spec is not None:
					raise UnsupportedSyntaxError(
						v, "f-string conversions/format specs are not supported")
				parts.append(_lower_expr(v.value))
			else:
				raise UnsupportedSyntaxError(v, "unsupported f-string segment")
		return ConcatExpr(parts)

	raise UnsupportedSyntaxError(node)




def _lower_if(s: _ast.If) -> IfNode:
	node = IfNode(cond=_lower_expr(s.test), body=_lower_stmts(s.body))
	orelse = s.orelse
	while len(orelse) == 1 and isinstance(orelse[0], _ast.If):
		e = orelse[0]
		node.elifs.append(ElifNode(cond=_lower_expr(e.test), body=_lower_stmts(e.body)))
		orelse = e.orelse
	if orelse:
		node.else_body = _lower_stmts(orelse)
	return node




def _lower_try(s: _ast.Try) -> TryNode:
	if s.orelse:
		raise UnsupportedSyntaxError(s, "try-else is not supported")
	node = TryNode(body=_lower_stmts(s.body))
	for h in s.handlers:
		node.excepts.append(ExceptNode(exc_type=_resolve_exc_type(h.type),
									   body=_lower_stmts(h.body), name=h.name))
	if s.finalbody:
		node.finally_body = _lower_stmts(s.finalbody)
	return node




def _lower_assign_target(tgt: _ast.expr, value: Expr) -> Node:
	"""Build the assignment node for a target (name, subscript, or attribute)."""
	if isinstance(tgt, _ast.Name):
		return AssignNode(tgt.id, value)
	if isinstance(tgt, _ast.Subscript):
		if isinstance(tgt.slice, _ast.Slice):
			raise UnsupportedSyntaxError(tgt, "slice assignment is not supported")
		return SetIndexNode(_lower_expr(tgt.value), _lower_expr(tgt.slice), value)
	if isinstance(tgt, _ast.Attribute):
		return SetAttrNode(_lower_expr(tgt.value), tgt.attr, value)
	raise UnsupportedSyntaxError(tgt, "unsupported assignment target (no tuple unpacking)")




def _lower_stmt(s: _ast.stmt) -> List[Node]:
	if isinstance(s, _ast.Assign):
		if len(s.targets) != 1:
			raise UnsupportedSyntaxError(s, "chained assignment (a = b = ...) is not supported")
		return [_lower_assign_target(s.targets[0], _lower_expr(s.value))]

	if isinstance(s, _ast.AugAssign):
		op = _AST_BINOP.get(type(s.op))
		if op is None:
			raise UnsupportedSyntaxError(s.op)
		val = _lower_expr(s.value)
		tgt = s.target
		if isinstance(tgt, _ast.Name):
			return [AssignNode(tgt.id, BinOpExpr(op, VarExpr(tgt.id), val))]
		# x[i] += v / x.a += v : the target is read and written, so it must be
		# side-effect-free (re-evaluated once each way).
		if isinstance(tgt, _ast.Subscript):
			if isinstance(tgt.slice, _ast.Slice):
				raise UnsupportedSyntaxError(tgt, "slice assignment is not supported")
			if not (_is_side_effect_free(tgt.value) and _is_side_effect_free(tgt.slice)):
				raise UnsupportedSyntaxError(tgt, "augmented subscript assignment requires a side-effect-free target")
			read = IndexExpr(_lower_expr(tgt.value), _lower_expr(tgt.slice))
			return [SetIndexNode(_lower_expr(tgt.value), _lower_expr(tgt.slice),
								 BinOpExpr(op, read, val))]
		if isinstance(tgt, _ast.Attribute):
			if not _is_side_effect_free(tgt.value):
				raise UnsupportedSyntaxError(tgt, "augmented attribute assignment requires a side-effect-free target")
			read = AttrExpr(_lower_expr(tgt.value), tgt.attr)
			return [SetAttrNode(_lower_expr(tgt.value), tgt.attr, BinOpExpr(op, read, val))]
		raise UnsupportedSyntaxError(tgt, "unsupported augmented assignment target")

	if isinstance(s, _ast.AnnAssign):
		if s.value is None:
			return []  # a bare annotation (x: int) is a no-op
		if not isinstance(s.target, _ast.Name):
			raise UnsupportedSyntaxError(s, "annotated assignment target must be a simple name")
		return [AssignNode(s.target.id, _lower_expr(s.value))]

	if isinstance(s, _ast.Expr):
		return [ExprNode(_lower_expr(s.value))]

	if isinstance(s, _ast.Return):
		value = _lower_expr(s.value) if s.value is not None else LiteralExpr(None)
		return [ReturnNode(value)]

	if isinstance(s, _ast.Pass):
		return []

	if isinstance(s, _ast.Break):
		return [BreakNode()]

	if isinstance(s, _ast.Continue):
		return [ContinueNode()]

	if isinstance(s, _ast.If):
		return [_lower_if(s)]

	if isinstance(s, _ast.For):
		if s.orelse:
			raise UnsupportedSyntaxError(s, "for-else is not supported")
		if not isinstance(s.target, _ast.Name):
			raise UnsupportedSyntaxError(s, "for-loop target must be a simple name")
		return [ForNode(s.target.id, _lower_expr(s.iter), _lower_stmts(s.body))]

	if isinstance(s, _ast.While):
		if s.orelse:
			raise UnsupportedSyntaxError(s, "while-else is not supported")
		return [WhileNode(_lower_expr(s.test), _lower_stmts(s.body))]

	if isinstance(s, _ast.Try):
		return [_lower_try(s)]

	raise UnsupportedSyntaxError(s)




def _lower_stmts(stmts: List[_ast.stmt]) -> List[Node]:
	out: List[Node] = []
	for s in stmts:
		out.extend(_lower_stmt(s))
	return out




def _defaults_to_prestate(fn: _ast.FunctionDef) -> Dict[str, Any]:
	"""Map a def's default argument values into prestate entries."""
	prestate: Dict[str, Any] = {}
	args = fn.args
	positional = args.posonlyargs + args.args
	for arg, default in zip(positional[len(positional) - len(args.defaults):], args.defaults):
		expr = _lower_expr(default)
		if not isinstance(expr, LiteralExpr):
			raise UnsupportedSyntaxError(default, "default argument values must be constants")
		prestate[arg.arg] = expr.value
	for arg, default in zip(args.kwonlyargs, args.kw_defaults):
		if default is None:
			continue
		expr = _lower_expr(default)
		if not isinstance(expr, LiteralExpr):
			raise UnsupportedSyntaxError(default, "default argument values must be constants")
		prestate[arg.arg] = expr.value
	return prestate




def parse_python(source: str, prestate: Optional[Dict[str, Any]] = None) -> Block:
	"""Parse a subset of real Python source into a Block.

	The source may be a sequence of module-level statements, or a single
	top-level ``def`` (its body becomes the block and constant parameter
	defaults seed the prestate). Explicit `prestate` entries override defaults.

	Raises UnsupportedSyntaxError (with a line number) on any construct outside
	the supported subset — it never silently mistranslates.
	"""
	tree = _ast.parse(source, mode="exec")

	seed: Dict[str, Any] = {}
	if len(tree.body) == 1 and isinstance(tree.body[0], _ast.FunctionDef):
		fn = tree.body[0]
		seed = _defaults_to_prestate(fn)
		body_ast = fn.body
	else:
		body_ast = tree.body

	if prestate:
		seed.update(prestate)

	block = Block(seed)
	block._root.body.extend(_lower_stmts(body_ast))
	return block
