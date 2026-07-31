from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union




###############################################################################
# Expression AST
###############################################################################

class Expr:
	"""Base class for expressions.

	Operator overloading lives here so *every* expression composes uniformly:
	``(b.x + 1) * 2``, ``b.items[0]``, ``b.obj.attr("field")`` all build AST.

	Note on ``==``/``!=``: like most expression-builder DSLs (e.g. SQLAlchemy),
	these are overloaded to build comparison nodes instead of returning a bool.
	That means Exprs are *not* value-comparable and ``expr in some_set`` will not
	behave like normal membership. Objects remain hashable by identity
	(``__hash__`` below), so they can still be used as dict keys / set members
	keyed on object identity.
	"""
	__slots__ = ()

	def _as_expr(self) -> "Expr":
		return self

	# arithmetic & comparison operators build BinOpExpr
	def __add__(self, other): return BinOpExpr("add", self, to_expr(other))
	def __sub__(self, other): return BinOpExpr("sub", self, to_expr(other))
	def __mul__(self, other): return BinOpExpr("mul", self, to_expr(other))
	def __truediv__(self, other): return BinOpExpr("div", self, to_expr(other))
	def __mod__(self, other): return BinOpExpr("mod", self, to_expr(other))
	def __pow__(self, other): return BinOpExpr("pow", self, to_expr(other))

	def __radd__(self, other): return BinOpExpr("add", to_expr(other), self)
	def __rsub__(self, other): return BinOpExpr("sub", to_expr(other), self)
	def __rmul__(self, other): return BinOpExpr("mul", to_expr(other), self)
	def __rtruediv__(self, other): return BinOpExpr("div", to_expr(other), self)
	def __rmod__(self, other): return BinOpExpr("mod", to_expr(other), self)
	def __rpow__(self, other): return BinOpExpr("pow", to_expr(other), self)

	def __lt__(self, other): return BinOpExpr("lt", self, to_expr(other))
	def __le__(self, other): return BinOpExpr("le", self, to_expr(other))
	def __gt__(self, other): return BinOpExpr("gt", self, to_expr(other))
	def __ge__(self, other): return BinOpExpr("ge", self, to_expr(other))
	def __eq__(self, other): return BinOpExpr("eq", self, to_expr(other))
	def __ne__(self, other): return BinOpExpr("ne", self, to_expr(other))

	def __neg__(self): return UnaryExpr("neg", self)
	def __pos__(self): return UnaryExpr("pos", self)

	# comparison overloading removes the default __hash__; restore identity hash
	__hash__ = object.__hash__

	# and/or cannot be operator-overloaded in Python; expose as methods.
	def and_(self, other) -> "BinOpExpr": return BinOpExpr("and", self, to_expr(other))
	def or_(self, other) -> "BinOpExpr": return BinOpExpr("or", self, to_expr(other))
	def not_(self) -> "UnaryExpr": return UnaryExpr("not", self)

	# subscript -> IndexExpr
	def __getitem__(self, key) -> "IndexExpr":
		return IndexExpr(self, to_expr(key))

	# attribute access on a *value* -> AttrExpr (explicit, to avoid clashing
	# with dataclass fields like VarExpr.name).
	def attr(self, name: str) -> "AttrExpr":
		return AttrExpr(self, name)

	# method / function call -> CallExpr
	def call(self, *args, **kwargs) -> "CallExpr":
		return CallExpr(self,
						[to_expr(a) for a in args],
						{k: to_expr(v) for k, v in kwargs.items()})




@dataclass(slots=True, eq=False)
class LiteralExpr(Expr):
	value: Any




@dataclass(slots=True, eq=False)
class VarExpr(Expr):
	name: str




@dataclass(slots=True, eq=False)
class UnaryExpr(Expr):
	op: str  # "neg", "pos", "not"
	operand: Expr




@dataclass(slots=True, eq=False)
class BinOpExpr(Expr):
	op: str  # "add", "sub", "mul", "div", "and", "or", "lt", ...
	left: Expr
	right: Expr




@dataclass(slots=True, eq=False)
class CallExpr(Expr):
	func: Expr
	args: List[Expr] = field(default_factory=list)
	kwargs: Dict[str, Expr] = field(default_factory=dict)




@dataclass(slots=True, eq=False)
class RangeExpr(Expr):
	start: Expr
	stop: Expr
	step: Optional[Expr] = None




@dataclass(slots=True, eq=False)
class IndexExpr(Expr):
	"""Subscript: obj[index]."""
	obj: Expr
	index: Expr




@dataclass(slots=True, eq=False)
class AttrExpr(Expr):
	"""Attribute access on a value: obj.name."""
	obj: Expr
	name: str




@dataclass(slots=True, eq=False)
class ListExpr(Expr):
	"""List literal whose elements may themselves be expressions."""
	elts: List[Expr] = field(default_factory=list)




@dataclass(slots=True, eq=False)
class DictExpr(Expr):
	"""Dict literal whose keys/values may themselves be expressions."""
	keys: List[Expr] = field(default_factory=list)
	values: List[Expr] = field(default_factory=list)




@dataclass(slots=True, eq=False)
class CondExpr(Expr):
	"""Conditional expression: then if cond else orelse (JS: cond ? then : orelse).

	Only the taken branch is evaluated (short-circuit).
	"""
	cond: Expr
	then: Expr
	orelse: Expr




@dataclass(slots=True, eq=False)
class SliceExpr(Expr):
	"""A slice used as a subscript index: obj[lower:upper:step]."""
	lower: Optional[Expr] = None
	upper: Optional[Expr] = None
	step: Optional[Expr] = None




@dataclass(slots=True, eq=False)
class ConcatExpr(Expr):
	"""String concatenation of stringified parts (f-strings, template literals)."""
	parts: List[Expr] = field(default_factory=list)




###############################################################################
# Statement AST
###############################################################################

@dataclass(slots=True)
class Node:
	"""Base class for statements."""
	pass




@dataclass(slots=True)
class AssignNode(Node):
	name: str
	expr: Expr




@dataclass(slots=True)
class SetIndexNode(Node):
	"""Subscript assignment: obj[index] = value."""
	obj: Expr
	index: Expr
	value: Expr




@dataclass(slots=True)
class SetAttrNode(Node):
	"""Attribute assignment: obj.name = value."""
	obj: Expr
	name: str
	value: Expr




@dataclass(slots=True)
class ExprNode(Node):
	expr: Expr




@dataclass(slots=True)
class ReturnNode(Node):
	expr: Expr




@dataclass(slots=True)
class BreakNode(Node):
	"""Breaks the innermost loop."""
	pass




@dataclass(slots=True)
class ContinueNode(Node):
	"""Continues the innermost loop."""
	pass




@dataclass(slots=True)
class IfNode(Node):
	cond: Expr
	body: List[Node] = field(default_factory=list)
	elifs: List["ElifNode"] = field(default_factory=list)
	else_body: Optional[List[Node]] = None




@dataclass(slots=True)
class ElifNode(Node):
	cond: Expr
	body: List[Node] = field(default_factory=list)




@dataclass(slots=True)
class ForNode(Node):
	var: str
	iterable: Expr
	body: List[Node] = field(default_factory=list)




@dataclass(slots=True)
class WhileNode(Node):
	cond: Expr
	body: List[Node] = field(default_factory=list)




@dataclass(slots=True)
class ExceptNode(Node):
	exc_type: Union[type, Tuple[type, ...]]
	body: List[Node] = field(default_factory=list)
	name: Optional[str] = None  # `except T as name:` binds the caught value




@dataclass(slots=True)
class TryNode(Node):
	body: List[Node] = field(default_factory=list)
	excepts: List[ExceptNode] = field(default_factory=list)
	finally_body: Optional[List[Node]] = None




@dataclass(slots=True)
class BlockNode(Node):
	body: List[Node] = field(default_factory=list)




###############################################################################
# Builder API (Block & symbolic variables)
###############################################################################

def to_expr(value: Any) -> Expr:
	"""Lift a Python value into an Expr.

	- Expr passes through unchanged.
	- list/tuple/dict are converted structurally, so elements/values may
	  themselves be expressions (e.g. ``[b.x, 1]`` or ``{"k": b.y}``).
	- anything else becomes a LiteralExpr.
	"""
	if isinstance(value, Expr):
		return value
	if isinstance(value, (list, tuple)):
		return ListExpr([to_expr(v) for v in value])
	if isinstance(value, dict):
		return DictExpr([to_expr(k) for k in value.keys()],
						[to_expr(v) for v in value.values()])
	return LiteralExpr(value)
