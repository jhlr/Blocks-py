from __future__ import annotations
import ast as _ast
import builtins as _builtins
import json
import operator
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

###############################################################################
# Operator tables — single source of truth for semantics
#
# Both the interpreter (eval_expr) and the code generator (_expr_to_source)
# read from these tables, so an operator's meaning is defined in exactly one
# place. Adding an operator means adding one row here.
###############################################################################

# Unary operators: op -> (python prefix, runtime function, javascript prefix).
UNARY_OPS: Dict[str, Tuple[str, Callable[[Any], Any], str]] = {
	"neg": ("-", operator.neg, "-"),
	"pos": ("+", operator.pos, "+"),
	"not": ("not ", operator.not_, "!"),
}

# Eager binary operators: op -> (python symbol, runtime function, js symbol).
# Both operands are always evaluated.
BINARY_OPS: Dict[str, Tuple[str, Callable[[Any, Any], Any], str]] = {
	"add": ("+", operator.add, "+"),
	"sub": ("-", operator.sub, "-"),
	"mul": ("*", operator.mul, "*"),
	"div": ("/", operator.truediv, "/"),
	"mod": ("%", operator.mod, "%"),
	"pow": ("**", operator.pow, "**"),
	"lt": ("<", operator.lt, "<"),
	"le": ("<=", operator.le, "<="),
	"gt": (">", operator.gt, ">"),
	"ge": (">=", operator.ge, ">="),
	"eq": ("==", operator.eq, "==="),
	"ne": ("!=", operator.ne, "!=="),
}

# Short-circuit binary operators: op -> (python symbol, js symbol). These
# cannot be plain functions (both sides would be evaluated), so the interpreter
# handles their laziness explicitly and generated code relies on the target
# language's own and/or.
SHORTCIRCUIT_OPS: Dict[str, Tuple[str, str]] = {
	"and": ("and", "&&"),
	"or": ("or", "||"),
}


###############################################################################
# Exceptions for internal control flow
###############################################################################

class BlockReturn(BaseException):
	"""Signal a Block-level return (top-level exit)."""
	pass


class LoopBreak(BaseException):
	"""Signal a break in the current loop."""
	pass


class LoopContinue(BaseException):
	"""Signal a continue in the current loop."""
	pass


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


@dataclass(slots=True)
class TryNode(Node):
	body: List[Node] = field(default_factory=list)
	excepts: List[ExceptNode] = field(default_factory=list)
	finally_body: Optional[List[Node]] = None


@dataclass(slots=True)
class BlockNode(Node):
	body: List[Node] = field(default_factory=list)


###############################################################################
# Expression evaluation (interpreter)
###############################################################################

def eval_expr(expr: Expr, state: Dict[str, Any]) -> Any:
	"""Evaluate an Expr against the given state.

	Missing keys in state are treated as None.
	Any Python exception may be raised and will be handled by the DSL's
	try/except or captured into state["error"] at top level.
	"""
	if isinstance(expr, LiteralExpr):
		return expr.value

	if isinstance(expr, VarExpr):
		return state.get(expr.name, None)

	if isinstance(expr, UnaryExpr):
		unary = UNARY_OPS.get(expr.op)
		if unary is None:
			raise ValueError(f"Unknown unary op: {expr.op}")
		return unary[1](eval_expr(expr.operand, state))

	if isinstance(expr, BinOpExpr):
		# Short-circuit for and/or: evaluate the right side lazily.
		if expr.op == "and":
			left_val = eval_expr(expr.left, state)
			return eval_expr(expr.right, state) if left_val else left_val
		if expr.op == "or":
			left_val = eval_expr(expr.left, state)
			return left_val if left_val else eval_expr(expr.right, state)

		binary = BINARY_OPS.get(expr.op)
		if binary is None:
			raise ValueError(f"Unknown binary op: {expr.op}")
		left_val = eval_expr(expr.left, state)
		right_val = eval_expr(expr.right, state)
		return binary[1](left_val, right_val)

	if isinstance(expr, CallExpr):
		func_val = eval_expr(expr.func, state)
		args_val = [eval_expr(a, state) for a in expr.args]
		kwargs_val = {k: eval_expr(v, state) for k, v in expr.kwargs.items()}
		return func_val(*args_val, **kwargs_val)

	if isinstance(expr, RangeExpr):
		start = eval_expr(expr.start, state)
		stop = eval_expr(expr.stop, state)
		if expr.step is None:
			return range(start, stop)
		step = eval_expr(expr.step, state)
		return range(start, stop, step)

	if isinstance(expr, IndexExpr):
		return eval_expr(expr.obj, state)[eval_expr(expr.index, state)]

	if isinstance(expr, AttrExpr):
		return getattr(eval_expr(expr.obj, state), expr.name)

	if isinstance(expr, ListExpr):
		return [eval_expr(e, state) for e in expr.elts]

	if isinstance(expr, DictExpr):
		return {eval_expr(k, state): eval_expr(v, state)
				for k, v in zip(expr.keys, expr.values)}

	raise TypeError(f"Unknown Expr type: {expr!r}")


###############################################################################
# Statement execution (interpreter)
###############################################################################

def exec_nodes(nodes: List[Node], state: Dict[str, Any]) -> None:
	"""Execute a list of nodes in the given state.

	Control-flow is implemented using internal exceptions:
	- BlockReturn causes immediate exit from the whole Block()
	- LoopBreak/LoopContinue apply to innermost loop
	"""
	i = 0
	while i < len(nodes):
		node = nodes[i]

		if isinstance(node, AssignNode):
			value = eval_expr(node.expr, state)
			state[node.name] = value

		elif isinstance(node, SetIndexNode):
			obj = eval_expr(node.obj, state)
			obj[eval_expr(node.index, state)] = eval_expr(node.value, state)

		elif isinstance(node, SetAttrNode):
			setattr(eval_expr(node.obj, state), node.name, eval_expr(node.value, state))

		elif isinstance(node, ExprNode):
			eval_expr(node.expr, state)

		elif isinstance(node, ReturnNode):
			value = eval_expr(node.expr, state)
			state["return"] = value
			# stop executing any further nodes in this block
			raise BlockReturn()

		elif isinstance(node, BreakNode):
			raise LoopBreak()

		elif isinstance(node, ContinueNode):
			raise LoopContinue()

		elif isinstance(node, IfNode):
			cond_val = eval_expr(node.cond, state)
			if cond_val:
				exec_nodes(node.body, state)
			else:
				handled = False
				for e in node.elifs:
					if eval_expr(e.cond, state):
						exec_nodes(e.body, state)
						handled = True
						break
				if not handled and node.else_body is not None:
					exec_nodes(node.else_body, state)

		elif isinstance(node, ForNode):
			iterable = eval_expr(node.iterable, state)
			try:
				for item in iterable:
					state[node.var] = item
					try:
						exec_nodes(node.body, state)
					except LoopContinue:
						continue
			except LoopBreak:
				# break out of this loop only
				pass

		elif isinstance(node, WhileNode):
			try:
				while eval_expr(node.cond, state):
					try:
						exec_nodes(node.body, state)
					except LoopContinue:
						continue
			except LoopBreak:
				# break out of this while only
				pass

		elif isinstance(node, TryNode):
			try:
				exec_nodes(node.body, state)
			except BlockReturn:
				# Return must still run finally
				if node.finally_body is not None:
					exec_nodes(node.finally_body, state)
				# re-raise BlockReturn to higher level
				raise
			except LoopBreak:
				# Loop control must still run finally
				if node.finally_body is not None:
					exec_nodes(node.finally_body, state)
				raise
			except LoopContinue:
				if node.finally_body is not None:
					exec_nodes(node.finally_body, state)
				raise
			except Exception as e:
				handled = False
				for ex in node.excepts:
					if isinstance(e, ex.exc_type):
						exec_nodes(ex.body, state)
						handled = True
						break
				if node.finally_body is not None:
					exec_nodes(node.finally_body, state)
				if not handled:
					# propagate upwards to possibly be captured in state["error"]
					raise
			else:
				# no exception
				if node.finally_body is not None:
					exec_nodes(node.finally_body, state)

		else:
			raise TypeError(f"Unknown Node type: {node!r}")

		i += 1


###############################################################################
# Code generation (compile to Python source)
###############################################################################

@dataclass
class _CodegenContext:
	lines: List[str] = field(default_factory=list)
	indent: int = 0
	tmp_counter: int = 0
	exc_types: List[Union[type, Tuple[type, ...]]] = field(default_factory=list)

	def add_line(self, text: str) -> None:
		self.lines.append("    " * self.indent + text)

	def next_tmp(self, prefix: str = "__tmp") -> str:
		name = f"{prefix}{self.tmp_counter}"
		self.tmp_counter += 1
		return name

	def register_exc_type(self, t: Union[type, Tuple[type, ...]]) -> None:
		if t not in self.exc_types:
			self.exc_types.append(t)


def _expr_to_source(expr: Expr) -> str:
	if isinstance(expr, LiteralExpr):
		return repr(expr.value)
	if isinstance(expr, VarExpr):
		return f"state.get({expr.name!r}, None)"
	if isinstance(expr, UnaryExpr):
		unary = UNARY_OPS.get(expr.op)
		if unary is None:
			raise ValueError(f"Unknown unary op: {expr.op}")
		return f"({unary[0]}{_expr_to_source(expr.operand)})"
	if isinstance(expr, BinOpExpr):
		if expr.op in BINARY_OPS:
			symbol = BINARY_OPS[expr.op][0]
		elif expr.op in SHORTCIRCUIT_OPS:
			symbol = SHORTCIRCUIT_OPS[expr.op][0]
		else:
			raise ValueError(f"Unknown binary op: {expr.op}")
		left = _expr_to_source(expr.left)
		right = _expr_to_source(expr.right)
		return f"({left} {symbol} {right})"
	if isinstance(expr, IndexExpr):
		return f"({_expr_to_source(expr.obj)})[{_expr_to_source(expr.index)}]"
	if isinstance(expr, AttrExpr):
		obj_src = _expr_to_source(expr.obj)
		if expr.name.isidentifier():
			return f"({obj_src}).{expr.name}"
		return f"getattr({obj_src}, {expr.name!r})"
	if isinstance(expr, ListExpr):
		return "[" + ", ".join(_expr_to_source(e) for e in expr.elts) + "]"
	if isinstance(expr, DictExpr):
		items = ", ".join(f"{_expr_to_source(k)}: {_expr_to_source(v)}"
						  for k, v in zip(expr.keys, expr.values))
		return "{" + items + "}"
	if isinstance(expr, CallExpr):
		func_src = _expr_to_source(expr.func)
		args_src = ", ".join(_expr_to_source(a) for a in expr.args)
		kwargs_src = ", ".join(f"{k}={_expr_to_source(v)}" for k, v in expr.kwargs.items())
		if args_src and kwargs_src:
			inner = f"{args_src}, {kwargs_src}"
		else:
			inner = args_src or kwargs_src
		return f"({func_src})({inner})"
	if isinstance(expr, RangeExpr):
		start = _expr_to_source(expr.start)
		stop = _expr_to_source(expr.stop)
		if expr.step is None:
			return f"range({start}, {stop})"
		step = _expr_to_source(expr.step)
		return f"range({start}, {stop}, {step})"
	raise TypeError(f"Unknown Expr type in codegen: {expr!r}")


def _exc_type_to_source(t: Union[type, Tuple[type, ...]]) -> str:
	"""Return Python source for an exception type or tuple of types.

	For compile() we bind the actual classes in the namespace, so we only
	need simple names here (ZeroDivisionError, MyError, etc.). For export(),
	we will generate imports separately.
	"""
	if isinstance(t, tuple):
		inner = ", ".join(_exc_type_to_source(x) for x in t)
		return f"({inner})"
	# Single type
	return t.__name__


def _gen_nodes(nodes: List[Node], ctx: _CodegenContext) -> None:
	# Every call site is a block body; an empty one must still be syntactically
	# valid Python (e.g. `if x: pass`, `finally: pass`).
	if not nodes:
		ctx.add_line("pass")
		return
	for node in nodes:
		if isinstance(node, AssignNode):
			expr_src = _expr_to_source(node.expr)
			ctx.add_line(f"state[{node.name!r}] = {expr_src}")

		elif isinstance(node, SetIndexNode):
			ctx.add_line(f"({_expr_to_source(node.obj)})[{_expr_to_source(node.index)}]"
						 f" = {_expr_to_source(node.value)}")

		elif isinstance(node, SetAttrNode):
			obj_src = _expr_to_source(node.obj)
			val_src = _expr_to_source(node.value)
			if node.name.isidentifier():
				ctx.add_line(f"({obj_src}).{node.name} = {val_src}")
			else:
				ctx.add_line(f"setattr({obj_src}, {node.name!r}, {val_src})")

		elif isinstance(node, ExprNode):
			expr_src = _expr_to_source(node.expr)
			ctx.add_line(expr_src)

		elif isinstance(node, ReturnNode):
			expr_src = _expr_to_source(node.expr)
			ctx.add_line(f"state['return'] = {expr_src}")
			ctx.add_line("raise BlockReturn()")

		elif isinstance(node, BreakNode):
			ctx.add_line("raise LoopBreak()")

		elif isinstance(node, ContinueNode):
			ctx.add_line("raise LoopContinue()")

		elif isinstance(node, IfNode):
			cond_src = _expr_to_source(node.cond)
			ctx.add_line(f"if {cond_src}:")
			ctx.indent += 1
			_gen_nodes(node.body, ctx)
			ctx.indent -= 1
			for i, e in enumerate(node.elifs):
				cond_e = _expr_to_source(e.cond)
				keyword = "elif"
				ctx.add_line(f"{keyword} {cond_e}:")
				ctx.indent += 1
				_gen_nodes(e.body, ctx)
				ctx.indent -= 1
			if node.else_body is not None:
				ctx.add_line("else:")
				ctx.indent += 1
				_gen_nodes(node.else_body, ctx)
				ctx.indent -= 1

		elif isinstance(node, ForNode):
			iterable_src = _expr_to_source(node.iterable)
			item_var = ctx.next_tmp("__item")
			ctx.add_line("try:")
			ctx.indent += 1
			ctx.add_line(f"for {item_var} in {iterable_src}:")
			ctx.indent += 1
			ctx.add_line(f"state[{node.var!r}] = {item_var}")
			ctx.add_line("try:")
			ctx.indent += 1
			_gen_nodes(node.body, ctx)
			ctx.indent -= 1
			ctx.add_line("except LoopContinue:")
			ctx.indent += 1
			ctx.add_line("continue")
			ctx.indent -= 2  # end for, end inner try
			ctx.indent -= 1
			ctx.add_line("except LoopBreak:")
			ctx.indent += 1
			ctx.add_line("pass")
			ctx.indent -= 1

		elif isinstance(node, WhileNode):
			cond_src = _expr_to_source(node.cond)
			ctx.add_line("try:")
			ctx.indent += 1
			ctx.add_line(f"while {cond_src}:")
			ctx.indent += 1
			ctx.add_line("try:")
			ctx.indent += 1
			_gen_nodes(node.body, ctx)
			ctx.indent -= 1
			ctx.add_line("except LoopContinue:")
			ctx.indent += 1
			ctx.add_line("continue")
			ctx.indent -= 2  # end while body, end inner try
			ctx.indent -= 1
			ctx.add_line("except LoopBreak:")
			ctx.indent += 1
			ctx.add_line("pass")
			ctx.indent -= 1

		elif isinstance(node, TryNode):
			ctx.add_line("try:")
			ctx.indent += 1
			_gen_nodes(node.body, ctx)
			ctx.indent -= 1

			# BlockReturn, LoopBreak, LoopContinue: run finally then re-raise
			if node.finally_body is not None:
				# BlockReturn
				ctx.add_line("except BlockReturn:")
				ctx.indent += 1
				_gen_nodes(node.finally_body, ctx)
				ctx.add_line("raise")
				ctx.indent -= 1

				# LoopBreak
				ctx.add_line("except LoopBreak:")
				ctx.indent += 1
				_gen_nodes(node.finally_body, ctx)
				ctx.add_line("raise")
				ctx.indent -= 1

				# LoopContinue
				ctx.add_line("except LoopContinue:")
				ctx.indent += 1
				_gen_nodes(node.finally_body, ctx)
				ctx.add_line("raise")
				ctx.indent -= 1

				# General exceptions
				e_name = ctx.next_tmp("__e")
				handled_name = ctx.next_tmp("__handled")
				ctx.add_line(f"except Exception as {e_name}:")
				ctx.indent += 1
				ctx.add_line(f"{handled_name} = False")
				for i, ex in enumerate(node.excepts):
					ctx.register_exc_type(ex.exc_type)
					cond_src = _exc_type_to_source(ex.exc_type)
					keyword = "if" if i == 0 else "elif"
					ctx.add_line(f"{keyword} isinstance({e_name}, {cond_src}):")
					ctx.indent += 1
					_gen_nodes(ex.body, ctx)
					ctx.add_line(f"{handled_name} = True")
					ctx.indent -= 1
				# finally must run whether or not the exception was handled
				_gen_nodes(node.finally_body, ctx)
				ctx.add_line(f"if not {handled_name}:")
				ctx.indent += 1
				ctx.add_line("raise")
				ctx.indent -= 1
				ctx.indent -= 1  # end general except

				# no exception
				ctx.add_line("else:")
				ctx.indent += 1
				_gen_nodes(node.finally_body, ctx)
				ctx.indent -= 1

			else:
				# no finally: catch specific exceptions only
				e_name = ctx.next_tmp("__e")
				handled_name = ctx.next_tmp("__handled")
				ctx.add_line(f"except Exception as {e_name}:")
				ctx.indent += 1
				ctx.add_line(f"{handled_name} = False")
				for i, ex in enumerate(node.excepts):
					ctx.register_exc_type(ex.exc_type)
					cond_src = _exc_type_to_source(ex.exc_type)
					keyword = "if" if i == 0 else "elif"
					ctx.add_line(f"{keyword} isinstance({e_name}, {cond_src}):")
					ctx.indent += 1
					_gen_nodes(ex.body, ctx)
					ctx.add_line(f"{handled_name} = True")
					ctx.indent -= 1
				ctx.add_line(f"if not {handled_name}:")
				ctx.indent += 1
				ctx.add_line("raise")
				ctx.indent -= 1
				ctx.indent -= 1

		else:
			raise TypeError(f"Unknown Node type in codegen: {node!r}")


def _generate_function_source(root: BlockNode, fn_name: str, prestate: Dict[str, Any]) -> Tuple[str, _CodegenContext]:
	ctx = _CodegenContext()
	# header with PRESTATE constant and function def
	ctx.add_line(f"PRESTATE = {repr(prestate)}")
	ctx.add_line("")
	ctx.add_line(f"def {fn_name}(argstate):")
	ctx.indent += 1
	ctx.add_line("state = PRESTATE.copy()")
	ctx.add_line("if argstate:")
	ctx.indent += 1
	ctx.add_line("state.update(argstate)")
	ctx.indent -= 1
	ctx.add_line("state.pop('return', None)")
	ctx.add_line("state.pop('error', None)")
	ctx.add_line("try:")
	ctx.indent += 1
	_gen_nodes(root.body, ctx)
	ctx.indent -= 1
	ctx.add_line("except BlockReturn:")
	ctx.indent += 1
	ctx.add_line("pass")
	ctx.indent -= 1
	ctx.add_line("except Exception as __e_top:")
	ctx.indent += 1
	ctx.add_line("state['error'] = __e_top")
	ctx.indent -= 1
	ctx.add_line("return state")
	ctx.indent -= 1

	source = "\n".join(ctx.lines)
	return source, ctx


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
"""


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
		return f"({_expr_to_js(expr.obj)})[{_expr_to_js(expr.index)}]"
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


class Block:
	"""Block is a small DSL program that transforms a state dict.

	Usage:

		pre = {"x": 1}
		with Block(pre) as b:
			b.x = b.x + 1
			with b.if_(b.x > 0):
				b.y = 10

		result = b({"z": 5})

	Semantics:
	  - state = prestate.copy(); state.update(argstate)
	  - missing keys read as None
	  - return_(expr) writes state["return"] and exits the block
	  - uncaught Python exceptions set state["error"]
	"""

	def __init__(self, prestate: Optional[Dict[str, Any]] = None):
		self._prestate = dict(prestate) if prestate is not None else {}
		self._stack: List[List[Node]] = []  # stack of current bodies
		self._root = BlockNode()

	# context manager for building
	def __enter__(self) -> "Block":
		self._stack.append(self._root.body)
		return self

	def __exit__(self, exc_type, exc, tb):
		self._stack.pop()

	# internal helper to get current body list
	@property
	def _body(self) -> List[Node]:
		if not self._stack:
			return self._root.body
		return self._stack[-1]

	# attribute access reads a symbolic variable as a VarExpr
	def __getattr__(self, name: str) -> VarExpr:
		if name.startswith("_"):
			raise AttributeError(name)
		return VarExpr(name)

	# assignment becomes AssignNode
	def __setattr__(self, name: str, value: Any) -> None:
		if name.startswith("_"):
			super().__setattr__(name, value)
		else:
			expr = to_expr(value)
			self._body.append(AssignNode(name, expr))

	###########################################################################
	# DSL control constructs
	###########################################################################

	# if / elif / else

	class _IfBuilder:
		def __init__(self, block: "Block", node: IfNode):
			self.block = block
			self.node = node

		def __enter__(self) -> "Block._IfBuilder":
			# push this if's body onto stack
			self.block._stack.append(self.node.body)
			return self

		def __exit__(self, exc_type, exc, tb):
			self.block._stack.pop()

		def elif_(self, cond: Expr) -> "Block._IfBuilder":
			enode = ElifNode(cond=to_expr(cond))
			self.node.elifs.append(enode)
			self.block._stack.append(enode.body)
			return self

		def else_(self) -> "Block._ElseBuilder":
			if self.node.else_body is None:
				self.node.else_body = []
			return Block._ElseBuilder(self.block, self.node)

	class _ElseBuilder:
		def __init__(self, block: "Block", if_node: IfNode):
			self.block = block
			self.if_node = if_node

		def __enter__(self) -> "Block._ElseBuilder":
			if self.if_node.else_body is None:
				self.if_node.else_body = []
			self.block._stack.append(self.if_node.else_body)
			return self

		def __exit__(self, exc_type, exc, tb):
			self.block._stack.pop()

	def if_(self, cond: Expr) -> "Block._IfBuilder":
		node = IfNode(cond=to_expr(cond))
		self._body.append(node)
		return Block._IfBuilder(self, node)

	# for / while

	class _LoopBuilder:
		def __init__(self, block: "Block", node: Union[ForNode, WhileNode]):
			self.block = block
			self.node = node

		def __enter__(self) -> "Block._LoopBuilder":
			self.block._stack.append(self.node.body)
			return self

		def __exit__(self, exc_type, exc, tb):
			self.block._stack.pop()

		def break_(self) -> None:
			# emit a break in the current scope (e.g. inside a nested if),
			# not unconditionally at the top of the loop body
			self.block._body.append(BreakNode())

		def continue_(self) -> None:
			self.block._body.append(ContinueNode())

	def for_(self, iterable: Expr, var: str = "item") -> "Block._LoopBuilder":
		node = ForNode(var=var, iterable=to_expr(iterable))
		self._body.append(node)
		return Block._LoopBuilder(self, node)

	def while_(self, cond: Expr) -> "Block._LoopBuilder":
		node = WhileNode(cond=to_expr(cond))
		self._body.append(node)
		return Block._LoopBuilder(self, node)

	# range helper
	def range(self, start: Any, stop: Any, step: Any = None) -> RangeExpr:
		return RangeExpr(start=to_expr(start), stop=to_expr(stop),
						 step=to_expr(step) if step is not None else None)

	# try / except / finally

	class _TryBuilder:
		def __init__(self, block: "Block", node: TryNode):
			self.block = block
			self.node = node

		def __enter__(self) -> "Block._TryBuilder":
			self.block._stack.append(self.node.body)
			return self

		def __exit__(self, exc_type, exc, tb):
			self.block._stack.pop()

		def except_(self, exc_type: Union[type, Tuple[type, ...]]) -> "Block._ExceptBuilder":
			enode = ExceptNode(exc_type=exc_type)
			self.node.excepts.append(enode)
			return Block._ExceptBuilder(self.block, self.node, enode)

		def finally_(self) -> "Block._FinallyBuilder":
			if self.node.finally_body is None:
				self.node.finally_body = []
			return Block._FinallyBuilder(self.block, self.node)

	class _ExceptBuilder:
		def __init__(self, block: "Block", try_node: TryNode, node: ExceptNode):
			self.block = block
			self.try_node = try_node
			self.node = node

		def __enter__(self) -> "Block._ExceptBuilder":
			self.block._stack.append(self.node.body)
			return self

		def __exit__(self, exc_type, exc, tb):
			self.block._stack.pop()

	class _FinallyBuilder:
		def __init__(self, block: "Block", try_node: TryNode):
			self.block = block
			self.try_node = try_node

		def __enter__(self) -> "Block._FinallyBuilder":
			if self.try_node.finally_body is None:
				self.try_node.finally_body = []
			self.block._stack.append(self.try_node.finally_body)
			return self

		def __exit__(self, exc_type, exc, tb):
			self.block._stack.pop()

	def try_(self) -> "Block._TryBuilder":
		node = TryNode()
		self._body.append(node)
		return Block._TryBuilder(self, node)

	# return
	def return_(self, value: Any) -> None:
		self._body.append(ReturnNode(to_expr(value)))

	# expression statement
	def expr(self, value: Any) -> None:
		self._body.append(ExprNode(to_expr(value)))

	# lvalue assignment: obj[index] = value  /  obj.name = value
	# (b.arr[i] = v can't be expressed via __setitem__ since the Expr has no
	# reference to the Block, so these explicit methods are the builder API.)
	def set_index(self, obj: Any, index: Any, value: Any) -> None:
		self._body.append(SetIndexNode(to_expr(obj), to_expr(index), to_expr(value)))

	def set_attr(self, obj: Any, name: str, value: Any) -> None:
		self._body.append(SetAttrNode(to_expr(obj), name, to_expr(value)))

	###########################################################################
	# Execution
	###########################################################################

	def __call__(self, argstate: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
		"""Execute this Block with the given argstate.

		Semantics:
		  - state = prestate.copy(); state.update(argstate or {})
		  - state["return"] and state["error"] are cleared at start
		  - On return_(expr): state["return"] = value
		  - On uncaught exception: state["error"] = exception
		"""
		state: Dict[str, Any] = self._prestate.copy()
		if argstate:
			state.update(argstate)
		# clear old return/error if present
		state.pop("return", None)
		state.pop("error", None)

		try:
			exec_nodes(self._root.body, state)
		except BlockReturn:
			# normal controlled return
			pass
		except Exception as e:
			# uncaught error: capture in state["error"]
			state["error"] = e

		return state

	###########################################################################
	# Compilation and export
	###########################################################################

	def compile(self, fn_name: str = "compiled_block") -> Tuple[Callable[[Dict[str, Any]], Dict[str, Any]], str]:
		"""Compile this Block into a standalone Python function.

		Returns (callable, source_code). The callable has the same behavior
		as calling this Block object directly.
		"""
		source, ctx = _generate_function_source(self._root, fn_name, self._prestate)
		# Prepare execution namespace
		ns: Dict[str, Any] = {}
		# Bind control-flow exceptions
		ns["BlockReturn"] = BlockReturn
		ns["LoopBreak"] = LoopBreak
		ns["LoopContinue"] = LoopContinue
		# Bind exception types used in TryNode except clauses
		for t in ctx.exc_types:
			if isinstance(t, tuple):
				for sub in t:
					ns[sub.__name__] = sub
			else:
				ns[t.__name__] = t
		exec(source, ns)
		fn = ns[fn_name]
		return fn, source

	def export(self, path: str, fn_name: str = "compiled_block") -> str:
		"""Export this Block as a standalone .py file defining a compiled function.

		The generated module will contain:
		  - BlockReturn, LoopBreak, LoopContinue definitions
		  - PRESTATE constant
		  - def fn_name(argstate): ...

		Returns the path written.
		"""
		source, ctx = _generate_function_source(self._root, fn_name, self._prestate)
		# Build a full module with control-flow exceptions + function
		module_lines: List[str] = []
		module_lines.append("from typing import Any, Dict")
		module_lines.append("")
		module_lines.append("class BlockReturn(BaseException):")
		module_lines.append("    pass")
		module_lines.append("")
		module_lines.append("class LoopBreak(BaseException):")
		module_lines.append("    pass")
		module_lines.append("")
		module_lines.append("class LoopContinue(BaseException):")
		module_lines.append("    pass")
		module_lines.append("")
		# Imports for exception types (non-builtin modules)
		imported: set[Tuple[str, str]] = set()
		for t in ctx.exc_types:
			types = t if isinstance(t, tuple) else (t,)
			for sub in types:
				mod = sub.__module__
				name = sub.__name__
				if mod != "builtins":
					key = (mod, name)
					if key not in imported:
						imported.add(key)
						module_lines.append(f"from {mod} import {name}")
		if imported:
			module_lines.append("")
		module_lines.append(source)
		module_text = "\n".join(module_lines)
		with open(path, "w", encoding="utf-8") as f:
			f.write(module_text)
		return path

	def compile_js(self, fn_name: str = "compiledBlock") -> str:
		"""Compile this Block into standalone JavaScript source.

		Returns the source (prelude helpers + ``function fn_name(argstate)``).
		Semantics mirror the Python backends; see the JavaScript codegen section
		for the deliberate, documented divergences (truthiness of []/{},
		=== equality, language-specific exception types and method names).
		"""
		fn_source = _generate_js_function_source(self._root, fn_name, self._prestate)
		return _JS_PRELUDE + "\n" + fn_source

	def export_js(self, path: str, fn_name: str = "compiledBlock") -> str:
		"""Export this Block as a standalone CommonJS ``.js`` module.

		The module defines ``function fn_name(argstate)`` and exports it via
		``module.exports``. Returns the path written.
		"""
		source = self.compile_js(fn_name)
		module_text = source + f"\n\nmodule.exports = {{ {fn_name}: {fn_name} }};\n"
		with open(path, "w", encoding="utf-8") as f:
			f.write(module_text)
		return path


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
			raise UnsupportedSyntaxError(node, "slices are not supported")
		return IndexExpr(_lower_expr(node.value), _lower_expr(node.slice))

	if isinstance(node, (_ast.List, _ast.Tuple)):
		return ListExpr([_lower_expr(e) for e in node.elts])

	if isinstance(node, _ast.Dict):
		if any(k is None for k in node.keys):
			raise UnsupportedSyntaxError(node, "dict unpacking (**) is not supported")
		return DictExpr([_lower_expr(k) for k in node.keys],
						[_lower_expr(v) for v in node.values])

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
		if h.name:
			raise UnsupportedSyntaxError(h, "'except ... as name' is not supported")
		node.excepts.append(ExceptNode(exc_type=_resolve_exc_type(h.type),
									   body=_lower_stmts(h.body)))
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
