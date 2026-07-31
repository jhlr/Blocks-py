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

# Eager binary operators: op -> (python symbol, runtime function, js symbol,
# lua symbol). Both operands are always evaluated. Most symbols coincide; the
# ones that differ are ==/!= (JS ===/!==, Lua ==/~=) and ** (Lua ^).
BINARY_OPS: Dict[str, Tuple[str, Callable[[Any, Any], Any], str, str]] = {
	"add": ("+", operator.add, "+", "+"),
	"sub": ("-", operator.sub, "-", "-"),
	"mul": ("*", operator.mul, "*", "*"),
	"div": ("/", operator.truediv, "/", "/"),
	"mod": ("%", operator.mod, "%", "%"),
	"pow": ("**", operator.pow, "**", "^"),
	"lt": ("<", operator.lt, "<", "<"),
	"le": ("<=", operator.le, "<=", "<="),
	"gt": (">", operator.gt, ">", ">"),
	"ge": (">=", operator.ge, ">=", ">="),
	"eq": ("==", operator.eq, "===", "=="),
	"ne": ("!=", operator.ne, "!==", "~="),
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
	loop_labels: List[str] = field(default_factory=list)  # Lua `continue` targets

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
"""


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
		return f"({_expr_to_lua(expr.obj)})[{_expr_to_lua(expr.index)}]"
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


def _lua_escapes_flow(nodes: List[Node]) -> bool:
	"""True if a break/continue/return in `nodes` would escape an enclosing try.

	Return always escapes. break/continue escape unless captured by a loop that
	lives inside the try body itself. Used to reject try bodies whose control
	flow a pcall'd closure could not carry across its boundary.
	"""
	for node in nodes:
		if isinstance(node, ReturnNode):
			return True
		if isinstance(node, (BreakNode, ContinueNode)):
			return True
		if isinstance(node, IfNode):
			if (_lua_escapes_flow(node.body)
					or any(_lua_escapes_flow(e.body) for e in node.elifs)
					or (node.else_body is not None and _lua_escapes_flow(node.else_body))):
				return True
		elif isinstance(node, (ForNode, WhileNode)):
			# a nested loop captures its own break/continue, but a return inside
			# it still escapes
			if _lua_return_inside(node.body):
				return True
		elif isinstance(node, TryNode):
			if (_lua_escapes_flow(node.body)
					or any(_lua_escapes_flow(e.body) for e in node.excepts)
					or (node.finally_body is not None and _lua_escapes_flow(node.finally_body))):
				return True
	return False


def _lua_return_inside(nodes: List[Node]) -> bool:
	"""True if any ReturnNode appears anywhere within `nodes` (through loops)."""
	for node in nodes:
		if isinstance(node, ReturnNode):
			return True
		if isinstance(node, IfNode):
			if (_lua_return_inside(node.body)
					or any(_lua_return_inside(e.body) for e in node.elifs)
					or (node.else_body is not None and _lua_return_inside(node.else_body))):
				return True
		elif isinstance(node, (ForNode, WhileNode)):
			if _lua_return_inside(node.body):
				return True
		elif isinstance(node, TryNode):
			if (_lua_return_inside(node.body)
					or any(_lua_return_inside(e.body) for e in node.excepts)
					or (node.finally_body is not None and _lua_return_inside(node.finally_body))):
				return True
	return False


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

	def compile_lua(self, fn_name: str = "compiled_block") -> str:
		"""Compile this Block into standalone Lua source.

		Returns the source (prelude helpers + ``function fn_name(argstate)``).
		Truthiness is made faithful to Python via helpers; see the Lua codegen
		section for the divergences and for constructs that raise
		LuaUnsupportedError (e.g. control flow escaping a try body).
		"""
		fn_source = _generate_lua_function_source(self._root, fn_name, self._prestate)
		return _LUA_PRELUDE + "\n" + fn_source

	def export_lua(self, path: str, fn_name: str = "compiled_block") -> str:
		"""Export this Block as a standalone Lua module returning ``fn_name``."""
		source = self.compile_lua(fn_name)
		module_text = source + f"\n\nreturn {fn_name}\n"
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


###############################################################################
# Front-end — parse a subset of JavaScript into the Blocks AST
#
# A small hand-written tokenizer + recursive-descent parser. Like parse_python,
# the result is a Block, so a JS program can then be interpreted or compiled to
# Python / JavaScript / Lua. This closes the loop: JavaScript -> Python.
#
# Key mapping: JS member access `obj.prop` and `obj[k]` are *property* access,
# which corresponds to Python dict/subscript access — both lower to IndexExpr
# (never AttrExpr). So `d.k` in JS becomes `d["k"]` in Python, the faithful
# data-access translation. Method calls on values (arr.push, s.toUpperCase) have
# no Python equivalent and are simply not portable.
#
# Supported: an optional single top-level `function name(params) { ... }` (its
# body becomes the block; constant default params seed the prestate) or bare
# statements. Statements: let/const/var declaration, assignment (name / index /
# member target, incl. += -= *= /= %=), ++/--, expression statement, return,
# break, continue, if/else if/else, while, for-of, C-style for (counting loops
# become a range for; others desugar to while and reject `continue`), try/catch/
# finally, block. Expressions: number/string/true/false/null/undefined,
# identifiers, unary (- + !), binary (+ - * / % ** < <= > >= === !== == !=,
# && ||), calls, member/index access, array and object literals, parentheses.
# Anything else raises JsSyntaxError.
###############################################################################

class JsSyntaxError(Exception):
	"""A JavaScript construct outside the subset that parse_js understands."""


# multi-char punctuators first so the tokenizer matches greedily
_JS_PUNCT = sorted([
	"===", "!==", "==", "!=", "<=", ">=", "&&", "||", "**",
	"+=", "-=", "*=", "/=", "%=", "++", "--",
	"(", ")", "{", "}", "[", "]", ",", ";", ".", ":",
	"+", "-", "*", "/", "%", "<", ">", "=", "!",
], key=len, reverse=True)

_JS_KEYWORDS = {
	"function", "return", "if", "else", "while", "for", "of", "in",
	"let", "const", "var", "break", "continue",
	"true", "false", "null", "undefined", "try", "catch", "finally",
}

# JS binary operator token -> (Blocks op, precedence). Higher binds tighter.
_JS_BINOP_PREC = {
	"||": ("or", 1), "&&": ("and", 2),
	"===": ("eq", 3), "!==": ("ne", 3), "==": ("eq", 3), "!=": ("ne", 3),
	"<": ("lt", 4), "<=": ("le", 4), ">": ("gt", 4), ">=": ("ge", 4),
	"+": ("add", 5), "-": ("sub", 5),
	"*": ("mul", 6), "/": ("div", 6), "%": ("mod", 6),
	"**": ("pow", 7),  # right-associative
}
_JS_AUG = {"+=": "add", "-=": "sub", "*=": "mul", "/=": "div", "%=": "mod"}


def _js_tokenize(src: str) -> List[Tuple[str, Any]]:
	tokens: List[Tuple[str, Any]] = []
	i, n = 0, len(src)
	while i < n:
		c = src[i]
		if c in " \t\r\n":
			i += 1
			continue
		if c == "/" and i + 1 < n and src[i + 1] == "/":
			i = src.find("\n", i)
			if i == -1:
				break
			continue
		if c == "/" and i + 1 < n and src[i + 1] == "*":
			end = src.find("*/", i + 2)
			if end == -1:
				raise JsSyntaxError("unterminated block comment")
			i = end + 2
			continue
		if c.isdigit() or (c == "." and i + 1 < n and src[i + 1].isdigit()):
			j = i
			while j < n and (src[j].isdigit() or src[j] in ".eE" or
							 (src[j] in "+-" and src[j - 1] in "eE")):
				j += 1
			text = src[i:j]
			value: Any = float(text) if any(ch in text for ch in ".eE") else int(text)
			tokens.append(("NUMBER", value))
			i = j
			continue
		if c in "'\"":
			j = i + 1
			buf = []
			while j < n and src[j] != c:
				if src[j] == "\\" and j + 1 < n:
					esc = src[j + 1]
					buf.append({"n": "\n", "t": "\t", "r": "\r", "\\": "\\",
								'"': '"', "'": "'", "/": "/", "0": "\0"}.get(esc, esc))
					j += 2
					continue
				buf.append(src[j])
				j += 1
			if j >= n:
				raise JsSyntaxError("unterminated string literal")
			tokens.append(("STRING", "".join(buf)))
			i = j + 1
			continue
		if c.isalpha() or c == "_" or c == "$":
			j = i
			while j < n and (src[j].isalnum() or src[j] in "_$"):
				j += 1
			word = src[i:j]
			tokens.append(("KEYWORD" if word in _JS_KEYWORDS else "NAME", word))
			i = j
			continue
		for p in _JS_PUNCT:
			if src.startswith(p, i):
				tokens.append(("PUNCT", p))
				i += len(p)
				break
		else:
			raise JsSyntaxError(f"unexpected character {c!r} at offset {i}")
	tokens.append(("EOF", None))
	return tokens


def _contains_continue(nodes: List[Node]) -> bool:
	"""True if a ContinueNode sits at this loop's level (not inside a nested loop)."""
	for node in nodes:
		if isinstance(node, ContinueNode):
			return True
		if isinstance(node, IfNode):
			if (_contains_continue(node.body)
					or any(_contains_continue(e.body) for e in node.elifs)
					or (node.else_body is not None and _contains_continue(node.else_body))):
				return True
		elif isinstance(node, TryNode):
			if (_contains_continue(node.body)
					or any(_contains_continue(e.body) for e in node.excepts)
					or (node.finally_body is not None and _contains_continue(node.finally_body))):
				return True
	return False


class _JsParser:
	def __init__(self, tokens: List[Tuple[str, Any]]):
		self.toks = tokens
		self.i = 0

	# --- token helpers ---
	def _peek(self, k: int = 0) -> Tuple[str, Any]:
		return self.toks[self.i + k]

	def _next(self) -> Tuple[str, Any]:
		tok = self.toks[self.i]
		self.i += 1
		return tok

	def _at(self, kind: str, value: Any = None) -> bool:
		t = self.toks[self.i]
		return t[0] == kind and (value is None or t[1] == value)

	def _eat(self, kind: str, value: Any = None) -> Tuple[str, Any]:
		t = self.toks[self.i]
		if t[0] != kind or (value is not None and t[1] != value):
			raise JsSyntaxError(f"expected {value or kind}, got {t[1]!r}")
		self.i += 1
		return t

	def _accept(self, kind: str, value: Any = None) -> bool:
		if self._at(kind, value):
			self.i += 1
			return True
		return False

	# --- program ---
	def parse_program(self) -> Tuple[List[Node], Dict[str, Any]]:
		# a single top-level function declaration, or bare statements
		if self._at("KEYWORD", "function") and self._peek(1)[0] == "NAME":
			self._eat("KEYWORD", "function")
			self._eat("NAME")
			prestate = self._parse_params()
			body = self._parse_block()
			self._eat("EOF")
			return body, prestate
		stmts: List[Node] = []
		while not self._at("EOF"):
			stmts.extend(self._statement())
		return stmts, {}

	def _parse_params(self) -> Dict[str, Any]:
		prestate: Dict[str, Any] = {}
		self._eat("PUNCT", "(")
		while not self._at("PUNCT", ")"):
			name = self._eat("NAME")[1]
			if self._accept("PUNCT", "="):
				default = self._expression()
				if not isinstance(default, LiteralExpr):
					raise JsSyntaxError("default parameter must be a constant")
				prestate[name] = default.value
			if not self._accept("PUNCT", ","):
				break
		self._eat("PUNCT", ")")
		return prestate

	def _parse_block(self) -> List[Node]:
		self._eat("PUNCT", "{")
		stmts: List[Node] = []
		while not self._at("PUNCT", "}"):
			stmts.extend(self._statement())
		self._eat("PUNCT", "}")
		return stmts

	# a statement body that may be a { block } or a single statement
	def _stmt_body(self) -> List[Node]:
		if self._at("PUNCT", "{"):
			return self._parse_block()
		return self._statement()

	# --- statements ---
	def _statement(self) -> List[Node]:
		t = self._peek()
		if t == ("PUNCT", "{"):
			return self._parse_block()
		if t == ("PUNCT", ";"):
			self.i += 1
			return []
		if t[0] == "KEYWORD":
			kw = t[1]
			if kw in ("let", "const", "var"):
				return self._var_decl()
			if kw == "return":
				return self._return_stmt()
			if kw == "break":
				self.i += 1
				self._accept("PUNCT", ";")
				return [BreakNode()]
			if kw == "continue":
				self.i += 1
				self._accept("PUNCT", ";")
				return [ContinueNode()]
			if kw == "if":
				return [self._if_stmt()]
			if kw == "while":
				return [self._while_stmt()]
			if kw == "for":
				return self._for_stmt()
			if kw == "try":
				return [self._try_stmt()]
			raise JsSyntaxError(f"unexpected keyword {kw!r}")
		return self._expr_or_assign_stmt()

	def _var_decl(self) -> List[Node]:
		self.i += 1  # let/const/var
		out: List[Node] = []
		while True:
			name = self._eat("NAME")[1]
			if self._accept("PUNCT", "="):
				out.append(AssignNode(name, self._expression()))
			# `let x;` with no initializer sets nothing (reads as None later)
			if not self._accept("PUNCT", ","):
				break
		self._accept("PUNCT", ";")
		return out

	def _return_stmt(self) -> List[Node]:
		self.i += 1
		if self._at("PUNCT", ";") or self._at("PUNCT", "}"):
			self._accept("PUNCT", ";")
			return [ReturnNode(LiteralExpr(None))]
		expr = self._expression()
		self._accept("PUNCT", ";")
		return [ReturnNode(expr)]

	def _if_stmt(self) -> IfNode:
		self._eat("KEYWORD", "if")
		self._eat("PUNCT", "(")
		cond = self._expression()
		self._eat("PUNCT", ")")
		node = IfNode(cond=cond, body=self._stmt_body())
		if self._accept("KEYWORD", "else"):
			if self._at("KEYWORD", "if"):
				# else if -> flatten into elif chain by absorbing the nested if
				nested = self._if_stmt()
				node.elifs.append(ElifNode(cond=nested.cond, body=nested.body))
				node.elifs.extend(nested.elifs)
				node.else_body = nested.else_body
			else:
				node.else_body = self._stmt_body()
		return node

	def _while_stmt(self) -> WhileNode:
		self._eat("KEYWORD", "while")
		self._eat("PUNCT", "(")
		cond = self._expression()
		self._eat("PUNCT", ")")
		return WhileNode(cond=cond, body=self._stmt_body())

	def _for_stmt(self) -> List[Node]:
		self._eat("KEYWORD", "for")
		self._eat("PUNCT", "(")
		save = self.i
		# try for-of: [let|const|var] NAME of ITER
		var: Optional[str] = None
		if self._peek()[0] == "KEYWORD" and self._peek()[1] in ("let", "const", "var"):
			self.i += 1
		if self._peek()[0] == "NAME" and self._peek(1) == ("KEYWORD", "of"):
			var = self._eat("NAME")[1]
			self._eat("KEYWORD", "of")
			iterable = self._expression()
			self._eat("PUNCT", ")")
			return [ForNode(var=var, iterable=iterable, body=self._stmt_body())]
		# C-style for (init; cond; update)
		self.i = save
		init: List[Node] = []
		if not self._at("PUNCT", ";"):
			if self._peek()[0] == "KEYWORD" and self._peek()[1] in ("let", "const", "var"):
				init = self._var_decl_no_semi()
			else:
				init = self._simple_assign()
		self._eat("PUNCT", ";")
		cond: Expr = self._expression() if not self._at("PUNCT", ";") else LiteralExpr(True)
		self._eat("PUNCT", ";")
		update: List[Node] = [] if self._at("PUNCT", ")") else self._simple_assign()
		self._eat("PUNCT", ")")
		body = self._stmt_body()
		return self._build_for(init, cond, update, body)

	def _var_decl_no_semi(self) -> List[Node]:
		self.i += 1
		name = self._eat("NAME")[1]
		self._eat("PUNCT", "=")
		return [AssignNode(name, self._expression())]

	def _simple_assign(self) -> List[Node]:
		# an assignment / ++ / -- used in for-init / for-update, no trailing ;
		return self._parse_assignment(require_semi=False)

	def _build_for(self, init: List[Node], cond: Expr, update: List[Node],
				   body: List[Node]) -> List[Node]:
		# recognise a counting loop -> range for (so `continue` works correctly)
		counting = self._as_counting_for(init, cond, update, body)
		if counting is not None:
			return [counting]
		# general C-style for -> while, with update appended each iteration
		if _contains_continue(body):
			raise JsSyntaxError(
				"`continue` in a non-counting C-style for is not supported "
				"(the update would be skipped); use for-of or a while loop")
		return [*init, WhileNode(cond=cond, body=[*body, *update])]

	def _as_counting_for(self, init: List[Node], cond: Expr, update: List[Node],
						  body: List[Node]) -> Optional[ForNode]:
		if len(init) != 1 or not isinstance(init[0], AssignNode):
			return None
		if len(update) != 1 or not isinstance(update[0], AssignNode):
			return None
		var = init[0].name
		start = init[0].expr
		if update[0].name != var:
			return None
		# update must be var = var + STEP
		up = update[0].expr
		if not (isinstance(up, BinOpExpr) and up.op == "add"
				and isinstance(up.left, VarExpr) and up.left.name == var
				and isinstance(up.right, LiteralExpr)):
			return None
		step = up.right.value
		# cond must be var < STOP or var <= STOP
		if not (isinstance(cond, BinOpExpr) and isinstance(cond.left, VarExpr)
				and cond.left.name == var and cond.op in ("lt", "le")):
			return None
		stop = cond.right
		if cond.op == "le":
			stop = BinOpExpr("add", stop, LiteralExpr(1))
		step_expr = None if step == 1 else LiteralExpr(step)
		return ForNode(var=var, iterable=RangeExpr(start, stop, step_expr), body=body)

	def _try_stmt(self) -> TryNode:
		self._eat("KEYWORD", "try")
		node = TryNode(body=self._parse_block())
		if self._accept("KEYWORD", "catch"):
			if self._accept("PUNCT", "("):  # optional binding, ignored
				self._eat("NAME")
				self._eat("PUNCT", ")")
			node.excepts.append(ExceptNode(exc_type=Exception, body=self._parse_block()))
		if self._accept("KEYWORD", "finally"):
			node.finally_body = self._parse_block()
		return node

	def _expr_or_assign_stmt(self) -> List[Node]:
		nodes = self._parse_assignment(require_semi=True)
		return nodes

	def _parse_assignment(self, require_semi: bool) -> List[Node]:
		lhs = self._expression()
		out: List[Node]
		if self._at("PUNCT", "=") or any(self._at("PUNCT", a) for a in _JS_AUG):
			op_tok = self._next()[1]
			rhs = self._expression()
			if op_tok in _JS_AUG:
				rhs = BinOpExpr(_JS_AUG[op_tok], lhs, rhs)
			out = [self._assign_node(lhs, rhs)]
		elif self._at("PUNCT", "++") or self._at("PUNCT", "--"):
			op = "add" if self._next()[1] == "++" else "sub"
			out = [self._assign_node(lhs, BinOpExpr(op, lhs, LiteralExpr(1)))]
		else:
			out = [ExprNode(lhs)]
		if require_semi:
			self._accept("PUNCT", ";")
		return out

	def _assign_node(self, lhs: Expr, rhs: Expr) -> Node:
		if isinstance(lhs, VarExpr):
			return AssignNode(lhs.name, rhs)
		if isinstance(lhs, IndexExpr):
			return SetIndexNode(lhs.obj, lhs.index, rhs)
		raise JsSyntaxError("invalid assignment target")

	# --- expressions (precedence climbing) ---
	def _expression(self) -> Expr:
		return self._binary(1)

	def _binary(self, min_prec: int) -> Expr:
		left = self._unary()
		while True:
			t = self._peek()
			if t[0] != "PUNCT" or t[1] not in _JS_BINOP_PREC:
				break
			op, prec = _JS_BINOP_PREC[t[1]]
			if prec < min_prec:
				break
			self.i += 1
			# ** is right-associative; others left
			next_min = prec if t[1] == "**" else prec + 1
			right = self._binary(next_min)
			left = BinOpExpr(op, left, right)
		return left

	def _unary(self) -> Expr:
		t = self._peek()
		if t == ("PUNCT", "-"):
			self.i += 1
			return UnaryExpr("neg", self._unary())
		if t == ("PUNCT", "+"):
			self.i += 1
			return UnaryExpr("pos", self._unary())
		if t == ("PUNCT", "!"):
			self.i += 1
			return UnaryExpr("not", self._unary())
		return self._postfix()

	def _postfix(self) -> Expr:
		expr = self._primary()
		while True:
			if self._accept("PUNCT", "."):
				name = self._eat("NAME")[1]
				expr = IndexExpr(expr, LiteralExpr(name))  # property = dict access
			elif self._accept("PUNCT", "["):
				index = self._expression()
				self._eat("PUNCT", "]")
				expr = IndexExpr(expr, index)
			elif self._at("PUNCT", "("):
				expr = CallExpr(expr, self._args())
			else:
				break
		return expr

	def _args(self) -> List[Expr]:
		self._eat("PUNCT", "(")
		args: List[Expr] = []
		while not self._at("PUNCT", ")"):
			args.append(self._expression())
			if not self._accept("PUNCT", ","):
				break
		self._eat("PUNCT", ")")
		return args

	def _primary(self) -> Expr:
		t = self._next()
		kind, value = t
		if kind == "NUMBER":
			return LiteralExpr(value)
		if kind == "STRING":
			return LiteralExpr(value)
		if kind == "NAME":
			return VarExpr(value)
		if kind == "KEYWORD":
			if value == "true":
				return LiteralExpr(True)
			if value == "false":
				return LiteralExpr(False)
			if value in ("null", "undefined"):
				return LiteralExpr(None)
			raise JsSyntaxError(f"unexpected keyword {value!r} in expression")
		if t == ("PUNCT", "("):
			expr = self._expression()
			self._eat("PUNCT", ")")
			return expr
		if t == ("PUNCT", "["):
			elts: List[Expr] = []
			while not self._at("PUNCT", "]"):
				elts.append(self._expression())
				if not self._accept("PUNCT", ","):
					break
			self._eat("PUNCT", "]")
			return ListExpr(elts)
		if t == ("PUNCT", "{"):
			keys: List[Expr] = []
			values: List[Expr] = []
			while not self._at("PUNCT", "}"):
				kt = self._next()
				if kt[0] == "STRING":
					keys.append(LiteralExpr(kt[1]))
				elif kt[0] in ("NAME", "KEYWORD"):
					keys.append(LiteralExpr(kt[1]))
				elif kt[0] == "NUMBER":
					keys.append(LiteralExpr(kt[1]))
				else:
					raise JsSyntaxError(f"invalid object key {kt[1]!r}")
				self._eat("PUNCT", ":")
				values.append(self._expression())
				if not self._accept("PUNCT", ","):
					break
			self._eat("PUNCT", "}")
			return DictExpr(keys, values)
		raise JsSyntaxError(f"unexpected token {value!r}")


def parse_js(source: str, prestate: Optional[Dict[str, Any]] = None) -> Block:
	"""Parse a subset of JavaScript source into a Block.

	The source may be a single top-level ``function name(params) { ... }`` (its
	body becomes the block and constant default params seed the prestate) or a
	sequence of statements. Explicit `prestate` overrides parameter defaults.

	Raises JsSyntaxError on any construct outside the supported subset. Member
	access (obj.prop / obj[k]) lowers to subscript, matching Python dict access.
	"""
	parser = _JsParser(_js_tokenize(source))
	body, seed = parser.parse_program()
	if prestate:
		seed = {**seed, **prestate}
	block = Block(seed)
	block._root.body.extend(body)
	return block
