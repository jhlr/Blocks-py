from __future__ import annotations
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

	def __radd__(self, other): return BinOpExpr("add", to_expr(other), self)
	def __rsub__(self, other): return BinOpExpr("sub", to_expr(other), self)
	def __rmul__(self, other): return BinOpExpr("mul", to_expr(other), self)
	def __rtruediv__(self, other): return BinOpExpr("div", to_expr(other), self)

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
	for node in nodes:
		if isinstance(node, AssignNode):
			expr_src = _expr_to_source(node.expr)
			ctx.add_line(f"state[{node.name!r}] = {expr_src}")

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
