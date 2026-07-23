from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

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

@dataclass(slots=True)
class Expr:
	"""Base class for expressions."""
	pass


@dataclass(slots=True)
class LiteralExpr(Expr):
	value: Any


@dataclass(slots=True)
class VarExpr(Expr):
	name: str


@dataclass(slots=True)
class UnaryExpr(Expr):
	op: str  # "neg", "pos", "not"
	operand: Expr


@dataclass(slots=True)
class BinOpExpr(Expr):
	op: str  # "add", "sub", "mul", "div", "and", "or", "lt", ...
	left: Expr
	right: Expr


@dataclass(slots=True)
class CallExpr(Expr):
	func: Expr
	args: List[Expr] = field(default_factory=list)
	kwargs: Dict[str, Expr] = field(default_factory=dict)


@dataclass(slots=True)
class RangeExpr(Expr):
	start: Expr
	stop: Expr
	step: Optional[Expr] = None


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
		value = eval_expr(expr.operand, state)
		if expr.op == "neg":
			return -value
		if expr.op == "pos":
			return +value
		if expr.op == "not":
			return not value
		raise ValueError(f"Unknown unary op: {expr.op}")

	if isinstance(expr, BinOpExpr):
		# Short-circuit for and/or
		if expr.op == "and":
			left_val = eval_expr(expr.left, state)
			if not left_val:
				return left_val
			return eval_expr(expr.right, state)
		if expr.op == "or":
			left_val = eval_expr(expr.left, state)
			if left_val:
				return left_val
			return eval_expr(expr.right, state)

		left_val = eval_expr(expr.left, state)
		right_val = eval_expr(expr.right, state)

		if expr.op == "add":
			return left_val + right_val
		if expr.op == "sub":
			return left_val - right_val
		if expr.op == "mul":
			return left_val * right_val
		if expr.op == "div":
			return left_val / right_val
		if expr.op == "lt":
			return left_val < right_val
		if expr.op == "le":
			return left_val <= right_val
		if expr.op == "gt":
			return left_val > right_val
		if expr.op == "ge":
			return left_val >= right_val
		if expr.op == "eq":
			return left_val == right_val
		if expr.op == "ne":
			return left_val != right_val
		raise ValueError(f"Unknown binary op: {expr.op}")

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
		inner = _expr_to_source(expr.operand)
		if expr.op == "neg":
			return f"(-{inner})"
		if expr.op == "pos":
			return f"(+{inner})"
		if expr.op == "not":
			return f"(not {inner})"
		raise ValueError(f"Unknown unary op: {expr.op}")
	if isinstance(expr, BinOpExpr):
		op_map = {
			"add": "+",
			"sub": "-",
			"mul": "*",
			"div": "/",
			"lt": "<",
			"le": "<=",
			"gt": ">",
			"ge": ">=",
			"eq": "==",
			"ne": "!=",
			"and": "and",
			"or": "or",
		}
		if expr.op not in op_map:
			raise ValueError(f"Unknown binary op: {expr.op}")
		left = _expr_to_source(expr.left)
		right = _expr_to_source(expr.right)
		op = op_map[expr.op]
		return f"({left} {op} {right})"
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
				ctx.add_line("if not {0}:".format(handled_name))
				ctx.indent += 1
				_gen_nodes(node.finally_body, ctx)
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
# Builder API (Block & symbolic Var)
###############################################################################

class VarProxy:
	"""Proxy for building expressions via attribute access on Block."""

	def __init__(self, name: str):
		self.name = name

	# convert to Expr
	def _as_expr(self) -> Expr:
		return VarExpr(self.name)

	# arithmetic & comparison operators build BinOpExpr
	def __add__(self, other):  return BinOpExpr("add", self._as_expr(), to_expr(other))
	def __sub__(self, other):  return BinOpExpr("sub", self._as_expr(), to_expr(other))
	def __mul__(self, other):  return BinOpExpr("mul", self._as_expr(), to_expr(other))
	def __truediv__(self, other):  return BinOpExpr("div", self._as_expr(), to_expr(other))

	def __radd__(self, other): return BinOpExpr("add", to_expr(other), self._as_expr())
	def __rsub__(self, other): return BinOpExpr("sub", to_expr(other), self._as_expr())
	def __rmul__(self, other): return BinOpExpr("mul", to_expr(other), self._as_expr())
	def __rtruediv__(self, other): return BinOpExpr("div", to_expr(other), self._as_expr())

	def __lt__(self, other):   return BinOpExpr("lt",  self._as_expr(), to_expr(other))
	def __le__(self, other):   return BinOpExpr("le",  self._as_expr(), to_expr(other))
	def __gt__(self, other):   return BinOpExpr("gt",  self._as_expr(), to_expr(other))
	def __ge__(self, other):   return BinOpExpr("ge",  self._as_expr(), to_expr(other))
	def __eq__(self, other):   return BinOpExpr("eq",  self._as_expr(), to_expr(other))
	def __ne__(self, other):   return BinOpExpr("ne",  self._as_expr(), to_expr(other))

	def __neg__(self):         return UnaryExpr("neg", self._as_expr())
	def __pos__(self):         return UnaryExpr("pos", self._as_expr())

	# method call -> CallExpr
	def call(self, *args, **kwargs) -> CallExpr:
		return CallExpr(self._as_expr(),
						[to_expr(a) for a in args],
						{k: to_expr(v) for k, v in kwargs.items()})


def to_expr(value: Any) -> Expr:
	if isinstance(value, Expr):
		return value
	if isinstance(value, VarProxy):
		return value._as_expr()
	return LiteralExpr(value)


class Block:
	"""Block is a small DSL program that transforms a state dict.

	Usage:

		pre = {"x": 1}
		with Block(pre) as b:
			b.x = b.x + 1
			with b.if_(b.x > 0) as then:
				then.y = 10

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

	# attribute access creates VarProxy (for expressions)
	def __getattr__(self, name: str) -> VarProxy:
		if name.startswith("_"):
			raise AttributeError(name)
		return VarProxy(name)

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
			# break out of this loop
			self.node.body.append(BreakNode())

		def continue_(self) -> None:
			self.node.body.append(ContinueNode())

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
