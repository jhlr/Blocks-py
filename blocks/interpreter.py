from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from .operators import BINARY_OPS, UNARY_OPS
from .errors import BlockReturn, LoopBreak, LoopContinue
from .nodes import (
	AssignNode, AttrExpr, BinOpExpr, BreakNode, CallExpr, ConcatExpr, CondExpr, ContinueNode, DictExpr, Expr, ExprNode, ForNode, IfNode, IndexExpr, ListExpr, LiteralExpr, Node, RangeExpr, ReturnNode, SetAttrNode, SetIndexNode, SliceExpr, TryNode, UnaryExpr, VarExpr, WhileNode,
)




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

	if isinstance(expr, SliceExpr):
		lower = eval_expr(expr.lower, state) if expr.lower is not None else None
		upper = eval_expr(expr.upper, state) if expr.upper is not None else None
		step = eval_expr(expr.step, state) if expr.step is not None else None
		return slice(lower, upper, step)

	if isinstance(expr, AttrExpr):
		return getattr(eval_expr(expr.obj, state), expr.name)

	if isinstance(expr, ListExpr):
		return [eval_expr(e, state) for e in expr.elts]

	if isinstance(expr, DictExpr):
		return {eval_expr(k, state): eval_expr(v, state)
				for k, v in zip(expr.keys, expr.values)}

	if isinstance(expr, CondExpr):
		if eval_expr(expr.cond, state):
			return eval_expr(expr.then, state)
		return eval_expr(expr.orelse, state)

	if isinstance(expr, ConcatExpr):
		return "".join(_pystr(eval_expr(p, state)) for p in expr.parts)

	raise TypeError(f"Unknown Expr type: {expr!r}")




def _pystr(v: Any) -> str:
	"""Stringify a value for concatenation, matching str() but with '' for None-ish."""
	return str(v)




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
						if ex.name is not None:
							state[ex.name] = e
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
