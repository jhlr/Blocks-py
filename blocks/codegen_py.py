from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from .operators import BINARY_OPS, SHORTCIRCUIT_OPS, UNARY_OPS
from .nodes import (
	AssignNode, AttrExpr, BinOpExpr, BlockNode, BreakNode, CallExpr, ConcatExpr, CondExpr, ContinueNode, DictExpr, Expr, ExprNode, ForNode, IfNode, IndexExpr, ListExpr, LiteralExpr, Node, RangeExpr, ReturnNode, SetAttrNode, SetIndexNode, SliceExpr, TryNode, UnaryExpr, VarExpr, WhileNode,
)
from .codegen_base import _CodegenContext




def _slice_to_source(sl: SliceExpr) -> str:
	lo = _expr_to_source(sl.lower) if sl.lower is not None else ""
	up = _expr_to_source(sl.upper) if sl.upper is not None else ""
	if sl.step is not None:
		return f"{lo}:{up}:{_expr_to_source(sl.step)}"
	return f"{lo}:{up}"




def _concat_part_to_source(part: Expr) -> str:
	# literal string segments emit verbatim; everything else is str()-wrapped
	if isinstance(part, LiteralExpr) and isinstance(part.value, str):
		return repr(part.value)
	return f"str({_expr_to_source(part)})"




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
		if isinstance(expr.index, SliceExpr):
			return f"({_expr_to_source(expr.obj)})[{_slice_to_source(expr.index)}]"
		return f"({_expr_to_source(expr.obj)})[{_expr_to_source(expr.index)}]"
	if isinstance(expr, CondExpr):
		return (f"({_expr_to_source(expr.then)} if {_expr_to_source(expr.cond)}"
				f" else {_expr_to_source(expr.orelse)})")
	if isinstance(expr, ConcatExpr):
		return "(" + " + ".join(_concat_part_to_source(p) for p in expr.parts) + ")"
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
					if ex.name is not None:
						ctx.add_line(f"state[{ex.name!r}] = {e_name}")
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
					if ex.name is not None:
						ctx.add_line(f"state[{ex.name!r}] = {e_name}")
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
