from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from .errors import BlockReturn, LoopBreak, LoopContinue
from .nodes import (
	AssignNode, BlockNode, BreakNode, ContinueNode, ElifNode, ExceptNode, Expr, ExprNode, ForNode, IfNode, Node, RangeExpr, ReturnNode, SetAttrNode, SetIndexNode, TryNode, VarExpr, WhileNode, to_expr,
)
from .interpreter import exec_nodes
from .codegen_py import _generate_function_source
from .codegen_js import _JS_PRELUDE, _generate_js_function_source
from .codegen_lua import _LUA_PRELUDE, _generate_lua_function_source
from .codegen_go import _generate_go_source




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

		def except_(self, exc_type: Union[type, Tuple[type, ...]],
					name: Optional[str] = None) -> "Block._ExceptBuilder":
			enode = ExceptNode(exc_type=exc_type, name=name)
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

	def compile_go(self, fn_name: str = "compiledBlock") -> str:
		"""Compile this Block into a runnable Go program (`package main`).

		Every value is carried as `any` and a runtime of helpers reproduces
		Python's dynamic semantics. The program reads a JSON argstate from
		os.Args[1] and prints the JSON of state["return"]. See the Go codegen
		section for the divergences; some constructs raise GoUnsupportedError.
		"""
		return _generate_go_source(self._root, fn_name, self._prestate)

	def export_go(self, path: str, fn_name: str = "compiledBlock") -> str:
		"""Export this Block as a standalone runnable ``.go`` program."""
		source = self.compile_go(fn_name)
		with open(path, "w", encoding="utf-8") as f:
			f.write(source)
		return path
