from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from .nodes import (
	BreakNode, ContinueNode, ForNode, IfNode, Node, ReturnNode, TryNode, WhileNode,
)




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
