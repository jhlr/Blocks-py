"""Blocks — an embedded DSL with one AST, four code generators, two front-ends.

The same tree can be interpreted, compiled to Python/JavaScript/Lua/Go, or built
by parsing real Python or JavaScript source.
"""
from __future__ import annotations

from .operators import UNARY_OPS, BINARY_OPS, SHORTCIRCUIT_OPS
from .errors import BlockReturn, LoopBreak, LoopContinue
from . import nodes
from .nodes import Expr, Node, to_expr
from .interpreter import eval_expr, exec_nodes
from .builder import Block
from .parse_python import parse_python, UnsupportedSyntaxError
from .parse_js import parse_js, JsSyntaxError
from .codegen_lua import LuaUnsupportedError, _lua_value
from .codegen_go import GoUnsupportedError

__all__ = [
    "Block", "parse_python", "parse_js",
    "eval_expr", "exec_nodes", "to_expr", "Expr", "Node", "nodes",
    "UnsupportedSyntaxError", "JsSyntaxError",
    "LuaUnsupportedError", "GoUnsupportedError",
    "BlockReturn", "LoopBreak", "LoopContinue",
    "UNARY_OPS", "BINARY_OPS", "SHORTCIRCUIT_OPS",
]
