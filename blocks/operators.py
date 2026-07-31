from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import operator



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
