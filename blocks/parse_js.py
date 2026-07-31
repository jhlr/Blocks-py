from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from .nodes import (
	AssignNode, BinOpExpr, BreakNode, CallExpr, ConcatExpr, CondExpr, ContinueNode, DictExpr, ElifNode, ExceptNode, Expr, ExprNode, ForNode, IfNode, IndexExpr, ListExpr, LiteralExpr, Node, RangeExpr, ReturnNode, SetIndexNode, TryNode, UnaryExpr, VarExpr, WhileNode,
)
from .builder import Block




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
	"===", "!==", "==", "!=", "<=", ">=", "&&", "||", "**", "=>",
	"+=", "-=", "*=", "/=", "%=", "++", "--",
	"(", ")", "{", "}", "[", "]", ",", ";", ".", ":", "?",
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
		if c == "`":  # template literal -> parts: ("str", text) | ("expr", source)
			j = i + 1
			parts: List[Tuple[str, str]] = []
			buf = []
			while j < n and src[j] != "`":
				if src[j] == "\\" and j + 1 < n:
					esc = src[j + 1]
					buf.append({"n": "\n", "t": "\t", "r": "\r", "\\": "\\",
								"`": "`", "$": "$"}.get(esc, esc))
					j += 2
					continue
				if src[j] == "$" and j + 1 < n and src[j + 1] == "{":
					parts.append(("str", "".join(buf)))
					buf = []
					depth = 1
					k = j + 2
					while k < n and depth > 0:
						if src[k] == "{":
							depth += 1
						elif src[k] == "}":
							depth -= 1
							if depth == 0:
								break
						k += 1
					if depth != 0:
						raise JsSyntaxError("unterminated template expression")
					parts.append(("expr", src[j + 2:k]))
					j = k + 1
					continue
				buf.append(src[j])
				j += 1
			if j >= n:
				raise JsSyntaxError("unterminated template literal")
			parts.append(("str", "".join(buf)))
			tokens.append(("TEMPLATE", parts))
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
		# a single top-level function declaration (`function name(){}`), a
		# top-level arrow function, or bare statements
		if self._at("KEYWORD", "function") and self._peek(1)[0] == "NAME":
			self._eat("KEYWORD", "function")
			self._eat("NAME")
			prestate = self._parse_params()
			body = self._parse_block()
			self._eat("EOF")
			return body, prestate
		save = self.i
		arrow = self._try_toplevel_arrow()
		if arrow is not None:
			return arrow
		self.i = save
		stmts: List[Node] = []
		while not self._at("EOF"):
			stmts.extend(self._statement())
		return stmts, {}

	def _try_toplevel_arrow(self) -> Optional[Tuple[List[Node], Dict[str, Any]]]:
		"""Parse `(params) => body` or `x => body` as the whole program.

		Only the top-level arrow form is supported (it maps to the block +
		prestate, no closures). Inline arrow values are not supported.
		"""
		try:
			if self._at("NAME") and self._peek(1) == ("PUNCT", "=>"):
				prestate: Dict[str, Any] = {}
				self._eat("NAME")
			elif self._at("PUNCT", "("):
				prestate = self._parse_params()
			else:
				return None
			if not self._accept("PUNCT", "=>"):
				return None
			if self._at("PUNCT", "{"):
				body = self._parse_block()
			else:
				body = [ReturnNode(self._expression())]
			self._eat("EOF")
			return body, prestate
		except JsSyntaxError:
			return None

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
			catch_name: Optional[str] = None
			if self._accept("PUNCT", "("):  # optional binding
				catch_name = self._eat("NAME")[1]
				self._eat("PUNCT", ")")
			node.excepts.append(ExceptNode(exc_type=Exception,
										   body=self._parse_block(), name=catch_name))
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
		expr = self._binary(1)
		if self._at("PUNCT", "?"):  # ternary cond ? then : orelse
			self.i += 1
			then = self._expression()
			self._eat("PUNCT", ":")
			orelse = self._expression()
			return CondExpr(expr, then, orelse)
		return expr

	def parse_expression_only(self) -> Expr:
		expr = self._expression()
		self._eat("EOF")
		return expr

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
		if kind == "TEMPLATE":
			parts: List[Expr] = []
			for ptype, text in value:
				if ptype == "str":
					if text:
						parts.append(LiteralExpr(text))
				else:  # embedded ${expr}
					parts.append(_JsParser(_js_tokenize(text)).parse_expression_only())
			return ConcatExpr(parts if parts else [LiteralExpr("")])
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
