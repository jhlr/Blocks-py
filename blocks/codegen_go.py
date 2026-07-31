from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import json
from .nodes import (
	AssignNode, AttrExpr, BinOpExpr, BlockNode, BreakNode, CallExpr, ConcatExpr, CondExpr, ContinueNode, DictExpr, Expr, ExprNode, ForNode, IfNode, IndexExpr, ListExpr, LiteralExpr, Node, RangeExpr, ReturnNode, SetAttrNode, SetIndexNode, SliceExpr, TryNode, UnaryExpr, VarExpr, WhileNode,
)
from .codegen_base import _CodegenContext, _lua_escapes_flow




###############################################################################
# Code generation — Go backend
#
# Go is statically typed, so every value is carried as `any` (interface{}) and a
# small runtime of helpers reproduces Python's dynamic semantics: `_add`/`_sub`/
# ... type-switch on int/float64/string, `_div` is always float (Python `/`),
# `_truthy` treats 0/""/[]/{} as falsy, `_index`/`_setindex`/`_slice` operate on
# []any and map[string]any (dict keys coerced to strings), and `_and`/`_or`/
# `_cond` take thunks to keep short-circuit + operand-return.
#
# compile_go emits a runnable `package main` CLI: it reads a JSON argstate from
# os.Args[1] and prints the JSON of state["return"]. Divergences mirror the JS/
# Lua backends (exception types, division-by-zero, str() of bool/None); a try
# body whose control flow escapes the recover() closure raises GoUnsupportedError.
###############################################################################

class GoUnsupportedError(Exception):
	"""A construct that cannot be faithfully compiled to Go."""




_GO_PRELUDE = """\
func _get(state map[string]any, k string) any { return state[k] }

func _num(v any) float64 {
	switch x := v.(type) {
	case int: return float64(x)
	case float64: return x
	case bool: if x { return 1 }; return 0
	}
	panic("not a number")
}

func _isNum(v any) bool {
	switch v.(type) { case int, float64: return true }
	return false
}

func _truthy(v any) bool {
	switch x := v.(type) {
	case nil: return false
	case bool: return x
	case int: return x != 0
	case float64: return x != 0
	case string: return x != ""
	case []any: return len(x) != 0
	case map[string]any: return len(x) != 0
	}
	return true
}

func _not(v any) any { return !_truthy(v) }
func _neg(v any) any {
	switch x := v.(type) { case int: return -x; case float64: return -x }
	panic("bad - operand")
}

func _add(a, b any) any {
	switch x := a.(type) {
	case int:
		switch y := b.(type) { case int: return x + y; case float64: return float64(x) + y }
	case float64:
		switch y := b.(type) { case int: return x + float64(y); case float64: return x + y }
	case string:
		if y, ok := b.(string); ok { return x + y }
	case []any:
		if y, ok := b.([]any); ok { return append(append([]any{}, x...), y...) }
	}
	panic("bad + operands")
}
func _sub(a, b any) any {
	if xi, ok := a.(int); ok { if yi, ok := b.(int); ok { return xi - yi } }
	return _num(a) - _num(b)
}
func _mul(a, b any) any {
	if xi, ok := a.(int); ok { if yi, ok := b.(int); ok { return xi * yi } }
	return _num(a) * _num(b)
}
func _div(a, b any) any { return _num(a) / _num(b) }
func _mod(a, b any) any {
	if xi, ok := a.(int); ok {
		if yi, ok := b.(int); ok {
			m := xi % yi
			if m != 0 && (m < 0) != (yi < 0) { m += yi }
			return m
		}
	}
	xf, yf := _num(a), _num(b)
	m := math.Mod(xf, yf)
	if m != 0 && (m < 0) != (yf < 0) { m += yf }
	return m
}
func _pow(a, b any) any {
	if xi, ok := a.(int); ok {
		if yi, ok := b.(int); ok && yi >= 0 {
			r := 1
			for i := 0; i < yi; i++ { r *= xi }
			return r
		}
	}
	return math.Pow(_num(a), _num(b))
}

func _cmp(a, b any) int {
	if as, ok := a.(string); ok {
		if bs, ok := b.(string); ok {
			if as < bs { return -1 }
			if as > bs { return 1 }
			return 0
		}
	}
	af, bf := _num(a), _num(b)
	if af < bf { return -1 }
	if af > bf { return 1 }
	return 0
}
func _lt(a, b any) any { return _cmp(a, b) < 0 }
func _le(a, b any) any { return _cmp(a, b) <= 0 }
func _gt(a, b any) any { return _cmp(a, b) > 0 }
func _ge(a, b any) any { return _cmp(a, b) >= 0 }
func _eq(a, b any) any {
	if a == nil || b == nil { return a == nil && b == nil }
	if ab, ok := a.(bool); ok { bb, ok := b.(bool); return ok && ab == bb }
	if _, ok := b.(bool); ok { return false }
	if as, ok := a.(string); ok { bs, ok := b.(string); return ok && as == bs }
	if _, ok := b.(string); ok { return false }
	if _isNum(a) && _isNum(b) { return _num(a) == _num(b) }
	return reflect.DeepEqual(a, b)
}
func _ne(a, b any) any { return !(_eq(a, b).(bool)) }

func _and(a any, bf func() any) any { if _truthy(a) { return bf() }; return a }
func _or(a any, bf func() any) any { if _truthy(a) { return a }; return bf() }
func _cond(c any, af, bf func() any) any { if _truthy(c) { return af() }; return bf() }

func _toInt(v any) int {
	switch x := v.(type) { case int: return x; case float64: return int(x); case bool: if x { return 1 }; return 0 }
	panic("not an index")
}
func _key(k any) string {
	switch x := k.(type) {
	case string: return x
	case int: return strconv.Itoa(x)
	case float64: return strconv.FormatFloat(x, 'g', -1, 64)
	case bool: if x { return "True" }; return "False"
	}
	return fmt.Sprint(k)
}
func _index(obj, idx any) any {
	switch o := obj.(type) {
	case []any: return o[_toInt(idx)]
	case string: return string([]rune(o)[_toInt(idx)])
	case map[string]any: return o[_key(idx)]
	}
	panic("not indexable")
}
func _setindex(obj, idx, val any) {
	switch o := obj.(type) {
	case []any: o[_toInt(idx)] = val
	case map[string]any: o[_key(idx)] = val
	default: panic("not assignable")
	}
}
func _iter(v any) []any {
	switch x := v.(type) {
	case []any: return x
	case string:
		out := []any{}
		for _, r := range x { out = append(out, string(r)) }
		return out
	}
	panic("not iterable")
}
func _range(start, stop, step any) []any {
	st := 1
	if step != nil { st = _toInt(step) }
	a, b := _toInt(start), _toInt(stop)
	out := []any{}
	if st > 0 { for i := a; i < b; i += st { out = append(out, i) } } else { for i := a; i > b; i += st { out = append(out, i) } }
	return out
}
func _slice(seq, loA, hiA, stA any) any {
	step := 1
	if stA != nil { step = _toInt(stA) }
	isStr := false
	var arr []any
	var runes []rune
	n := 0
	switch s := seq.(type) {
	case []any: arr = s; n = len(s)
	case string: isStr = true; runes = []rune(s); n = len(runes)
	default: panic("slice on non-sequence")
	}
	li, hii := 0, 0
	hasLo, hasHi := loA != nil, hiA != nil
	if hasLo { li = _toInt(loA) }
	if hasHi { hii = _toInt(hiA) }
	var lo, hi int
	if step > 0 {
		if !hasLo { lo = 0 } else if li < 0 { lo = max(0, n+li) } else { lo = min(n, li) }
		if !hasHi { hi = n } else if hii < 0 { hi = max(0, n+hii) } else { hi = min(n, hii) }
	} else {
		if !hasLo { lo = n - 1 } else if li < 0 { lo = max(-1, n+li) } else { lo = min(n-1, li) }
		if !hasHi { hi = -1 } else if hii < 0 { hi = max(-1, n+hii) } else { hi = min(n-1, hii) }
	}
	if isStr {
		var sb strings.Builder
		if step > 0 { for i := lo; i < hi; i += step { sb.WriteRune(runes[i]) } } else { for i := lo; i > hi; i += step { sb.WriteRune(runes[i]) } }
		return sb.String()
	}
	out := []any{}
	if step > 0 { for i := lo; i < hi; i += step { out = append(out, arr[i]) } } else { for i := lo; i > hi; i += step { out = append(out, arr[i]) } }
	return out
}
func _str(v any) string {
	switch x := v.(type) {
	case nil: return "None"
	case bool: if x { return "True" }; return "False"
	case string: return x
	case int: return strconv.Itoa(x)
	case float64: return strconv.FormatFloat(x, 'g', -1, 64)
	}
	return fmt.Sprint(v)
}
func _concat(parts ...any) string {
	var sb strings.Builder
	for _, p := range parts { sb.WriteString(_str(p)) }
	return sb.String()
}
func _call(fn any, args ...any) any {
	if f, ok := fn.(func(...any) any); ok { return f(args...) }
	panic("not callable")
}
func _matches(e any, name string) bool {
	if m, ok := e.(map[string]any); ok { return m["name"] == name }
	return false
}
func _try(f func()) (ok bool, err any) {
	defer func() { if r := recover(); r != nil { ok = false; err = r } }()
	f()
	return true, nil
}
"""




_GO_BINOP_HELPER = {
	"add": "_add", "sub": "_sub", "mul": "_mul", "div": "_div", "mod": "_mod",
	"pow": "_pow", "lt": "_lt", "le": "_le", "gt": "_gt", "ge": "_ge",
	"eq": "_eq", "ne": "_ne",
}




def _go_literal(value: Any) -> str:
	if value is None:
		return "nil"
	if value is True:
		return "true"
	if value is False:
		return "false"
	if isinstance(value, str):
		return json.dumps(value)
	if isinstance(value, int):
		return str(value)
	if isinstance(value, float):
		return repr(value)
	raise GoUnsupportedError(f"cannot emit Go literal for {value!r}")




def _go_value(value: Any) -> str:
	if isinstance(value, dict):
		items = ", ".join(f"{json.dumps(str(k))}: {_go_value(v)}" for k, v in value.items())
		return "map[string]any{" + items + "}"
	if isinstance(value, (list, tuple)):
		return "[]any{" + ", ".join(_go_value(v) for v in value) + "}"
	return _go_literal(value)




def _expr_to_go(expr: Expr) -> str:
	if isinstance(expr, LiteralExpr):
		return _go_literal(expr.value)
	if isinstance(expr, VarExpr):
		return f"_get(state, {json.dumps(expr.name)})"
	if isinstance(expr, UnaryExpr):
		inner = _expr_to_go(expr.operand)
		if expr.op == "neg":
			return f"_neg({inner})"
		if expr.op == "pos":
			return inner
		if expr.op == "not":
			return f"_not({inner})"
		raise ValueError(f"Unknown unary op: {expr.op}")
	if isinstance(expr, BinOpExpr):
		if expr.op == "and":
			return f"_and({_expr_to_go(expr.left)}, func() any {{ return {_expr_to_go(expr.right)} }})"
		if expr.op == "or":
			return f"_or({_expr_to_go(expr.left)}, func() any {{ return {_expr_to_go(expr.right)} }})"
		helper = _GO_BINOP_HELPER.get(expr.op)
		if helper is None:
			raise ValueError(f"Unknown binary op: {expr.op}")
		return f"{helper}({_expr_to_go(expr.left)}, {_expr_to_go(expr.right)})"
	if isinstance(expr, IndexExpr):
		if isinstance(expr.index, SliceExpr):
			sl = expr.index
			lo = _expr_to_go(sl.lower) if sl.lower is not None else "nil"
			hi = _expr_to_go(sl.upper) if sl.upper is not None else "nil"
			st = _expr_to_go(sl.step) if sl.step is not None else "nil"
			return f"_slice({_expr_to_go(expr.obj)}, {lo}, {hi}, {st})"
		return f"_index({_expr_to_go(expr.obj)}, {_expr_to_go(expr.index)})"
	if isinstance(expr, CondExpr):
		return (f"_cond({_expr_to_go(expr.cond)}, func() any {{ return {_expr_to_go(expr.then)} }},"
				f" func() any {{ return {_expr_to_go(expr.orelse)} }})")
	if isinstance(expr, ConcatExpr):
		return "_concat(" + ", ".join(_expr_to_go(p) for p in expr.parts) + ")"
	if isinstance(expr, AttrExpr):
		raise GoUnsupportedError("attribute access on a value is not portable to Go")
	if isinstance(expr, ListExpr):
		return "[]any{" + ", ".join(_expr_to_go(e) for e in expr.elts) + "}"
	if isinstance(expr, DictExpr):
		items = ", ".join(f"_key({_expr_to_go(k)}): {_expr_to_go(v)}"
						  for k, v in zip(expr.keys, expr.values))
		return "map[string]any{" + items + "}"
	if isinstance(expr, CallExpr):
		if expr.kwargs:
			raise GoUnsupportedError("keyword arguments are not portable to Go")
		args = ", ".join(_expr_to_go(a) for a in expr.args)
		sep = ", " if args else ""
		return f"_call({_expr_to_go(expr.func)}{sep}{args})"
	if isinstance(expr, RangeExpr):
		start = _expr_to_go(expr.start)
		stop = _expr_to_go(expr.stop)
		step = _expr_to_go(expr.step) if expr.step is not None else "nil"
		return f"_range({start}, {stop}, {step})"
	raise TypeError(f"Unknown Expr type in Go codegen: {expr!r}")




def _gen_go(nodes: List[Node], ctx: _CodegenContext) -> None:
	for node in nodes:
		if isinstance(node, AssignNode):
			ctx.add_line(f"state[{json.dumps(node.name)}] = {_expr_to_go(node.expr)}")

		elif isinstance(node, SetIndexNode):
			ctx.add_line(f"_setindex({_expr_to_go(node.obj)}, {_expr_to_go(node.index)},"
						 f" {_expr_to_go(node.value)})")

		elif isinstance(node, SetAttrNode):
			raise GoUnsupportedError("attribute assignment is not portable to Go")

		elif isinstance(node, ExprNode):
			ctx.add_line(f"_ = {_expr_to_go(node.expr)}")

		elif isinstance(node, ReturnNode):
			ctx.add_line(f"state[\"return\"] = {_expr_to_go(node.expr)}")
			ctx.add_line("return")

		elif isinstance(node, BreakNode):
			ctx.add_line("break")

		elif isinstance(node, ContinueNode):
			ctx.add_line("continue")

		elif isinstance(node, IfNode):
			ctx.add_line("if _truthy(" + _expr_to_go(node.cond) + ") {")
			ctx.indent += 1
			_gen_go(node.body, ctx)
			ctx.indent -= 1
			for e in node.elifs:
				ctx.add_line("} else if _truthy(" + _expr_to_go(e.cond) + ") {")
				ctx.indent += 1
				_gen_go(e.body, ctx)
				ctx.indent -= 1
			if node.else_body is not None:
				ctx.add_line("} else {")
				ctx.indent += 1
				_gen_go(node.else_body, ctx)
				ctx.indent -= 1
			ctx.add_line("}")

		elif isinstance(node, ForNode):
			item = ctx.next_tmp("__item")
			ctx.add_line("for _, " + item + " := range _iter(" + _expr_to_go(node.iterable) + ") {")
			ctx.indent += 1
			ctx.add_line(f"state[{json.dumps(node.var)}] = {item}")
			_gen_go(node.body, ctx)
			ctx.indent -= 1
			ctx.add_line("}")

		elif isinstance(node, WhileNode):
			ctx.add_line("for _truthy(" + _expr_to_go(node.cond) + ") {")
			ctx.indent += 1
			_gen_go(node.body, ctx)
			ctx.indent -= 1
			ctx.add_line("}")

		elif isinstance(node, TryNode):
			if _lua_escapes_flow(node.body):
				raise GoUnsupportedError(
					"try body with break/continue/return that escapes the try "
					"cannot be compiled to Go (recover boundary)")
			ok = ctx.next_tmp("__ok")
			err = ctx.next_tmp("__err")
			ctx.add_line(ok + ", " + err + " := _try(func() {")
			ctx.indent += 1
			_gen_go(node.body, ctx)
			ctx.indent -= 1
			ctx.add_line("})")
			handled = ctx.next_tmp("__handled")
			ctx.add_line("_ = " + err)
			ctx.add_line(handled + " := true")
			if node.excepts:
				ctx.add_line("if !" + ok + " {")
				ctx.indent += 1
				ctx.add_line(handled + " = false")
				for i, ex in enumerate(node.excepts):
					names = ex.exc_type if isinstance(ex.exc_type, tuple) else (ex.exc_type,)
					cond = " || ".join(f"_matches({err}, {json.dumps(t.__name__)})" for t in names)
					prefix = "if " if i == 0 else "} else if "
					ctx.add_line(prefix + cond + " {")
					ctx.indent += 1
					if ex.name is not None:
						ctx.add_line(f"state[{json.dumps(ex.name)}] = {err}")
					_gen_go(ex.body, ctx)
					ctx.add_line(handled + " = true")
					ctx.indent -= 1
				ctx.add_line("}")
				ctx.indent -= 1
				ctx.add_line("}")
			if node.finally_body is not None:
				_gen_go(node.finally_body, ctx)
			ctx.add_line("if !" + ok + " && !" + handled + " { panic(" + err + ") }")

		else:
			raise TypeError(f"Unknown Node type in Go codegen: {node!r}")




def _generate_go_source(root: BlockNode, fn_name: str, prestate: Dict[str, Any]) -> str:
	ctx = _CodegenContext()
	ctx.add_line(f"func {fn_name}(argstate map[string]any) map[string]any {{")
	ctx.indent += 1
	ctx.add_line("state := " + _go_value(dict(prestate)))
	ctx.add_line("for k, v := range argstate { state[k] = v }")
	ctx.add_line('delete(state, "return")')
	ctx.add_line('delete(state, "error")')
	ctx.add_line("__ok, __err := _try(func() {")
	ctx.indent += 1
	_gen_go(root.body, ctx)
	ctx.indent -= 1
	ctx.add_line("})")
	ctx.add_line('if !__ok { state["error"] = __err }')
	ctx.add_line("return state")
	ctx.indent -= 1
	ctx.add_line("}")
	func_src = "\n".join(ctx.lines)

	header = ('package main\n\n'
			  'import (\n'
			  '\t"encoding/json"\n\t"fmt"\n\t"math"\n\t"os"\n'
			  '\t"reflect"\n\t"strconv"\n\t"strings"\n)\n\n')
	main = ('\n\nfunc main() {\n'
			'\tvar argstate map[string]any\n'
			'\tif len(os.Args) > 1 { json.Unmarshal([]byte(os.Args[1]), &argstate) }\n'
			'\t_ = fmt.Sprint\n'
			f'\tst := {fn_name}(argstate)\n'
			'\tb, _ := json.Marshal(st["return"])\n'
			'\tos.Stdout.Write(b)\n'
			'}\n')
	return header + _GO_PRELUDE + "\n" + func_src + main
