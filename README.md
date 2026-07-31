# Blocks-py

Uma pequena **DSL embarcada em Python** que você escreve como código Python normal,
mas que constrói uma **AST** — e essa mesma árvore pode ser **interpretada**,
**compilada em uma função Python**, **exportada como um `.py` autônomo** ou
**compilada para JavaScript e Lua**. E dá pra chegar nessa árvore também
**parseando Python OU JavaScript de verdade** — o que faz do Blocks um pequeno
tradutor source-to-source (Python ↔ JavaScript, e daí pra Lua).

Um arquivo, sem dependências além da biblioteca padrão.

```python
from blocks import Block

# soma 1..n com for / if / return — mas isto NÃO executa ainda,
# monta uma árvore de sintaxe
b = Block({"n": 5})
b.total = 0
with b.for_(b.range(1, b.n + 1), var="i") as loop:
    b.total = b.total + b.i
b.return_(b.total)
```

O truque: `b.x` devolve uma **expressão simbólica** (`VarExpr`), e os operadores
(`+`, `<`, `==`, `%`, `**`, ...) e os *context managers* (`if_`, `for_`,
`while_`, `try_`) montam nós da árvore em vez de rodar. Depois você escolhe o que
fazer com ela.

Se preferir não usar o builder, dá pra **parsear Python de verdade** direto pra
mesma árvore (veja "Front-end" abaixo) — inclusive traduzindo Python → JavaScript.

## Cinco modos, o mesmo programa

```python
# 1) Interpretar (tree-walking)
b({"n": 5})["return"]     # 15
b({"n": 10})["return"]    # 55

# 2) Compilar para uma função Python de verdade
soma, source = b.compile("soma")
soma({"n": 5})["return"]  # 15
print(source)             # o código-fonte Python gerado

# 3) Exportar um módulo .py autônomo
b.export("soma_gerada.py", fn_name="soma")

# 4) Compilar / exportar para JavaScript (mesma AST, backend JS)
js = b.compile_js("soma")          # string com o código JS
b.export_js("soma.js", fn_name="soma")   # módulo CommonJS autônomo (Node/browser)

# 5) Compilar / exportar para Lua (mesma AST, backend Lua)
lua = b.compile_lua("soma")
b.export_lua("soma.lua", fn_name="soma")   # módulo Lua autônomo
```

## Front-ends: Python E JavaScript de verdade → AST

Além do builder, dá pra **parsear um subset de código real** direto pra mesma
árvore. Como o resultado é um `Block`, todos os backends funcionam — então isto é
**tradução source-to-source de verdade** para o subset suportado.

### `parse_python` (via `ast` da stdlib) → Python, JS ou Lua

```python
from blocks import parse_python

b = parse_python('''
def soma(n, base=0):
    total = base
    for i in range(1, n + 1):
        if i % 2 == 0:
            total += i * 2
        else:
            total += i
    return total
''')

b({"n": 5})["return"]        # 21  (interpretado)
print(b.compile_js("soma"))  # o MESMO programa em JavaScript
print(b.compile_lua("soma")) # e em Lua
```

Um `def` de topo vira o bloco (defaults constantes viram prestate); ou passe
statements soltos. **Subset:** assign a nome / subscrito / atributo (`x`, `x[i]`,
`x.a`), `+=` (incl. em subscrito/atributo), anotações, `return`, `pass`,
`break`/`continue`, `if/elif/else`, `for`, `while`, `try/except/finally`;
expressões: constantes, nomes, `+ - not`, `+ - * / % **`, `and`/`or` (n-ário),
comparações (incl. encadeadas sem efeito colateral no meio), chamadas
(posicional + keyword), `range()`, atributo, subscrito, literais list/tuple/dict.

### `parse_js` (tokenizer + parser próprios) → Python, Lua ou JS

Fecha o ciclo na outra direção: **JavaScript → Python**.

```python
from blocks import parse_js

b = parse_js('''
function dictAccess(x) {
  const d = {lo: x, hi: x * 10};
  d["mid"] = x + 5;
  return d.lo + d.hi + d["mid"];
}
''')

b({"x": 3})["return"]         # 41  (interpretado com semântica Python)
print(b.compile("dictAccess")[1])   # o MESMO programa em Python
```

**Subset:** `function` de topo (defaults viram prestate) ou statements;
`let/const/var`, assign a nome/índice/membro, `+=`/`-=`/`*=`/`/=`/`%=`, `++`/`--`,
`return`, `break`/`continue`, `if/else if/else`, `while`, `for-of`, `for` estilo-C
(contável vira `range`; senão desugar pra `while`), `try/catch/finally`; expressões
com precedência completa. **Nota-chave:** `obj.prop` e `obj[k]` do JS são acesso a
propriedade, então mapeiam pra **subscrito** (`obj["prop"]`) — a tradução fiel pra
`dict` do Python. Métodos em valores (`arr.push`, `s.toUpperCase`) não têm
equivalente e não portam.

Fora do subset, ambos levantam erro (`UnsupportedSyntaxError` / `JsSyntaxError`)
— **nunca traduzem errado calado.**

## O que tem dentro

- **Builder simbólico** — atribuição vira `AssignNode`, operadores viram `BinOpExpr`,
  `with b.if_(...)`/`for_`/`while_`/`try_` empilham corpos numa pilha de escopos.
  Os operadores vivem no próprio `Expr`, então **toda** expressão compõe:
  `(b.x + 1) * 2`, `b.items[0]`, `b.obj.attr("campo").call()`, `[b.x, b.y]`,
  `{"k": b.v}`, e `b.total += b.i` (o `+=` cai de graça no `AssignNode`).
- **Dois front-ends** — `parse_python()` (via `ast` da stdlib) e `parse_js()`
  (tokenizer + parser recursivo próprios) fazem *lowering* de um subset de cada
  linguagem para a mesma AST; falham alto (com linha/token) fora do subset. É o
  que habilita a tradução Python ↔ JavaScript → Lua.
- **AST tipada** — expressões (`Expr`) e comandos (`Node`) como `dataclass`es com `slots`.
- **Interpretador** — `eval_expr` / `exec_nodes`, com controle de fluxo por exceções
  internas (`BlockReturn`, `LoopBreak`, `LoopContinue`).
- **Três geradores de código (Python, JavaScript, Lua)** — percorrem a mesma
  árvore e emitem código-fonte indentado, preservando a semântica do
  `try/except/finally` (o `finally` roda antes de propagar return/break e também
  quando a exceção é tratada; em JS/Lua isso cai no `finally`/pcall nativo).
- **Fonte única de semântica** — o significado de cada operador vive em UMA
  tabela (`UNARY_OPS` / `BINARY_OPS` / `SHORTCIRCUIT_OPS`,
  `op → (símbolo Python, função, símbolo JS, símbolo Lua)`) lida pelo
  interpretador e pelos três geradores. Adicionar um operador é adicionar uma
  linha. `test_blocks.py` é um harness *golden* que roda os mesmos programas nos
  backends e falha se divergirem — inclusive rodando o JS no `node` e o Lua no
  `lua` e comparando com o interpretador Python. Garantia anti-drift
  cross-linguagem.

## Semântica

- `state = prestate.copy(); state.update(argstate)`.
- Variável não definida lê como `None`.
- `return_(expr)` escreve `state["return"]` e encerra o bloco.
- Exceção não tratada é capturada em `state["error"]`.
- **`==`/`!=` e operadores constroem AST, não avaliam.** Como em DSLs de
  expressão (SQLAlchemy etc.), `b.x == 5` devolve um nó de comparação, não um
  `bool`. Os objetos `Expr` são hasheáveis por *identidade* (servem de chave de
  dict / membro de set por identidade), mas não são comparáveis por valor —
  `expr in algum_set` não se comporta como pertinência normal.
- **`and`/`or`/`not`** não dá pra sobrecarregar em Python: use os métodos
  `b.a.and_(b.b)`, `b.a.or_(b.b)`, `b.x.not_()` (short-circuit preservado nos dois
  backends).

## Backend JavaScript

JS é dinâmico como Python, então o mapeamento é quase 1:1. As divergências são
**deliberadas e documentadas** (não gambiarra):

- **Truthiness de `[]` e `{}`** — *truthy* em JS, *falsy* em Python. Como
  `if`/`while`/`and`/`or` mapeiam direto pra JS, programas que dependem de
  container vazio ser falsy vão divergir. `0`, `""` e `null` batem.
- **`==`/`!=` viram `===`/`!==`** — identidade pra objetos, não igualdade
  profunda como em Python.
- **Variável ausente → `null`** (via helper `_get`), espelhando o `None`.
- **Tipos de exceção são específicos da linguagem** — `try/except` vira um
  `try/catch/finally` estrutural que casa pelo `name`/construtor do erro
  (significativo pra erros lançados em JS, não pros do Python). Divisão por zero
  não lança em JS (`Infinity`).
- **Nome de atributo/método** (`b.x.attr("upper")`) e **kwargs** em chamadas são
  específicos do Python; o backend JS levanta erro em kwargs.

O harness só cross-checa contra o `node` os programas marcados como portáveis.

## Backend Lua

Também dinâmico, mas com diferenças que exigem helpers pra manter fidelidade:

- **Truthiness FIEL** — Lua trata `0`, `""`, `[]` e `{}` como *truthy*. Pra bater
  com Python, condições e `and`/`or` passam por `_truthy` (tabela vazia lida como
  falsy). Ou seja, o backend Lua é **mais fiel** que o JS nesse ponto.
- **Sequências 0-indexadas** — listas viram tabelas `{[0]=.., [1]=..}` iteradas
  por `_seq`, driblando o 1-index nativo pra manter `x[0]` do Python.
- **`and`/`or`/`not`** viram `_and`/`_or` (com *thunk* no lado direito, preservando
  short-circuit + retorno do operando) e `_not`; `**` vira `^`; `!=` vira `~=`.
- **`return`** é embrulhado em `do ... end` (return do Lua tem que fechar o bloco);
  o estado propaga por mutação da tabela. **`continue`** usa `goto` a um label por
  loop.
- **`try`** usa `pcall`; um corpo de `try` com break/continue/return que **escapa**
  não compila pra Lua (fronteira do pcall) → `LuaUnsupportedError`. Tipos de
  exceção e divisão por zero seguem específicos da linguagem, como no JS.

## Testes

```bash
python test_blocks.py   # sem deps. Três níveis:
                        #  1) AST-level: builder -> interpretador vs compilador Python
                        #  2) Python-source: parse_python -> interpretar/compilar
                        #     comparado ao Python REAL executado (exec)
                        #  3) JS-source: parse_js -> interpretar/compilar comparado
                        #     ao JavaScript REAL rodado no node
                        # Onde `node`/`lua` existirem, o JS e o Lua gerados também
                        # rodam e são comparados. Sai != 0 se algum backend divergir.
```

Os runtimes `node` (JS) e `lua` (Lua) são **opcionais** — se ausentes, os
cross-checks correspondentes são pulados com aviso; o resto da suíte roda só com
Python da stdlib.

## Fora do escopo (por ora)

Definição de função aninhada / closures. É outra categoria de linguagem
(escopo léxico) e inflaria o DSL; para "chamar código de verdade" já existe
`.call()` sobre qualquer callable no `state`.

## Requisitos

Python 3.10+ (usa `dataclass(slots=True)`). Sem dependências externas. Os
cross-checks de teste usam `node` e `lua` se disponíveis (opcionais).

## Licença

MIT — veja [LICENSE](LICENSE).
