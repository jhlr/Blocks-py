# Blocks-py

Uma pequena **DSL embarcada em Python** que você escreve como código Python normal,
mas que constrói uma **AST** — e essa mesma árvore pode ser **interpretada**,
**compilada em uma função Python**, **exportada como um `.py` autônomo** ou
**compilada para JavaScript**.

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

## Quatro modos, o mesmo programa

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
```

## Front-end: Python de verdade → AST (e daí pra JavaScript)

Além do builder, dá pra **parsear um subset de Python real** direto pra mesma
árvore, via `ast` da stdlib. Como o resultado é um `Block`, todos os backends
funcionam — então isto é **tradução source-to-source Python → JavaScript** de
verdade para o subset suportado.

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
b.compile("soma")            # função Python
print(b.compile_js("soma"))  # o MESMO programa em JavaScript
```

Um `def` de topo vira o bloco (defaults constantes viram prestate); ou passe
statements soltos. **Subset suportado:** assign a nome simples, `+=`, anotações,
`return`, `pass`, `break`/`continue`, `if/elif/else`, `for` (alvo nome simples),
`while`, `try/except/finally`; expressões: constantes, nomes, `+ - not`,
`+ - * / % **`, `and`/`or` (n-ário), comparações (incl. encadeadas com operando
do meio sem efeito colateral), chamadas (posicional + keyword), `range()`,
atributo, subscrito (índice, sem slice), literais list/tuple/dict.

Qualquer coisa fora do subset levanta `UnsupportedSyntaxError` **com número da
linha** — o parser nunca traduz errado calado.

## O que tem dentro

- **Builder simbólico** — atribuição vira `AssignNode`, operadores viram `BinOpExpr`,
  `with b.if_(...)`/`for_`/`while_`/`try_` empilham corpos numa pilha de escopos.
  Os operadores vivem no próprio `Expr`, então **toda** expressão compõe:
  `(b.x + 1) * 2`, `b.items[0]`, `b.obj.attr("campo").call()`, `[b.x, b.y]`,
  `{"k": b.v}`, e `b.total += b.i` (o `+=` cai de graça no `AssignNode`).
- **Front-end de Python real** — `parse_python()` faz *lowering* de um subset de
  Python (via `ast` da stdlib) para a mesma AST; falha alto (com linha) fora do
  subset. É o que habilita a tradução Python → JavaScript.
- **AST tipada** — expressões (`Expr`) e comandos (`Node`) como `dataclass`es com `slots`.
- **Interpretador** — `eval_expr` / `exec_nodes`, com controle de fluxo por exceções
  internas (`BlockReturn`, `LoopBreak`, `LoopContinue`).
- **Geradores de código (Python e JavaScript)** — percorrem a mesma árvore e
  emitem código-fonte indentado. No Python, a semântica do `try/except/finally`
  (rodar o `finally` antes de propagar return/break, **e também quando a exceção
  é tratada**) é preservada; no JavaScript, o `finally` nativo já roda em
  return/break/continue/throw.
- **Fonte única de semântica** — o significado de cada operador vive em UMA
  tabela (`UNARY_OPS` / `BINARY_OPS` / `SHORTCIRCUIT_OPS`,
  `op → (símbolo Python, função, símbolo JS)`) lida pelo interpretador e pelos
  dois geradores. Adicionar um operador é adicionar uma linha. `test_blocks.py`
  é um harness *golden* que roda os mesmos programas nos backends e falha se
  divergirem — inclusive rodando o JS gerado no `node` e comparando com o
  interpretador Python. Garantia anti-drift, agora cross-linguagem.

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

## Testes

```bash
python test_blocks.py   # sem deps. Dois níveis:
                        #  1) AST-level: builder -> interpretador vs compilador Python
                        #  2) source-level: parse_python -> interpretar/compilar
                        #     comparado ao Python REAL executado (exec).
                        # Em ambos, se `node` existir, o JS gerado roda e é
                        # comparado também. Sai != 0 se qualquer backend divergir.
```

## Fora do escopo (por ora)

Definição de função aninhada / closures. É outra categoria de linguagem
(escopo léxico) e inflaria o DSL; para "chamar código de verdade" já existe
`.call()` sobre qualquer callable no `state`.

## Requisitos

Python 3.10+ (usa `dataclass(slots=True)`). Sem dependências externas.

## Licença

MIT — veja [LICENSE](LICENSE).
