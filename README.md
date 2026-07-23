# Blocks-py

Uma pequena **DSL embarcada em Python** que você escreve como código Python normal,
mas que constrói uma **AST** — e essa mesma árvore pode ser **interpretada**,
**compilada em uma função Python** ou **exportada como um `.py` autônomo**.

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

O truque: `b.x` devolve um proxy simbólico, e os operadores (`+`, `<`, `==`, ...)
e os *context managers* (`if_`, `for_`, `while_`, `try_`) montam nós da árvore em
vez de rodar. Depois você escolhe o que fazer com ela.

## Três modos, o mesmo programa

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
```

## O que tem dentro

- **Builder simbólico** — atribuição vira `AssignNode`, operadores viram `BinOpExpr`,
  `with b.if_(...)`/`for_`/`while_`/`try_` empilham corpos numa pilha de escopos.
- **AST tipada** — expressões (`Expr`) e comandos (`Node`) como `dataclass`es com `slots`.
- **Interpretador** — `exec_expr` / `exec_nodes`, com controle de fluxo por exceções
  internas (`BlockReturn`, `LoopBreak`, `LoopContinue`).
- **Gerador de código** — percorre a mesma árvore e emite Python indentado; a
  semântica do `try/except/finally` (rodar o `finally` antes de propagar return/break)
  é preservada no código gerado.

## Semântica

- `state = prestate.copy(); state.update(argstate)`.
- Variável não definida lê como `None`.
- `return_(expr)` escreve `state["return"]` e encerra o bloco.
- Exceção não tratada é capturada em `state["error"]`.

## Requisitos

Python 3.10+ (usa `dataclass(slots=True)`). Sem dependências externas.

## Licença

MIT — veja [LICENSE](LICENSE).
