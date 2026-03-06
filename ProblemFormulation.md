# Problem Formulation

- [ ] Formulação do problema
  - [ ] representação da solução
  - [ ] função de avaliação
  - [ ] funções de vizinhança/mutação e crossover (nome, pré-condições, efeitos e custo)
  - [ ] restrições

---

# Representação da Solução

## Lista de colocações

### Representação da solução (estado / indivíduo)

- \( S = [p_1, p_2, ..., p_N] \)
- Cada colocação:

  \( p_k = (b_k, r_k, c_k) \)

  - \( b_k \): índice do *building project* (0..B-1)
  - \( r_k, c_k \): coordenadas do canto superior-esquerdo do plano no mapa

---

### Como “ler” a solução (decoding)

- Para cada \( p_k \), traduz-se o plano do projeto \( b_k \) para coordenadas globais:

  - Cada célula ocupada `#` do plano em \( (i,j) \) passa para
    \( (r_k + i, c_k + j) \)
- Mantém-se uma estrutura auxiliar de ocupação **apenas para as células `#`**
  (por exemplo, uma *boolean grid*, *bitset* ou *hash set* de células ocupadas).

---

### Vantagens

- É simples e coincide com o formato de output.
- Mutação/vizinhança (mover, remover, adicionar, trocar projeto) fica natural.
- Permite ter \( N \) variável (importante porque é possível construir 0, 1 ou mais vezes cada projeto).

---

### Nota prática

- Como \( H \) e \( W \) podem ser grandes (até 1000), evita guardar “o tabuleiro completo por edifício”.

  Guarda apenas:

  - a lista \( S \)
  - e uma estrutura global `ocupado#` para validação rápida de colisões

---

## Função de Avaliação (Score / Fitness)

### Definições

Considere uma solução $S$ (lista de colocações $(b,r,c)$).

- Seja $R(S)$ o conjunto de edifícios **residenciais colocados**.
- Seja $U_t(S)$ o conjunto de edifícios **utility colocados** que fornecem o serviço do tipo $t$.
- A **distância entre dois edifícios** é a menor distância Manhattan entre qualquer célula `#` de um e qualquer célula `#` do outro.
- Um tipo de serviço $t$ é **acessível** a um residencial $x$ se existir pelo menos um utility desse tipo a distância $\le D$.

---

### Função

Para cada residencial $x \in R(S)$, com capacidade $\text{cap}(x)$, definimos:

$A(x) = \left|\left\{ t \;:\; \exists\, y \in U_t(S)\ \text{com}\ \text{dist}(x,y)\le D \right\}\right|$

Ou seja: $A(x)$ representa o número de **tipos diferentes de serviços** que o residencial consegue alcançar (cada tipo conta no máximo uma vez).

Então o score total é:

$\text{Score}(S) = \sum_{x \in R(S)} \text{cap}(x)\cdot A(x)$

Isto corresponde à regra: um residencial com capacidade $r$ ganha $r$ pontos por cada tipo de utility acessível (contando cada tipo no máximo uma vez).

---

## Funções de vizinhança / mutação e crossover

Assume-se a representação de solução como uma lista de colocações
$S = [p_1, \dots, p_N]$, onde cada colocação é $p_k = (b_k, r_k, c_k)$.

Em todos os operadores abaixo, uma solução é considerada **válida** se:
(i) todas as células `#` de todos os edifícios estiverem dentro do grid e
(ii) não existirem colisões entre células `#` de edifícios diferentes.

---

### Operador 1 — `ADD` (Inserir edifício)

- **Nome:** `ADD(b, r, c)`
- **Pré-condições:**
  - O edifício $b$ colocado com canto superior-esquerdo em $(r,c)$ está dentro do grid.
  - As células `#` do edifício não colidem com células `#` já ocupadas.
- **Efeitos:**
  - $S \leftarrow S \cup \{(b,r,c)\}$ (adiciona uma nova colocação ao conjunto/lista).
  - Atualiza a estrutura auxiliar `ocupado#` com as novas células `#`.
- **Custo (aprox.):**
  - 1

---

### Operador 2 — `REMOVE` (Remover edifício)

- **Nome:** `REMOVE(k)`
- **Pré-condições:**
  - Existe um edifício colocado com índice $k$ na lista ($1 \le k \le N$).
- **Efeitos:**
  - Remove $p_k$ da solução.
  - Remove da estrutura `ocupado#` as células `#` correspondentes.
- **Custo (aprox.):**
  - 1

---

### Operador 3 — `MOVE` (Mover edifício)

- **Nome:** `MOVE(k, r', c')`
- **Pré-condições:**
  - Existe $p_k = (b_k, r_k, c_k)$ em $S$.
  - O edifício $b_k$ em $(r',c')$ fica dentro do grid.
  - Ao mover, as células `#` do edifício não colidem com outras células `#`
    (descontando as suas próprias células antes do movimento).
- **Efeitos:**
  - Atualiza $p_k \leftarrow (b_k, r', c')$.
  - Atualiza `ocupado#` (desmarca posição antiga e marca a nova).
- **Custo (aprox.):**
  - 1

---

### Operador 5 — `CHANGE_TYPE` (Trocar o projeto do edifício)

- **Nome:** `CHANGE_TYPE(k, b')`
- **Pré-condições:**
  - Existe $p_k = (b_k, r_k, c_k)$ em $S$.
  - O projeto $b'$ colocado em $(r_k,c_k)$ cabe no grid.
  - As células `#` de $b'$ em $(r_k,c_k)$ não colidem com outras células `#`
    (descontando as do edifício antigo).
- **Efeitos:**
  - Atualiza $p_k \leftarrow (b', r_k, c_k)$.
  - Atualiza `ocupado#` removendo `#` antigas e inserindo `#` novas.
- **Custo (aprox.):**
  - 1

---



## Crossover (para Algoritmos Genéticos)

### Crossover 1 — `UNION_CROSSOVER` (união com filtragem)

- **Nome:** `UNION_CROSSOVER(S_1, S_2)`
- **Pré-condições:**
  - $S_1$ e $S_2$ são soluções válidas.
- **Efeitos:**
  - Cria $C$ combinando edifícios de ambos (por exemplo, mistura aleatória de colocações).
  - Percorre as colocações por ordem aleatória e insere no filho apenas as que não colidem (`#`).
- **Custo (aprox.):**
  - 1

### Crossover 2 — `ONE_POINT` (um ponto na lista)

- **Nome:** `ONE_POINT(S_1, S_2, cut)`
- **Pré-condições:**
  - $cut$ é um índice válido.
- **Efeitos:**
  - Filho $C =$ primeiro segmento de $S_1$ + segundo segmento de $S_2$.
  - Aplica-se validação: ignora inserções que causem colisões `#`.
- **Custo (aprox.):**
  - 1

### Mutação pós-crossover (GA)

Após crossover, aplica-se uma mutação simples (com baixa probabilidade), tipicamente uma de:
`SHIFT`, `MOVE`, `ADD`, `REMOVE` ou `CHANGE_TYPE`.

> ### Custo dos Operadores

> Neste problema trata-se de uma tarefa de maximização e não de minimização
> de custo acumulado. Assim, os operadores não possuem custo associado
> no sentido clássico de problemas de procura.

> Todos os operadores têm custo unitário, sendo utilizados apenas para
> gerar novas soluções a partir da solução atual.

> A qualidade de uma solução é exclusivamente determinada pela função
> de avaliação:

> $\text{Score}(S)$

> A função de avaliação é calculada sempre que uma nova solução é gerada,
> permitindo comparar soluções e orientar o processo de otimização.

---

## Restrições do Problema

Uma solução $S$ (lista/conjunto de colocações $(b,r,c)$) é considerada válida
se e só se respeitar todas as **hard constraints** (restrições obrigatórias).
Existem também condições do problema que não invalidam a solução, mas afetam
a pontuação (restrições “soft” ligadas à avaliação).

---

# A) Restrições do Input (Building Projects / Building Plans)

Estas restrições são garantidas pelos dados de entrada, mas fazem parte da
definição formal dos projetos disponíveis.

### A1. Estrutura do building plan

Cada projeto $b$ possui um plano retangular com dimensões $h_b \times w_b$,
onde cada célula é:

- `#` (ocupada)
- `.` (livre)

### A2. Ocupação em todas as margens (edge-occupied)

Em cada building plan existe pelo menos uma célula ocupada `#` em **cada uma**
das quatro margens do retângulo:

- na linha superior ($r=0$)
- na linha inferior ($r=h_b-1$)
- na coluna esquerda ($c=0$)
- na coluna direita ($c=w_b-1$)

### A3. Conectividade das células ocupadas

As células `#` de cada building plan formam **uma única componente conexa**,
considerando vizinhança 4 (cima, baixo, esquerda, direita).

### A4. Ausência de “buracos”

Não existem “holes” no interior do plano: todas as células `.` são alcançáveis
a partir da fronteira do plano usando vizinhança 4.

### A5. Orientação fixa

Building plans **não podem ser rodados nem espelhados**; apenas se decide a
sua posição $(r,c)$ no city plan.

---

# B) Restrições de Colocação no City Plan (Hard Constraints)

### B1. Índice de projeto válido

Para cada colocação $(b,r,c)\in S$:

- $0 \le b < B$

### B2. Limites do grid (fit within the city plan)

Todas as células do retângulo do plano do edifício (não só as `#`) devem caber
dentro do city plan $H \times W$:

- $0 \le r \le H - h_b$
- $0 \le c \le W - w_b$

### B3. Não sobreposição de células ocupadas `#`

Uma célula do city plan é considerada ocupada se algum edifício a cobre com `#`.
Cada célula do city plan pode estar ocupada por **no máximo um** edifício, i.e.,
não pode haver sobreposição de `#` entre edifícios distintos. :

**Observação:** células `.` podem sobrepor-se a qualquer coisa (`.` ou `#`).

### B4. Multiplicidade de projetos

Cada projeto pode ser construído 0, 1 ou mais vezes (não há limite por tipo).

---

# C) Restrições Ligadas à Avaliação (Soft / não invalidam a solução)

### C1. Distância de acesso a utilities

Um tipo de utility é acessível a um edifício residencial se existir um utility
desse tipo cuja distância ao residencial seja $\le D$.

A distância entre dois edifícios $A$ e $B$ é a menor distância Manhattan entre
qualquer célula `#` de $A$ e qualquer célula `#` de $B$:
$\text{dist}(A,B) = \min_{a \in A_{\#}, b \in B_{\#}} (|a_r-b_r| + |a_c-b_c|)$ {index=11}

### C2. Tipos de utility contam no máximo uma vez

Para um residencial, múltiplos utilities do mesmo tipo não acumulam pontos:
o tipo conta 0 ou 1 vez para esse residencial.
