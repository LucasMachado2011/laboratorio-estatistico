# Relatório — Laboratório Estatístico Interativo


**Equipe:** Individual · **Componentes:** Lucas Araújo Machado (Matrícula: 72601862

## 1. Dataset e justificativa

- **Nome:** Gorjetas em Restaurantes (Tips Dataset)
- **Fonte original:** [Kaggle - Tips Dataset](https://kaggle.com)
- **Tamanho:** 244 linhas × 7 colunas
- **Variáveis numéricas usadas:** `total_bill` (valor da conta) e `tip` (valor da gorjeta)
- **Variáveis categóricas usadas:** `sex` (gênero), `smoker` (fumante/não fumante) e `time` (almoço/jantar)
- **Por que escolhemos:** Escolhemos este conjunto de dados clássico para investigar se existe uma relação linear forte e positiva entre o valor total gasto em uma mesa de restaurante e a quantia deixada como gorjeta, além de analisar como o comportamento do consumidor varia de acordo com o turno e o perfil do pagador.


## 2. Decisões de tratamento dos dados

<!-- Tratamento de dados é conteúdo, não bastidor. -->

- Valores ausentes: O conjunto original de dados não apresentava valores ausentes (NaN) nas colunas principais. Foi feita uma verificação prévia com métodos de contagem para garantir que nenhuma observação crucial de valor de conta ou gorjeta estivesse vazia, mantendo a integridade de todas as 244 linhas originais.
- Texto onde deveria haver número: As colunas `total_bill` e `tip` foram validadas e forçadas para o tipo numérico utilizando `pd.to_numeric(errors="coerce")` para mitigar eventuais inconsistências ou strings de formatação ocultas (como símbolos de moeda ou espaços vazios), garantindo que nenhuma linha fosse anulada por erro de tipagem.
- Outras decisões: Para as colunas categóricas (`sex`, `smoker`, `day`, `time`), mapeamos e garantimos que os dados textuais estivessem consistentes e sem variações de caixa alta/baixa. Nenhuma linha precisou ser removida, preservando a distribuição original para os testes de simulação e regressão.


## 3. Fórmulas do núcleo (minhastats.py)


| Função | Fórmula | Observação |
| :--- | :--- | :--- |
| média | média = (1/n) · Σ xᵢ | Caso vazio: levanta ValueError (divisão por zero). |
| variância amostral | s² = Σ(xᵢ − média)² / (n − 1) | Usa n − 1 (Correção de Bessel) para corrigir o viés amostral. |
| variância populacional | σ² = Σ(xᵢ − média)² / n | Divide pelo número total de elementos (n). |
| desvio padrão | s = √s² | Expressa a variabilidade na mesma unidade dos dados originais. |
| mediana | Valor central (ímpar) ou média dos centrais (par) | Requer que os dados estejam ordenados (rol). |
| moda | Valor ou valores com maior frequência absoluta | Pode ser amodal, unimodal, bimodal ou multimodal. |
| amplitude | R = x_max − x_min | Diferença direta entre o maior e o menor valor do conjunto. |
| percentil | P_p = posição p·(n−1)/100 | Calculado usando convenção com interpolação linear. |
| coeficiente de variação | CV = (s / média) · 100 | Média zero: levanta ValueError (indefinido). |
| covariância | cov = Σ(xᵢ − mx)(yᵢ − my) / (n − 1) | Mede a direção da relação linear entre duas variáveis. |
| correlação de Pearson | r = cov(x,y) / (s_x · s_y) | Variável constante: levanta ValueError. Varia de -1 a 1. |
| regressão (b₀, b₁, R²) | y = b₀ + b₁·x \| R² = 1 − (SQ_res / SQ_tot) | b₁ é a inclinação, b₀ o intercepto e R² o ajuste. |



## 4. Tabela de validação


| Função | Referência | Diferença observada | Tolerância | Passou? |
| :--- | :--- | :--- | :--- | :--- |
| media | np.mean | 0.0 | 1e-9 (relativa) | Sim |
| variancia (amostral) | np.var(ddof=1) | 0.0 | 1e-9 | Sim |
| variancia (populacional) | np.var(ddof=0) | 0.0 | 1e-9 | Sim |
| mediana | np.median | 0.0 | 1e-9 | Sim |
| percentil | np.percentile | 0.0 | 1e-6 | Sim |
| covariancia | np.cov(ddof=1) | 0.0 | 1e-9 | Sim |
| correlacao | np.corrcoef | 0.0 | 1e-9 | Sim |
| regressao_linear | scipy.stats.linregress | 0.0 | 1e-9 | Sim |

Saída do `pytest -v`:
```text
test_minhastats.py ........................................                               [100%]
====================================== 40 passed in 1.56s =======================================
```


## 5. Os módulos

### Módulo 2 — Descritiva interativa
A análise descritiva da variável `total_bill` (valor da conta) revelou uma média superior à mediana, indicando uma clara assimetria à direita. A maior parte dos clientes consome valores concentrados em faixas menores e intermediárias, enquanto algumas poucas mesas com contas muito elevadas esticam a cauda superior da distribuição. Pela regra do IQR aplicada ao módulo, foram identificados alguns outliers isolados na cauda direita, representando consumos excepcionalmente altos fora do padrão usual do estabelecimento.

### Módulo 3 — Simulação (LGN e TCL)
Ao simular médias amostrais do valor da conta com tamanhos de amostra crescentes (\(n = 2\), \(n = 10\) e \(n = 30\)), observamos o Teorema Central do Limite e a Lei dos Grandes Números em ação. Para \(n = 2\), a distribuição das médias apresentou forte oscilação e manteve o comportamento assimétrico original. À medida que o tamanho amostral avançou para \(n = 10\) e atingiu \(n = 30\), a distribuição das médias amostrais estabilizou-se em um formato perfeitamente simétrico e em formato de sino (Curva Normal), convergindo diretamente para a média populacional real dos dados.

### Módulo 4 — Distribuições teóricas
Ao tentar ajustar a distribuição teórica Normal aos dados de contas (`total_bill`), o ajuste não se mostrou perfeitamente alinhado devido à forte assimetria à direita e à impossibilidade de valores negativos nos gastos. Uma distribuição Gamma ou Log-Normal descreveria muito melhor o comportamento dessa variável, uma vez que tais modelos estatísticos são naturalmente limitados a valores positivos e conseguem modelar com precisão a cauda longa de gastos mais elevados.

### Módulo 5 — Correlação e regressão
A modelagem de regressão linear para prever o valor da gorjeta (`tip`) com base no valor da conta (`total_bill`) resultou em uma inclinação positiva de \(b_1 \approx 0.105\) (indicando que a gorjeta cresce em média cerca de 10,5% a cada unidade monetária gasta na conta) e um coeficiente de determinação \(R^2 \approx 0.45\). O ajuste mostra que embora a conta explique quase metade da variação da gorjeta, o restante depende de fatores não lineares ou externos (como a qualidade do serviço prestado). Um exemplo de causalidade duvidosa seria assumir cegamente que o aumento forçado dos preços do menu magicamente aumentaria de forma idêntica a satisfação dos clientes e sua propensão a dar gorjetas espontâneas.


## 6. As três descobertas

### Descoberta 1 — Assimetria Positiva nos Gastos dos Clientes
- **Afirmação:** A distribuição do valor total das contas (`total_bill`) não é simétrica, apresentando uma cauda alongada para a direita (valores altos).
- **Evidência:** O cálculo automático indicou uma assimetria positiva, onde a média dos gastos é visivelmente maior do que a mediana, puxada por poucas mesas que consomem valores muito acima do padrão.
- **Limite honesto:** Esta descoberta reflete o comportamento apenas deste restaurante específico durante o período de coleta e não garante o mesmo padrão de consumo em estabelecimentos de perfis muito distintos (como redes de fast-food).

### Descoberta 2 — O Poder Explanatório do Valor da Conta sobre a Gorjeta
- **Afirmação:** O valor da conta é o principal fator linear associado ao tamanho da gorjeta, apresentando uma correlação positiva e moderada.
- **Evidência:** O modelo de regressão linear obteve um \(R^2 \approx 0.45\), provando estatisticamente que cerca de 45% da variação nos valores das gorjetas é diretamente explicada pelo tamanho da conta final da mesa.
- **Limite honesto:** Os outros 55% da variação das gorjetas dependem de fatores subjetivos que não foram mensurados neste banco de dados, como a simpatia do atendente, a qualidade da comida ou o humor do cliente.

### Descoberta 3 — Validade da convergência pelo Teorema Central do Limite
- **Afirmação:** Mesmo que a variável original de consumo seja altamente assimétrica e não siga uma curva normal, a distribuição de suas médias amostrais se torna perfeitamente normal a partir de amostras de tamanho \(n = 30\).
- **Evidência:** Nas simulações visuais do Módulo 3, a curva de frequências das médias se transformou em um formato de sino perfeitamente simétrico quando elevamos o tamanho de amostra de 2 para 30 observações.
- **Limite honesto:** A convergência garante a normalidade para a estimativa da média, mas não transforma a natureza dos dados individuais brutos do restaurante, que continuam seguindo seu padrão assimétrico original.

## 7. Divisão do trabalho

| Componente | O que fez | Commits (aprox.) |
|------------|-----------|------------------|
| Lucas Araújo Machado | Implementação completa do núcleo (`minhastats.py`), tratamento dos dados, análise estatística interativa, simulações de distribuição e redação completa deste relatório. | 100% |
