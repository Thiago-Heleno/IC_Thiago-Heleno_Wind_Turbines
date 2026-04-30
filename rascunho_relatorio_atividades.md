# Rascunho — Relatório de Atividades de Bolsas (FUSP)

> Preencha cada bloco no .docx. Texto pode ser ajustado conforme situação real.
> Marque [REVISAR] onde precisar conferir/preencher.

---

## Cabeçalho

- **Projeto nº:** [REVISAR — número FUSP]
- **Data:** [data de submissão]
- **Tipo:** [X] Parcial  [ ] Final
- **Título do Projeto:** Predição de Falhas em Turbinas Eólicas Utilizando Modelos de Aprendizado Profundo e Dados Meteorológicos Exógenos
- **Nome do(a) Bolsista:** Thiago Solé Gomes Heleno
- **Modalidade da Bolsa:** Iniciação Científica
- **Nível:** Graduação
- **Duração:** Início 01/06/2025 — Término 31/05/2026
- **Unidade:** Instituto de Ciência e Tecnologia (ICT) — UNIFESP
- **Departamento:** [REVISAR — Departamento de Ciência e Tecnologia / DCT]
- **Laboratório/Núcleo:** Grupo de Pesquisa do Prof. Marcos G. Quiles
- **Coordenador(a) do Projeto:** [REVISAR — confirmar coordenador FUSP]
- **Orientador(a):** Prof. Dr. Marcos G. Quiles (Coorientador: Prof. Dr. Mateus Giesbrecht — FEEC/Unicamp)
- **Período das atividades desenvolvidas:** de 01/06/2025 a [data atual do relatório]

---

## 1. Principais objetivos iniciais do Plano de Pesquisa

Desenvolver e comparar modelos preditivos de falhas em turbinas eólicas
combinando dados operacionais SCADA com variáveis meteorológicas exógenas
(velocidade do vento, temperatura, umidade, pressão). Investigar arquiteturas
de aprendizado profundo (LSTM, CNN1D, Transformer) na detecção precoce de
anomalias em séries temporais. Avaliar o impacto da inclusão de dados
meteorológicos no desempenho dos modelos por meio de métricas de classificação
(Acurácia, Precisão, Revocação, F1-Score, AUC-ROC) e erro de previsão (RMSE,
MAE).

Objetivos formativos: capacitar o aluno em aprendizado de máquina, análise de
séries temporais, programação científica em Python (Pandas, Scikit-learn,
PyTorch/TensorFlow) e escrita científica.

---

## 2. Objetivos alcançados pela pesquisa até a presente data

**Estudo teórico:**

- Estudo dos fundamentos de aprendizado de máquina (regressão, classificação,
  métricas de avaliação) e aprofundamento em redes neurais profundas para
  séries temporais (LSTM, CNN1D, mecanismos de atenção). Acompanhamento de
  curso externo de redes neurais com exercícios práticos (PyTorch, NumPy,
  Pandas — material em `aprendizado_curso/`).
- Levantamento e análise da literatura sobre detecção de falhas em turbinas
  eólicas com dados SCADA, com destaque para os trabalhos de Qi et al.
  (Energies 2024) e o framework CARE (Tautz-Weinert et al. /
  EnergyFaultDetector).

**Implementação técnica (5 notebooks):**

1. **NB1 --- `wind_turbine_cnn_lstm_paper.ipynb`** (supervisionado) ---
   replicação de Qi et al. (Energies 2024). Comparação CNN, LSTM e CNN-LSTM
   sobre 285 features selecionadas via XGBoost.
2. **NB2 --- `wind_turbine_anomaly_detection_v4.ipynb`** (semi-supervisionado,
   PyTorch) --- autoencoder CNN-BiLSTM-Attention treinado apenas em dados
   normais, com 5 estratégias pós-hoc (pesos AUROC, suavização temporal,
   máscara informativa, XGBoost pós-hoc, threshold Beta-F1).
3. **NB3 --- `wind_turbine_autoencoder_keras_pipeline.ipynb`**
   (semi-supervisionado, Keras + Optuna) --- autoencoder MLP simétrico com
   hiperparâmetros otimizados via Optuna (100 trials), threshold adaptativo.
4. **NB4 --- `wind_turbine_classifier_induced_failure.ipynb`**
   (supervisionado) --- MLP 3-class com data augmentation via injeção de
   falhas sintéticas (7\% do treino).
5. **NB5 --- `wind_turbine_autoencoder_induced_failure.ipynb`**
   (semi-supervisionado) --- AE MLP com calibração de threshold via falhas
   induzidas, comparando Standard P95, Induced P95 e Adaptive.

**Atribuições importantes:**

- Os módulos **CARE Score** e **ARCANA** utilizados nos notebooks 3, 4 e 5
  foram **adotados/replicados do repositório público EnergyFaultDetector**
  (https://github.com/AEFDI/EnergyFaultDetector) e não constituem
  contribuição original deste trabalho. Foram utilizados como ferramentas
  padronizadas de avaliação.
- A **arquitetura Transformer completa** prevista no plano original
  **não foi implementada**. Foi utilizada apenas uma **camada de atenção
  (MultiHeadAttention)** integrada ao autoencoder CNN-BiLSTM-Attention (NB2).

**Limitação metodológica identificada:** o dataset CARE_To_Compare anonimiza
completamente a localização das turbinas (apenas identificadores Wind Farm
A/B/C, sem coordenadas geográficas reais), inviabilizando a sincronização
com bases meteorológicas externas (CPTEC/INPE, OpenWeatherMap). A comparação
"modelos com vs sem dados exógenos" prevista no plano original **não pôde
ser executada**. O escopo foi readequado para focar na comparação entre
paradigmas de detecção (supervisionado vs semi-supervisionado) e na
exploração de técnicas alternativas (falhas induzidas como data
augmentation).

---

## 3. Principais resultados alcançados

**Resultados quantitativos consolidados** (fonte:
`resultados/metricas_consolidadas.csv`):

| Notebook | Modelo | F1 (amostra) | AUC-ROC | CARE | Recall (evento) |
|----------|--------|---|---|---|---|
| NB1 | CNN | 5,47% | 0,55 | --- | --- |
| NB1 | CNN-LSTM | 3,16% | 0,66 | --- | --- |
| NB2 | CNN-BiLSTM-Attention AE (P99) | 5,78% | 0,865 | --- | 96,4% |
| NB2 | v4 Score Pond. + Beta-F1 | 4,52% | **0,872** | --- | --- |
| NB3 | Dense MLP AE + Optuna | --- | 0,872 | **0,699** | **94,1%** |
| NB4 | MLP 3-class Induzido | **67,4%** | --- | 3.6e-6 | --- |
| NB5 | MLP AE + Induced (P95) | 12,9% | --- | 0,545 | --- |

Principais conclusões: o paradigma semi-supervisionado (autoencoders) supera
o supervisionado direto em datasets severamente desbalanceados (98:2);
o LSTM isolado colapsa no teste (F1=0) sem mecanismos robustos de
balanceamento; a injeção de falhas sintéticas (NB4) eleva F1 amostral de
50,5\% (baseline) para 67,4\%; o AE Keras (NB3) atinge 94,1\% de sensibilidade
por evento, melhor resultado do projeto.

**Artefatos técnicos produzidos:**

- 5 notebooks (`.ipynb` + scripts `.py` + logs `.txt`) em `notebooks/`
- 5 pastas de resultados padronizados em `resultados/0X_*/` com
  `metricas.json`, `README.md`, modelos serializados e gráficos em PNG
- Comparação agregada em `resultados/metricas_consolidadas.csv`
- Análise comparativa integrada em `analise_comparativa.md`
- Anotações técnicas individuais em `anotacoes/analise_*.md`

**Produção acadêmica em andamento:**

- Relatório técnico parcial em LaTeX (ABNT NBR 14724) em redação,
  estruturado em 18 seções + 3 apêndices (`relatorio_ic.tex` /
  `relatorio_ic.pdf`).
- Slides de apresentação em `slides/`.
- [REVISAR — adicionar apresentações em seminários, reuniões com
  orientador/coorientador, eventuais submissões a eventos]

**Formação:**

- Aprofundamento em PyTorch, TensorFlow/Keras, Optuna, Pandas, NumPy,
  Scikit-learn, XGBoost.
- Participação contínua no grupo de pesquisa do Prof. Marcos Quiles desde
  agosto/2024.
- Curso externo de redes neurais acompanhado em paralelo
  (`aprendizado_curso/`).
- Disciplinas cursadas relevantes: [REVISAR — listar matérias do semestre].

---

## 4. Impacto do projeto junto à área acadêmica

O projeto contribui para a manutenção preditiva no setor de energia eólica,
área estratégica na transição energética brasileira. A pesquisa fornece
evidência empírica sobre o desempenho de cinco arquiteturas distintas
(CNN, LSTM, CNN-LSTM, CNN-BiLSTM-Attention AE, MLP AE) em séries temporais
SCADA, além de documentar uma técnica complementar (falhas induzidas como
data augmentation) que mostrou ganho consistente de F1 sobre baseline.
A pesquisa também documenta limitações práticas relevantes — particularmente
a dificuldade de integração de dados meteorológicos quando datasets públicos
anonimizam a localização das turbinas — informação útil para futuros
trabalhos na área.

Para o bolsista, o projeto consolida formação em ciência de dados,
aprendizado profundo (PyTorch e TensorFlow), otimização de hiperparâmetros
(Optuna) e escrita científica, abrindo perspectivas de continuidade em
pós-graduação. Para o grupo de pesquisa do Prof. Quiles e em colaboração
com o Prof. Giesbrecht (FEEC/Unicamp), o trabalho fortalece a linha de
aplicação de aprendizado de máquina em sistemas de engenharia, com
potencial de gerar publicação em conferência ou periódico ao final da
vigência.

---

## 5. Outras metas a serem atingidas até o término da bolsa

(Preencher apenas se Relatório Parcial)

- Investigar **ensemble** entre NB3 (Keras AE adaptativo, alto recall) e
  NB5 (Induced calibration, alta precision) — pontos opostos da curva ROC
  sugerem complementaridade.
- Explorar **arquitetura Transformer completa** (não realizada até o
  momento), comparando com a camada de atenção já utilizada no NB2.
- Buscar **datasets alternativos** que disponibilizem coordenadas
  geográficas para viabilizar a integração com dados meteorológicos
  conforme proposta original.
- Reduzir **falsos positivos amostrais** (precision ~2-3\% em todos os
  AEs) via threshold por regime operacional ou Focal Loss.
- Finalizar o **relatório técnico completo** em LaTeX/ABNT.
- Preparar **manuscrito de artigo** consolidando os resultados dos 5
  notebooks para submissão a evento ou periódico.

---

## 6. Documentos que fundamentam a pesquisa realizada (anexos)

- Plano de Pesquisa original — `Plano de Pesquisa - Thiago Heleno.pdf`
- Relatório técnico parcial em elaboração (LaTeX/ABNT, mais completo) —
  `relatorio_ic.tex` / `relatorio_ic.pdf`
- Análise comparativa integrada dos 5 notebooks — `analise_comparativa.md`
- Anotações técnicas detalhadas — `anotacoes/analise_*.md`:
  - `analise_cnn_lstm_paper.md`
  - `analise_cnn_bilstm_autoencoder_v4.md`
  - `analise_autoencoder_keras.md`
  - `analise_classifier_induced_failure.md`
  - `analise_autoencoder_induced_failure.md`
- Notebooks de experimentos em `notebooks/`:
  - `wind_turbine_cnn_lstm_paper.ipynb`
  - `wind_turbine_anomaly_detection_v4.ipynb`
  - `wind_turbine_autoencoder_keras_pipeline.ipynb`
  - `wind_turbine_classifier_induced_failure.ipynb`
  - `wind_turbine_autoencoder_induced_failure.ipynb`
- Resultados padronizados em `resultados/0X_*/` (5 subpastas) +
  `resultados/metricas_consolidadas.csv`
- Material de estudo do curso de redes neurais em `aprendizado_curso/`

---

## Outras informações relevantes

**Atribuições e desvios em relação ao plano original (transparência):**

1. Os módulos **CARE Score** e **ARCANA** foram adotados/replicados do
   repositório público EnergyFaultDetector
   (https://github.com/AEFDI/EnergyFaultDetector), não constituindo
   contribuição original deste trabalho.
2. A **arquitetura Transformer completa** prevista no plano original não
   foi implementada. Foi utilizada uma **camada de atenção
   (MultiHeadAttention)** dentro do autoencoder CNN-BiLSTM-Attention (NB2).
3. **Dados meteorológicos exógenos** não foram integrados, pois o dataset
   público (CARE_To_Compare) anonimiza a localização das turbinas. O
   escopo foi readequado para incluir uma técnica adicional não prevista
   originalmente: **injeção de falhas sintéticas como data augmentation**
   (NB4 e NB5).

**Continuidade no grupo de pesquisa:** o bolsista mantém participação ativa
no grupo do Prof. Marcos Quiles desde agosto de 2024 (anterior ao início
formal da bolsa), o que contribuiu para o ramp-up técnico no início do
projeto.
