# Rascunho — Relatório de Atividades de Bolsas (FUSP)

> Preencha cada bloco no .docx. Texto pode ser ajustado conforme situação real
> (datas, modelos efetivamente treinados, métricas). Marque [REVISAR] onde
> precisar conferir/preencher.

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

- Estudo concluído dos fundamentos de aprendizado de máquina (regressão,
  classificação, métricas de avaliação) e aprofundamento em redes neurais
  profundas para séries temporais (LSTM, CNN1D, Transformer).
- Levantamento e análise da literatura sobre detecção de falhas em turbinas
  eólicas com dados SCADA e abordagens multimodais.
- Coleta de bases de dados públicas SCADA e implementação do pipeline de
  pré-processamento (limpeza, tratamento de outliers, imputação de faltantes,
  sincronização temporal, normalização).
- Implementação e treinamento dos modelos LSTM, CNN1D e híbrido CNN-LSTM em
  cenário sem dados exógenos (baseline). [REVISAR — incluir Transformer se já
  treinado]
- Avaliação experimental dos modelos com métricas padrão e análise comparativa
  preliminar entre arquiteturas.
- Identificação de limitação metodológica relevante: as bases SCADA públicas
  não divulgam a localização exata dos sítios, inviabilizando a sincronização
  com dados meteorológicos externos (CPTEC/INPE, OpenWeatherMap). Esta
  constatação levou a uma readequação parcial do escopo, focando inicialmente
  em modelos baseados apenas em dados SCADA.

---

## 3. Principais resultados alcançados

**Resultados técnicos:**
- Pipeline reproduzível de pré-processamento de dados SCADA implementado em
  Python (notebooks `wind_turbine_anomaly_detection.ipynb` v1, v2, v3 e
  `wind_turbine_cnn_lstm_paper.ipynb`).
- Modelos treinados e avaliados: [REVISAR — listar arquiteturas e métricas
  efetivas, ex.: "LSTM atingiu F1-Score de 0,XX; CNN-LSTM atingiu 0,YY"].
- Artefatos gerados em `artefatos/`, `artefatos_v2/`, `artefatos_v3/` e
  `results_cnn_lstm_paper/` contendo modelos treinados, gráficos de avaliação
  e tabelas de métricas.

**Produção acadêmica:**
- Relatório técnico parcial em LaTeX (ABNT NBR 14724) sendo redigido,
  estruturado em 18 seções cobrindo fundamentação teórica, metodologia,
  experimentos, problemas com dados, comparação com outras pesquisas e
  resultados. Esse documento mais completo está sendo elaborado em paralelo
  como referência detalhada da pesquisa e está anexado a este relatório
  (item 6).
- [REVISAR — adicionar apresentações em seminários do grupo de pesquisa,
  reuniões com orientador/coorientador, eventuais submissões a eventos]

**Formação:**
- Aprofundamento em PyTorch/TensorFlow, Pandas, NumPy, Scikit-learn.
- Participação contínua no grupo de pesquisa do Prof. Marcos Quiles desde
  agosto/2024.
- Disciplinas cursadas relevantes ao projeto: [REVISAR — listar matérias do
  semestre].

---

## 4. Impacto do projeto junto à área acadêmica

O projeto contribui para a manutenção preditiva no setor de energia eólica,
área estratégica na transição energética brasileira. A pesquisa fornece
evidência empírica sobre o desempenho de arquiteturas de aprendizado profundo
(LSTM, CNN1D, Transformer) em séries temporais SCADA, além de documentar
limitações práticas relevantes — particularmente a dificuldade de integração
de dados meteorológicos quando datasets públicos anonimizam a localização das
turbinas.

Para o bolsista, o projeto consolida formação em ciência de dados, aprendizado
profundo e escrita científica, abrindo perspectivas de continuidade em
pós-graduação. Para o grupo de pesquisa do Prof. Quiles e em colaboração com
o Prof. Giesbrecht (FEEC/Unicamp), o trabalho fortalece a linha de aplicação
de aprendizado de máquina em sistemas de engenharia, com potencial de gerar
publicação em conferência ou periódico ao final da vigência.

---

## 5. Outras metas a serem atingidas até o término da bolsa

(Preencher apenas se Relatório Parcial)

- Concluir treinamento e avaliação do modelo Transformer.
- Investigar estratégias para mitigar o desbalanceamento de classes (focal
  loss, reponderação, oversampling sintético).
- Realizar ablation study para quantificar o impacto de cada componente do
  pipeline.
- Explorar bases de dados alternativas que disponibilizem coordenadas
  geográficas para viabilizar a integração com dados meteorológicos.
- Finalizar o relatório técnico completo em LaTeX/ABNT.
- Preparar manuscrito de artigo para submissão a evento ou periódico.
- [REVISAR — adicionar metas específicas alinhadas com orientador]

---

## 6. Documentos que fundamentam a pesquisa realizada (anexos)

- Plano de Pesquisa original — `Plano de Pesquisa - Thiago Heleno.pdf`
- Relatório técnico parcial em elaboração (LaTeX/ABNT, mais completo) —
  `relatorio_ic.tex` / `relatorio_ic.pdf`
- Notebooks de experimentos:
  - `wind_turbine_anomaly_detection.ipynb`
  - `wind_turbine_anomaly_detection_v2.ipynb`
  - `wind_turbine_anomaly_detection_v3.ipynb`
  - `wind_turbine_cnn_lstm_paper.ipynb`
- Artefatos de treinamento e avaliação: `artefatos/`, `artefatos_v2/`,
  `artefatos_v3/`, `results_cnn_lstm_paper/`
- Referência principal da literatura usada como baseline:
  `energies-17-04497.pdf`

---

## Outras informações relevantes

Durante a execução, foi identificado que datasets SCADA públicos anonimizam a
localização das turbinas, o que limitou a integração com bases meteorológicas
externas conforme proposto inicialmente. A equipe optou por consolidar
primeiro os modelos baseados apenas em dados SCADA, mantendo o objetivo
original de avaliar o impacto de dados exógenos como linha de investigação
para a etapa final, condicionada à obtenção de dataset com geolocalização.

O bolsista mantém participação ativa no grupo de pesquisa do orientador desde
agosto de 2024 (anterior ao início formal da bolsa), o que contribuiu para o
ramp-up técnico no início do projeto.
