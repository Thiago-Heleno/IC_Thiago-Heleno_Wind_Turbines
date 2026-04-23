---
name: sync-notebook-docs
description: Use SEMPRE que um notebook em `notebooks/*.ipynb` (ou seu `.py` exportado) for criado, modificado, renomeado ou executado com novos resultados. Mantem `anotacoes/<nome>.md`, `analise_comparativa.md`, `resultados/<pasta>/README.md`, `resultados/<pasta>/metricas.json`, `resultados/README.md` e `resultados/metricas_consolidadas.csv` em sincronia com o estado atual do notebook. Gatilhos: usuario edita celula de arquitetura/hiperparametros/split/preprocessamento, adiciona novo notebook, renomeia notebook, roda pipeline gerando novos artefatos em `resultados/`, ou altera `save_dir`/`OUTPUT_DIR`/`ARTIFACTS_DIR`.
argument-hint: "<notebook_basename> [--create|--update|--rename old→new]"
---

# Skill: sync-notebook-docs

Mantem a documentacao do projeto alinhada com os notebooks. Qualquer mudanca estrutural em notebook exige atualizacao correspondente em varios arquivos — esta skill garante que nada fique defasado.

## Quando invocar

**Obrigatorio** invocar esta skill ao detectar qualquer um dos eventos abaixo:

| Evento | Acao esperada |
|--------|---------------|
| Novo notebook `notebooks/<novo>.ipynb` criado | Criar `anotacoes/analise_<novo>.md`, criar `resultados/NN_<novo>/` com `README.md` + `metricas.json` skeleton, adicionar linha em `resultados/README.md` + CSV consolidado, adicionar secao em `analise_comparativa.md` e entrada em `README.md` raiz |
| Arquitetura do modelo mudou (camadas, heads, dimensoes) | Atualizar secao `Arquitetura` / `Modelo` em `anotacoes/analise_<nome>.md` + campo `modelo` em `metricas.json` |
| Hiperparametros alterados (epochs, LR, batch, janela, dropout, etc.) | Atualizar `hiperparametros` / `configuracao` em `metricas.json` + bloco `Configuracao Experimental` em `anotacoes/*.md` |
| Split ou preprocessamento mudou | Atualizar secao `Dados e Rotulagem` / `Pre-processamento` em `anotacoes/*.md` + `configuracao.split` em `metricas.json` |
| Novos thresholds / metricas geradas em execucao | Regerar bloco `metricas_teste` / `metricas_CARE` em `metricas.json` + tabela em `resultados/<pasta>/README.md` + linha em `resultados/metricas_consolidadas.csv` |
| Path de saida alterado (`save_dir`, `OUTPUT_DIR`, `ARTIFACTS_DIR`) | Atualizar campo `codigo_fonte`/`artefatos` em `metricas.json` + `Artefatos Salvos` em `anotacoes/*.md` + tabela em `README.md` raiz + `.claude/skills/run-notebook/SKILL.md` |
| Notebook renomeado | Renomear arquivos correspondentes: `anotacoes/analise_<nome>.md`, `resultados/NN_<nome>/`, atualizar todas as referencias no projeto |
| Notebook removido | Remover `anotacoes/analise_<nome>.md`, arquivar `resultados/NN_<nome>/` para `resultados/_arquivados/`, atualizar indices |

## Convencoes obrigatorias

### Nomenclatura

- Notebook: `notebooks/wind_turbine_<tema>.ipynb`
- Script exportado: `notebooks/wind_turbine_<tema>.py` (mesmo basename)
- Log de treino: `notebooks/wind_turbine_<tema>.txt`
- Anotacao tecnica: `anotacoes/analise_<tema>.md`
- Pasta de resultados: `resultados/NN_<slug-descritivo>/` onde `NN` e ordinal (01, 02, 03...) e `<slug>` descreve paradigma+modelo (ex: `01_cnn_lstm_supervisionado`)

### Schema uniforme de `metricas.json`

Ver `resultados/README.md` para referencia canonica. Campos obrigatorios:

```json
{
  "notebook": "basename sem extensao",
  "codigo_fonte": "notebooks/<arquivo>.ipynb",
  "paradigma": "supervisionado | semi-supervisionado",
  "modelo": "descricao curta da arquitetura",
  "dataset": "CARE_To_Compare Wind Farm C",
  "data_execucao": "YYYY-MM-DD ou null",
  "configuracao": {...},
  "hiperparametros": {...},
  "thresholds": {...},
  "metricas_teste": {...},
  "artefatos": {...},
  "notas": [...]
}
```

Se notebook usa metricas CARE benchmark, adicionar `metricas_CARE`. Se ainda nao rodou, `status: "pendente"` e metricas como `null`.

### Estrutura de `anotacoes/analise_<tema>.md`

Secoes minimas (na ordem):

1. Titulo + bloco citacional com link para `resultados/NN_<pasta>/` e `metricas.json`
2. `## Objetivo` — o que o notebook faz e por que
3. `## Dados e Rotulagem` — base, split, janela, rotulos
4. `## Pre-processamento` — clipping, scaler, feature selection
5. `## Arquitetura do Modelo` — camadas, dimensoes, funcao de perda
6. `## Configuracao Experimental` — hiperparametros, epochs, otimizador
7. `## Avaliacao e Metricas` — como mede sucesso
8. `## Artefatos Salvos` — lista de arquivos em `resultados/NN_<pasta>/`
9. `## Pontos Fortes do Pipeline`
10. `## Limites e Cuidados`

## Procedimento de sincronizacao

Quando invocada, executar nesta ordem:

1. **Identificar mudancas** — `git diff` ou comparar estado atual do notebook vs anotacao. Listar: celulas adicionadas/removidas/modificadas que afetam alguma das secoes acima.
2. **Ler arquivos alvo** — `anotacoes/analise_<tema>.md`, `resultados/NN_<pasta>/metricas.json`, `resultados/NN_<pasta>/README.md`.
3. **Propagar mudancas** — aplicar edicoes cirurgicas (nao reescrever secoes inteiras se so um campo mudou). Manter estilo e idioma existentes (portugues sem acentos, seguindo convencao do projeto).
4. **Atualizar indices** — se mudou numero/resultado final, atualizar:
   - `resultados/metricas_consolidadas.csv` (linha do notebook)
   - `resultados/README.md` (tabela comparativa)
   - `analise_comparativa.md` (secao do notebook)
   - `README.md` raiz (mapa notebook→pasta se nome mudou)
5. **Validar consistencia** — rodar `grep -rn "<nome_antigo>\|<path_antigo>" "<PROJECT_ROOT>" --include="*.md" --include="*.py" --include="*.ipynb"` para confirmar zero referencias orfas.
6. **Reportar** — listar arquivos tocados + 1 linha do que mudou em cada.

## Checklist rapido (copiar para TodoWrite ao invocar)

- [ ] Diff identificado: quais secoes afetadas
- [ ] `anotacoes/analise_<tema>.md` atualizado
- [ ] `resultados/NN_<pasta>/metricas.json` atualizado
- [ ] `resultados/NN_<pasta>/README.md` atualizado (se numeros mudaram)
- [ ] `resultados/metricas_consolidadas.csv` atualizado (se metricas mudaram)
- [ ] `resultados/README.md` — tabela comparativa revisada
- [ ] `analise_comparativa.md` — secao do notebook revisada
- [ ] `README.md` raiz — revisado se houve rename/novo/removido
- [ ] `grep` final confirma zero refs orfas

## Exemplos de invocacao

```
# Apos editar arquitetura em notebooks/wind_turbine_anomaly_detection_v4.ipynb
/sync-notebook-docs wind_turbine_anomaly_detection_v4 --update

# Criando novo notebook
/sync-notebook-docs wind_turbine_transformer_v1 --create

# Renomeando
/sync-notebook-docs --rename wind_turbine_anomaly_detection_v4→wind_turbine_anomaly_detection_v5
```

## Regras inegociaveis

- **Nunca** deixar pasta `resultados/` com nome diferente do slug padronizado (`NN_<descritivo>`).
- **Nunca** atualizar apenas a anotacao sem propagar para `metricas.json` (e vice-versa).
- **Nunca** alterar metricas numericas em documentos sem confirmar origem no relatorio do notebook (`results.json`, `care_summary.json`, log `.txt` da execucao).
- **Sempre** usar links markdown relativos para navegacao entre os arquivos da pesquisa.
- Idioma e estilo: portugues sem acentos, fragmentos OK em tabelas, frases diretas. Manter coerencia com demais `anotacoes/*.md`.
