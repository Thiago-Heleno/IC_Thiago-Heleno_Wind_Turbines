---
name: run-notebook
description: Roda, monitora e diagnostica scripts Python/notebooks na workstation. Cobre: verificar se está executando, tempo decorrido/restante, uso de GPU/CPU, escolha de GPU com menor carga, limitar cores da CPU, e executar em background com conda theleno.
argument-hint: "[status|run|gpu] [script.py]"
---

# Skill: run-notebook

Gerencie a execução de scripts Python/notebooks na workstation com controle de recursos.
O ambiente conda padrão é **`theleno`**.

---

## 1. Verificar se um script está rodando

```bash
ps aux | grep python | grep -v grep
```

Para ver detalhes de um PID específico (tempo rodando, CPU, RAM, estado):
```bash
ps -p <PID> -o pid,start,etime,pcpu,pmem,stat --no-headers
```

Exemplo de saída:
```
1567153  02:19:52    11:06:37  134  21.7  Rl
         ^início     ^tempo    ^CPU ^RAM  ^estado
```

---

## 2. Estimar tempo restante (Optuna)

Ler o log para calcular progresso:

```bash
# Contar trials completos
grep "Trial .* finished" notebooks/<notebook>.txt | wc -l

# Ver último trial e horário
grep "Trial .* finished" notebooks/<notebook>.txt | tail -5

# Ver o total de trials configurado no script
grep "N_OPTUNA_TRIALS_AE" notebooks/<notebook>.py | head -5
```

**Notebooks e seus logs/scripts (convenção de nomenclatura):**
| Notebook | Script | Log de treino | Resultados |
|----------|--------|--------------|-----------|
| `wind_turbine_cnn_lstm_paper.ipynb` | `wind_turbine_cnn_lstm_paper.py` | `wind_turbine_cnn_lstm_paper.txt` | `resultados/01_cnn_lstm_supervisionado/` |
| `wind_turbine_anomaly_detection_v4.ipynb` | `wind_turbine_anomaly_detection_v4.py` | `wind_turbine_anomaly_detection_v4.txt` | `resultados/02_cnn_bilstm_autoencoder/` |
| `wind_turbine_autoencoder_keras_pipeline.ipynb` | `wind_turbine_autoencoder_keras_pipeline.py` | `wind_turbine_autoencoder_keras_pipeline.txt` | `resultados/03_keras_mlp_autoencoder/` |

Todos os arquivos ficam dentro de `notebooks/` (exceto os resultados).

**Cálculo manual:**
- `trials_completos / total_trials` = progresso
- Tempo médio por trial = `tempo_total / trials_completos`
- Tempo restante ≈ `(total - completos) × média_por_trial`

---

## 3. Acompanhar log em tempo real

```bash
tail -f notebooks/<notebook>.txt
# ou as últimas N linhas:
tail -50 notebooks/<notebook>.txt
```

Exemplo para o Keras pipeline:
```bash
tail -f notebooks/wind_turbine_autoencoder_keras_pipeline.txt
```

---

## 4. Escolher GPU com menor uso

```bash
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total \
  --format=csv,noheader,nounits
```

Saída (exemplo):
```
0, NVIDIA RTX 3090, 95, 20480, 24576   ← ocupada
1, NVIDIA RTX 3090,  2,  1024, 24576   ← livre ✓
```

Escolher a GPU com menor `utilization.gpu` e menor `memory.used`.
Setar no comando via `CUDA_VISIBLE_DEVICES=<índice>`.

Ver uso detalhado em tempo real:
```bash
watch -n 2 nvidia-smi
```

---

## 5. Limitar cores da CPU

### 4 cores (recomendado para scripts longos em background)
```
taskset -c 0-3
OMP_NUM_THREADS=4  MKL_NUM_THREADS=4  OPENBLAS_NUM_THREADS=4
NUMEXPR_NUM_THREADS=4  VECLIB_MAXIMUM_THREADS=4
```

### 6 cores (quando precisar de mais throughput)
```
taskset -c 0-5
OMP_NUM_THREADS=6  MKL_NUM_THREADS=6  OPENBLAS_NUM_THREADS=6
NUMEXPR_NUM_THREADS=6  VECLIB_MAXIMUM_THREADS=6
```

---

## 6. Template completo — executar em background

O log deve ter o **mesmo nome do notebook** (convenção do projeto).  
Substitua `<GPU>` (0 ou 1), `<N>` (4 ou 6) e `<notebook>` pelo nome base do notebook:

**4 cores, GPU escolhida:**
```bash
nohup conda run --no-capture-output -n theleno env \
  CUDA_VISIBLE_DEVICES=<GPU> \
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 \
  MPLBACKEND=Agg \
  taskset -c 0-3 \
  python notebooks/<notebook>.py \
  > notebooks/<notebook>.txt 2>&1 &

echo "PID: $!"
```

**6 cores, GPU escolhida:**
```bash
nohup conda run --no-capture-output -n theleno env \
  CUDA_VISIBLE_DEVICES=<GPU> \
  OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 \
  OPENBLAS_NUM_THREADS=6 NUMEXPR_NUM_THREADS=6 VECLIB_MAXIMUM_THREADS=6 \
  MPLBACKEND=Agg \
  taskset -c 0-5 \
  python notebooks/<notebook>.py \
  > notebooks/<notebook>.txt 2>&1 &

echo "PID: $!"
```

**Exemplo para o Keras pipeline:**
```bash
nohup conda run --no-capture-output -n theleno env \
  CUDA_VISIBLE_DEVICES=1 \
  OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 \
  OPENBLAS_NUM_THREADS=6 NUMEXPR_NUM_THREADS=6 VECLIB_MAXIMUM_THREADS=6 \
  MPLBACKEND=Agg \
  taskset -c 0-5 \
  python notebooks/wind_turbine_autoencoder_keras_pipeline.py \
  > notebooks/wind_turbine_autoencoder_keras_pipeline.txt 2>&1 &

echo "PID: $!"
```

> `MPLBACKEND=Agg` evita erros de display em servidores headless.
> `nohup` garante que o processo continue mesmo se a sessão for encerrada.

---

## 7. Instalar pacotes no ambiente theleno

```bash
# via pip (preferido para pacotes Python)
conda run -n theleno pip install <pacote>

# via conda
conda install -n theleno -c conda-forge <pacote>

# verificar pacotes instalados
conda run -n theleno pip list | grep <pacote>
```

---

## 8. Diagnóstico rápido — checklist

Quando o usuário perguntar sobre o status de um script, execute estes passos em ordem:

1. `ps aux | grep python | grep -v grep` — está rodando?
2. Se sim: `ps -p <PID> -o pid,start,etime,pcpu,pmem,stat --no-headers` — detalhes
3. `tail -20 notebooks/<notebook>.txt` — últimas mensagens do log
4. `grep "Trial .* finished" notebooks/<notebook>.txt | wc -l` — progresso Optuna
5. `ls -lh resultados/<notebook>/` — artefatos gerados
6. `nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits` — uso de GPU

**Convenção:** `<notebook>` é sempre o nome base do arquivo `.ipynb` (sem extensão).  
Exemplo: notebook `wind_turbine_autoencoder_keras_pipeline.ipynb` → log em `notebooks/wind_turbine_autoencoder_keras_pipeline.txt` → resultados em `resultados/03_keras_mlp_autoencoder/`.

**Mapa nomeclatura notebook → pasta de resultados:**
- `wind_turbine_cnn_lstm_paper.ipynb` → `resultados/01_cnn_lstm_supervisionado/`
- `wind_turbine_anomaly_detection_v4.ipynb` → `resultados/02_cnn_bilstm_autoencoder/`
- `wind_turbine_autoencoder_keras_pipeline.ipynb` → `resultados/03_keras_mlp_autoencoder/`

Com essas informações, calcule e apresente:
- Status (rodando / parado / erro)
- Tempo decorrido e % de progresso
- Estimativa de tempo restante
- Recursos sendo usados (GPU, CPU, RAM)
- Artefatos já gerados
