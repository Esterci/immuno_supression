# Experimentos de modelos de regressão

O runner usa `lym` como alvo e, por padrão, treina Random Forest, Gradient
Boosting e AdaBoost com `data/clean_processed.parquet`.

## Formatação das entradas

`prepare_measurement_features`, em `runners/run_ml_experiment.py`, cria uma linha
para cada medida observada de `lym`. As entradas são:

- `time` da medida do alvo;
- variáveis não temporais definidas em `NON_TEMPORAL_COLUMNS`;
- `initial_<variável>`: primeira observação temporal disponível até aquele tempo;
- `val_<variável>`: observação temporal mais recente, quando há ao menos duas;
- `d_time_<variável>`: intervalo entre o alvo e a observação mais recente;
- `tend_<variável>`: inclinação entre as duas observações mais recentes.

Somente observações do mesmo paciente com `time` estritamente menor que o tempo
do alvo são elegíveis. `lym` não gera atributos históricos, evitando que o valor do alvo
entre nas features. `id` também não entra no modelo: ele é preservado apenas para
separar os grupos.

Se qualquer feature temporal gerada estiver ausente, a medida de `lym`
correspondente é considerada não previsível e removida antes da divisão.
O MinMax de todas as features é ajustado somente nas linhas de treino de cada fold. O alvo
`lym` também recebe MinMax durante o ajuste, e as previsões são automaticamente
devolvidas à escala original. Métricas e plots incluem somente medidas previsíveis.

O modo padrão `VARMODE=all` inclui as variáveis não temporais. Com
`VARMODE=time-series`, o runner usa `full_processed.parquet` e oferece ao modelo
somente `time` e os atributos derivados das séries temporais. Ausências em
variáveis não temporais, como `cisq`, não removem pacientes nesse modo.

## Divisão dos dados

O teste é reservado por paciente com `GroupShuffleSplit`. A seleção de
hiperparâmetros usa `GroupKFold` somente nos pacientes de treino. Assim, o mesmo
paciente nunca aparece simultaneamente em treino e teste, nem nos lados de treino
e validação de um fold. As divisões ficam registradas em `splits.csv`.

## Execução

```bash
conda env update -f transplant_ml.yml
conda activate transplant_ml

# Estudo padrão: três modelos, cinco folds e busca aleatória.
bash shells/run_ml_experiments.sh

# Somente séries temporais, sem exigir variáveis clínicas não temporais.
VARMODE=time-series bash shells/run_ml_experiments.sh

# Regenerar os dados mantendo mais de uma ocasião antes do merge.
OCC="1 2 3" PREPARE_DATA=1 bash shells/run_ml_experiments.sh

# Validação rápida.
PREPARE_DATA=0 FOLDS=2 N_ITER=1 N_JOBS=1 \
  bash shells/run_ml_experiments.sh --quick --shap-samples 4 --shap-background 5

# Execução direta.
python -m runners.run_ml_experiment \
  --models random_forest --search random --folds 5 --n-iter 30 \
  --test-size 0.2 --normalization all --shap-samples 100
```

O shell aceita `OCC`, `VARMODE`, `SEARCHES`, `FOLDS`, `SEEDS`, `N_ITER`, `N_JOBS`, `OUTPUT_DIR`,
`PREPARE_DATA` e argumentos adicionais do runner. `--quick` usa uma grade pequena
apenas para validar o fluxo.

## Saídas

Cada experimento registra:

- `metadata.json`, hashes das fontes e configuração completa;
- `input_summary.csv` com as features formatadas e o alvo;
- `splits.csv` para auditoria por paciente;
- métricas, parâmetros, resultados de CV, previsões e modelo serializado;
- `model_comparison_rmse.png`, `model_comparison_mae.png` e
  `model_comparison_mape.png`, em figuras separadas;
- gráficos e tabelas SHAP;
- análise temporal por paciente.

Os gráficos temporais exibem todos os pontos originais de `lym` dos pacientes de
teste e sobrepõem as previsões somente nos tempos em que elas foram possíveis.

O `model.joblib` contém normalização, transformação do alvo e o modelo.
Para previsão, forneça as features no formato e na ordem registrados em
`metadata.json`; não normalize os valores previamente.
