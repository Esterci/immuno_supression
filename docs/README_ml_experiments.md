# Experimentos de modelos de regressão

Estrutura baseada em `Evolutionary_Algorithms/work_4`:

- `methods/ml_models.py`: features, lags, modelos e grades de hiperparâmetros.
- `utils/ml_reports.py`: prints, métricas, gráficos e SHAP.
- `runners/run_ml_experiment.py`: execução, divisão dos dados e metadados.
- `shells/run_ml_experiments.sh`: preparação e execução de combinações.

O notebook original permanece disponível. Random Forest, Gradient Boosting e
AdaBoost usam as grades do notebook; o target é `lym`.

## Ambiente e execução

```bash
conda env update -f transplant_ml.yml
conda activate transplant_ml
bash shells/run_ml_experiments.sh
```

O padrão prepara os dados e roda os três modelos, busca aleatória com 30 candidatos
por modelo, 3 lags, 5 folds, seed 42 e variáveis clínicas mais histórico temporal (`VARMODE=all`).
`PYTHON_BIN` permite escolher o interpretador.
SHAP e joblib estão incluídos no YAML do ambiente.

```bash
# Variáveis clínicas e histórico temporal (padrão atual).
VARMODE=all N_ITER=30 N_JOBS=4 bash shells/run_ml_experiments.sh

# Tempo atual e histórico t0..tN das séries farmacológicas, tempo e LYM.
VARMODE=time-series N_ITER=30 N_JOBS=4 bash shells/run_ml_experiments.sh
```

`VARMODE=time-series` mantém somente os atributos de `t0` até o número de lags
configurado para `amt_tac`, `dv_tac`, `amt_map`, `dv_map`, `time` e `lym`,
além do tempo atual `time`. As variáveis clínicas e demográficas são excluídas.
`id` e `lym` atual ficam fora de `X`; `lym` atual permanece como alvo.
Essa opção é independente de
`--normalization`: por exemplo, as séries selecionadas ainda podem usar a
normalização `all` do Pipeline.

```bash
# Uma pasta por combinação (2 x 3 x 2 x 2 = 24 experimentos).
SEARCHES="grid random" LAGS="1 3 5" FOLDS="3 5" SEEDS="42 123" \
N_ITER=30 N_JOBS=4 bash shells/run_ml_experiments.sh

# Validação rápida de toda a estrutura, incluindo SHAP.
SEARCHES="grid random" FOLDS=2 N_ITER=1 N_JOBS=1 \
bash shells/run_ml_experiments.sh --quick --shap-samples 4 --shap-background 5

# Reutilizar os Parquets já preparados.
PREPARE_DATA=0 LAGS="2 4" bash shells/run_ml_experiments.sh

# Experimento único, sem executar data_preparation.py.
python -m runners.run_ml_experiment --search random --lags 3 --folds 5 \
  --n-iter 30 --test-size 0.2 --normalization all --shap-samples 100

python -m runners.run_ml_experiment --help
```

`OUTPUT_DIR` define a pasta raiz do shell (padrão `studies/ml`). Cada chamada cria
uma pasta de estudo exclusiva; cada experimento cria outra pasta com data UTC,
busca, lags, folds e seed. Execuções antigas não são sobrescritas. O shell interrompe
em caso de falha; o runner registra o erro em `metadata.json` e `run.log`.

Use as variáveis `SEARCHES`, `LAGS`, `FOLDS`, `SEEDS`, `VARMODE`, `N_ITER`, `N_JOBS` e
`OUTPUT_DIR` para configurar o shell; os demais argumentos são encaminhados ao
runner, como `--models random_forest adaboost` e `--normalization none`.
`N_ITER` só limita a busca aleatória. Grid completo testa 720 combinações para RF,
1.728 para GB e 60 para AdaBoost, cada uma multiplicada pelo número de folds.
`--quick` substitui essas grades por dois tamanhos pequenos de ensemble: seus
resultados servem apenas para verificar a execução.

## Protocolo e interpretação

A divisão de teste reserva 20% dos **pacientes**, com seed configurável. A seleção
de hiperparâmetros usa GroupKFold somente nos pacientes restantes. Nenhum paciente
aparece simultaneamente no treino e teste, nem no treino e validação de um fold.
O melhor modelo é reajustado em todo o treino. O maior R² médio de CV é o critério de seleção nas buscas grid e aleatória;
`cv_r2` registra esse score; as métricas finais (incluindo RMSE) usam exclusivamente o teste. `splits.csv` permite auditar a divisão.
Com lags diferentes, pacientes com séries curtas podem sair da base, mudando a divisão.

A preparação mantém a semântica do notebook: `t0` é o primeiro valor da série;
`t1..tN` são observações anteriores do mesmo paciente, não intervalos fixos de tempo.
As quatro variáveis farmacológicas correntes são excluídas em todos os modos. Lags zero
inclui somente `t0` como histórico; a primeira observação de cada paciente é
excluída para que `t0` não coincida com o ponto atual. `id` e o target `lym` atual são excluídos das features.
O tempo atual `time`, seu histórico `time_t0..tN` e o histórico `lym_t0..tN`
entram em ambos os modos de variáveis. Os metadados conservam o tempo original.
Os lags de LYM pressupõem medidas anteriores disponíveis no momento da predição;
não são previsões recursivas.

A preparação exclui `OCC = 5` das três bases antes do merge, preenchimento
e interpolação. As contagens removidas ficam em `data_preparation_metadata.json`.
Bases processadas antigas precisam ser regeneradas com `python data_preparation.py`.

Modos de normalização (`all` é o padrão):

- `all`: MinMax de todos os atributos de entrada, incluindo lags, e do alvo
  `lym`. Os scalers ficam no modelo e são ajustados somente no treino de cada
  fold; no refit, usam somente o treino completo. `id` não entra no modelo.
  As previsões, métricas e valores SHAP são convertidos de volta para a unidade
  original de `lym`.
  Valores de teste podem ficar fora de [0, 1] se excederem os limites de treino.
- `time-series`: nome novo do antigo `patient`; MinMax somente das quatro
  variáveis temporais, usando toda a série de cada paciente, inclusive
  observações posteriores.
- `none`: sem normalização adicional das variáveis farmacológicas ou clínicas.

Em **todos** os modos, tempo atual, lags de tempo e histórico de LYM recebem
MinMax por atributo dentro do Pipeline, ajustado somente no treino de cada fold.
LYM não é normalizado pela série completa do paciente, para evitar que seu valor
atual/futuro afete os lags. O alvo original não é alterado em `prepare_features`.

`prepare_features(..., normalization="all")` gera os atributos sem escalá-los;
`make_search(..., normalization="all")` aplica os scalers durante o ajuste.

O data_preparation também pode usar preenchimento reverso
e interpolação. Portanto, este é um estudo retrospectivo: a separação por paciente
não transforma o fluxo em previsão prospectiva. `--normalization none` desativa
o MinMax das variáveis farmacológicas/clínicas, preservando a normalização
obrigatória de tempo e histórico LYM, sem desfazer os tratamentos anteriores.

Métricas: RMSE, MAE, R² e MAPE percentual (exclui targets exatamente zero, com
contagem registrada). R² indefinido e MAPE sem targets não zero são `null`.
As métricas ponderam observações igualmente, e não pacientes igualmente.

## Arquivos de cada experimento

- `metadata.json`: configuração, tempos, status, versões, SHA256 da entrada e
  fontes, contagens, modo de variáveis, features, grades, melhores parâmetros e
  configuração SHAP.
- `source/`: cópia dos scripts e YAML; `data_preparation_metadata.json`, quando
  disponível ao lado da entrada, é copiado como referência da preparação.
- `run.log`, `input_summary.csv`, `splits.csv`, `results.csv`, `model_comparison.png`.
- Uma pasta por modelo com `model.joblib`, `best_params.json`, `metrics.json`,
  `cv_results.csv`, `predictions.csv` e `predictions.png`.
- Nessa mesma pasta: `shap_bar.png`, `shap_beeswarm.png`, `shap_importance.csv`,
  `shap_values.parquet`, `shap_features.parquet` e `shap_samples.parquet`.
- `patient_temporal_analysis/`: métricas por paciente, resumo dos pacientes com
  menor e maior RMSE e gráficos temporais do valor real e previsto por modelo.

A análise temporal é executada automaticamente depois dos modelos e do SHAP. O
experimento só recebe status `complete` após esses arquivos serem gerados. O
critério de melhor/pior paciente é, respectivamente, o menor/maior RMSE individual
no conjunto de teste.

O modelo serializado espera as features já preparadas, na ordem de `features`
em `metadata.json`. No modo `all`, `model.joblib` inclui o scaler: forneça os
atributos e lags sem normalização e chame `predict`. A transformação inversa de
`lym` ocorre automaticamente. As chaves da busca têm prefixo
`model__regressor__` nesse modo. Nos outros modos, as chaves têm prefixo `model__` e o Pipeline inclui
o scaler de tempo e histórico LYM. Em `time-series`, somente as quatro séries
farmacológicas são normalizadas por paciente antes do Pipeline.

SHAP em todos os modos explica o Pipeline completo por permutação. As cores dos
gráficos e `shap_features.parquet` usam os valores preparados antes do MinMax.
As contribuições, previsões e métricas ficam na unidade original de `lym`. A soma
de SHAP com o valor de referência é validada contra a previsão do Pipeline.

SHAP explica uma amostra determinística das observações de teste e usa somente
observações de treino como referência. Os Pipelines usam PermutationExplainer
via `predict`, incluindo a normalização; a verificação de aditividade é obrigatória. `--shap-samples`,
`--shap-background` e `--shap-permutations` controlam custo e amostragem.
A importância é a média de |SHAP| por atributo, incluindo cada lag separadamente.
A tabela inclui todos os atributos; os gráficos mostram os 20 mais relevantes.
SHAP descreve contribuições à previsão; não estabelece causalidade, e atributos
correlacionados podem compartilhar importância.

Referências das APIs: [GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html),
[RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html),
[TreeExplainer](https://shap.readthedocs.io/en/stable/generated/shap.TreeExplainer.html) e
[PermutationExplainer](https://shap.readthedocs.io/en/latest/generated/shap.PermutationExplainer.html).

## Verificação

```bash
python -m unittest discover -s tests -p 'test_ml_*.py' -v
bash -n shells/run_ml_experiments.sh
```
