# Estudo de interpolacao temporal

O estudo compara, por serie continua, interpolacao linear, PCHIP, spline cubica
natural e regressoes temporais linear, quadratica e cubica. Ele usa os dados
depois do merge e da traducao de flags, mas antes da interpolacao do pipeline.
Por padrao, cada serie recebe MinMax ajustado nos exames de treino do paciente.

```bash
python -m runners.run_interpolation_study
```

Para executar pelo shell simplificado:

```bash
bash shells/run_interpolation_study.sh
```

Escolha uma ou várias ocasiões pela variável `OCC`, como no estudo de ML:

```bash
OCC=1 bash shells/run_interpolation_study.sh
OCC="1 2 3" bash shells/run_interpolation_study.sh
```

Também podem ser configurados `STUDY_NAME`, `OUTPUT_DIR`, `MIN_OBSERVATIONS`,
`NORMALIZATION` e `PYTHON_BIN` por variáveis de ambiente.

Por paciente e serie, cada exame interno observado e removido uma vez. O valor e
estimado pelos demais exames do mesmo paciente. O primeiro e o ultimo exame nao
sao avaliados, pois sua remocao exigiria extrapolacao. Pacientes precisam ter ao
menos cinco tempos observados, garantindo folds comuns a todos os metodos.

O MinMax e reajustado em cada LOOCV usando somente os exames de treino daquele
paciente. O exame retido e os demais pacientes nao participam do ajuste. As
metricas principais ficam na escala normalizada; valores observados e preditos na
escala original continuam em `exam_predictions.parquet`. Use
`--normalization none` somente para uma analise adicional na escala original.
Quando todos os exames de treino tem o mesmo valor, a amplitude efetiva e 1,
reproduzindo o tratamento de atributos constantes do `MinMaxScaler`; essa
situacao fica marcada em `constant_training_range`.

As metricas principais sao macroagregadas: calcula-se primeiro MAE, MSE e vies
em cada paciente; depois, MAE e vies sao as medias entre pacientes, e RMSE e a
raiz da media dos MSE individuais. Assim, um paciente com mais exames nao domina
o resultado. As versoes micro tambem sao registradas apenas para auditoria.

Arquivos produzidos:

- `exam_predictions.parquet`: valor observado, predito e erros de cada LOOCV;
- `patient_metrics.csv`: metricas por paciente, serie e metodo;
- `macro_metrics.csv`: comparação principal com RMSE, MAE, MAPE e respectivos
  desvios padrão entre pacientes;
- `best_methods.csv`: menor RMSE macro por serie;
- `eligibility.csv`: pacientes e exames elegiveis;
- `series_catalog.csv`: classificação de todas as séries temporais, incluindo
  exclusões semânticas e suas justificativas;
- `metadata.json`: protocolo, hashes dos dados e versoes;
- `plots/<serie>_metrics.png`: um gráfico por série com RMSE, MAE e MAPE,
  incluindo barras de erro de um desvio padrão entre pacientes;
- `README.md`: resumo visual e legível, com média e desvio padrão.

O MAPE é calculado na escala original da variável. Valores observados iguais a
zero são excluídos apenas do MAPE e suas contagens são registradas nas tabelas.

Por padrão, as séries contínuas são descobertas a partir da classificação atual
do pipeline. Atualmente são `m155`, `wgt`, `gf`, `lym`, `dv_tac` e `dv_map`.
`tad` e `gfc` são removidas durante a preparação e não chegam ao estudo. As
colunas `amt_*` são catalogadas, mas excluídas dos interpoladores porque
representam eventos de dose. Variáveis estáticas também são excluídas. A
auditoria completa fica em `series_catalog.csv`. Outra seleção pode ser
executada explicitamente com `--series`.

O estudo usa `OCC=1`, como o pipeline principal. Use `--occ` para avaliar outras
ocasiões explicitamente.

Para `m155`, somente valores estritamente positivos contam como exames: `0` e
uma linha estrutural sem medida no conjunto original, e `9999` e traduzido para
ausente antes da validacao.

