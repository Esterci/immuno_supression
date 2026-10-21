# Estudo de interpolacao temporal

O estudo compara, por serie continua, interpolacao linear, PCHIP, spline cubica
natural e regressoes temporais linear, quadratica e cubica. Ele usa os dados
depois do merge e da traducao de flags, mas antes da interpolacao do pipeline.
Por padrao, cada serie recebe MinMax ajustado nos exames de treino do paciente.

```bash
python -m runners.run_interpolation_study
```

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
- `macro_metrics.csv`: comparacao principal com peso igual por paciente;
- `best_methods.csv`: menor RMSE macro por serie;
- `eligibility.csv`: pacientes e exames elegiveis;
- `metadata.json`: protocolo, hashes dos dados e versoes;
- `rmse_comparison.png` e `README.md`: resumo visual e legivel.

As series padrao sao `m155`, `wgt`, `gf`, `lym`, `dv_tac` e `dv_map`. Indicadores
binarios (`dm`, infeccoes), doses prescritas e variaveis estaticas sao excluidos,
pois interpoladores continuos nao respeitam sua semantica. Outra selecao pode ser
executada explicitamente com `--series`.

Para `m155`, somente valores estritamente positivos contam como exames: `0` e
uma linha estrutural sem medida no conjunto original, e `9999` e traduzido para
ausente antes da validacao.

