# Estudo de interpolacao temporal por paciente

LOOCV intrapaciente restrito a exames internos (com observacoes antes e depois).
As series usam MinMax ajustado apenas nos exames de treino do paciente.
Folds com treino constante seguem o MinMaxScaler (amplitude efetiva igual a 1).
Erros usam `predito - observado`. MAE, RMSE, MAPE e vies dao o mesmo peso
a cada paciente; as metricas `*_micro` sao fornecidas apenas para auditoria.
MAPE e calculado na escala original; observacoes reais iguais a zero sao excluidas.
Em `m155`, zero e marcador estrutural sem medida e nao entra como exame observado.

## Melhores metodos por serie

| Serie | Metodo | Pacientes | Exames | MAE ± DP | RMSE ± DP | MAPE (%) ± DP | Vies ± DP |
|---|---|---:|---:|---:|---:|---:|---:|
| dv_map | pchip | 55 | 957 | 0.23563 ± 0.10814 | 0.365117 ± 0.152354 | 76.0636 ± 82.7625 | 0.0336629 ± 0.0979728 |
| dv_tac | pchip | 58 | 986 | 0.186489 ± 0.0710203 | 0.282402 ± 0.0906744 | 13.1434 ± 6.0737 | 0.0940565 ± 0.0606177 |
| gf | pchip | 55 | 1408 | 0.0011074 ± 0.00414985 | 0.0209928 ± 0.0205587 | 0.315641 ± 1.77564 | 0.000440814 ± 0.00402932 |
| lym | linear | 52 | 1052 | 0.00491385 ± 0.0199827 | 0.0585829 ± 0.0572235 | 0.24742 ± 1.15713 | 0.00204865 ± 0.0173633 |
| m155 | temporal_regression_quadratic | 11 | 33 | 1.59086 ± 1.87375 | 3.82253 ± 3.12714 | 462.499 ± 1053.34 | -1.10764 ± 1.77091 |
| wgt | linear | 55 | 1409 | 0.00274525 ± 0.00938651 | 0.0303254 ± 0.0291615 | 0.0280861 ± 0.0886027 | -0.000402099 ± 0.00272446 |

## Elegibilidade

| Serie | Observados (DP/paciente) | Pacientes observados | Pacientes elegiveis | Exames LOOCV (DP/paciente) |
|---|---:|---:|---:|---:|
| m155 | 183 (1.35) | 58 | 11 | 33 (0) |
| wgt | 1525 (9.91) | 58 | 55 | 1409 (8.31) |
| gf | 1524 (9.9) | 58 | 55 | 1408 (8.31) |
| lym | 1162 (8.74) | 56 | 52 | 1052 (7.15) |
| dv_tac | 1102 (5.79) | 58 | 58 | 986 (5.79) |
| dv_map | 1071 (6.11) | 56 | 55 | 957 (5.81) |

A escolha automatica usa o menor RMSE macro, com MAE como desempate.
Indicadores binarios, doses prescritas e variaveis estaticas nao fazem parte
deste estudo, pois estes interpoladores continuos nao preservam sua semantica.
