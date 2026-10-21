# Estudo de interpolacao temporal por paciente

LOOCV intrapaciente restrito a exames internos (com observacoes antes e depois).
As series usam MinMax ajustado apenas nos exames de treino do paciente.
Folds com treino constante seguem o MinMaxScaler (amplitude efetiva igual a 1).
Erros usam `predito - observado`. MAE, RMSE e vies principais dao o mesmo peso
a cada paciente; as metricas `*_micro` sao fornecidas apenas para auditoria.
Em `m155`, zero e marcador estrutural sem medida e nao entra como exame observado.

## Melhores metodos por serie

| Serie | Metodo | Pacientes | Exames | MAE | RMSE | Vies |
|---|---|---:|---:|---:|---:|---:|
| dv_map | pchip | 52 | 816 | 0.244176 | 0.3795 | 0.0332899 |
| dv_tac | pchip | 52 | 632 | 0.130894 | 0.190694 | 0.0352025 |
| gf | pchip | 55 | 1294 | 0.00110911 | 0.0209933 | 0.000442164 |
| lym | linear | 52 | 942 | 0.00498953 | 0.0585898 | 0.00205287 |
| m155 | temporal_regression_quadratic | 11 | 33 | 1.59086 | 3.82253 | -1.10764 |
| wgt | linear | 55 | 1295 | 0.0020846 | 0.0202811 | -0.000425014 |

## Elegibilidade

| Serie | Observados | Pacientes observados | Pacientes elegiveis | Exames LOOCV |
|---|---:|---:|---:|---:|
| m155 | 183 | 58 | 11 | 33 |
| wgt | 1411 | 58 | 55 | 1295 |
| gf | 1410 | 58 | 55 | 1294 |
| lym | 1052 | 56 | 52 | 942 |
| dv_tac | 744 | 54 | 52 | 632 |
| dv_map | 923 | 55 | 52 | 816 |

A escolha automatica usa o menor RMSE macro, com MAE como desempate.
Indicadores binarios, doses prescritas e variaveis estaticas nao fazem parte
deste estudo, pois estes interpoladores continuos nao preservam sua semantica.
