# Bases de dados de pacientes transplantados renais

Este diretório reúne três bases disponibilizadas como material suplementar de Quintairos et al. (2021). O estudo acompanhou pacientes adultos submetidos a transplante renal durante os seis primeiros meses após o procedimento e avaliou dados clínicos, rejeição aguda, biomarcadores urinários e a farmacocinética do tacrolimus e do ácido micofenólico (MPA).

Referência:

> Quintairos L, Colom H, Millán O, et al. *Early prognostic performance of miR155-5p monitoring for the risk of rejection: Logistic regression with a population pharmacokinetic approach in adult kidney transplant patients*. PLOS ONE, 2021. [Artigo e materiais suplementares](https://doi.org/10.1371/journal.pone.0245880).

## Visão geral dos arquivos

| Arquivo | Conteúdo | Dimensão observada |
|---|---|---:|
| `tacrolimus.csv` | Registros de administração e concentrações de tacrolimus em sangue total, acompanhados de covariáveis clínicas. | 1.596 linhas, 29 campos e 58 pacientes |
| `mpa.csv` | Registros de administração e concentrações plasmáticas de MPA, acompanhados de covariáveis clínicas. | 1.527 linhas, 29 campos e 56 pacientes |
| `lr_input.csv` | Base preparada para o modelo de regressão logística da rejeição aguda, incluindo expressão urinária de miR155-5p. | 1.525 linhas, 25 campos e 58 pacientes |

Os arquivos estão no formato de eventos usado pelo NONMEM. Portanto, uma linha não corresponde necessariamente a um paciente ou a uma visita: existem linhas de dose, linhas de coleta e repetições das covariáveis. O identificador do paciente é `#ID`.

## Campos de tempo e identificação

| Campo | Descrição | Unidade/codificação |
|---|---|---|
| `#ID` | Identificador anonimizado do paciente. O mesmo paciente aparece em várias linhas. | Inteiro de 1 a 58; `mpa.csv` contém 56 desses pacientes |
| `TIME` | Tempo acumulado desde a origem temporal adotada no estudo. | Horas |
| `TAD` | *Time After Dose*: tempo transcorrido desde a dose anterior. | Horas |
| `OCC` | Ocasião ou visita clínica. | `1` = primeira semana; `2` = mês 1; `3` = mês 2; `4` = mês 3; `5` = mês 6 |

## Campos farmacocinéticos

Estes campos aparecem em `tacrolimus.csv` e `mpa.csv`.

| Campo | Descrição | Unidade/codificação |
|---|---|---|
| `AMT` | Quantidade administrada no evento de dose. Em `tacrolimus.csv`, corresponde à dose de tacrolimus em mg. Em `mpa.csv`, a dose de micofenolato de mofetila foi convertida para equivalente de MPA para a modelagem. | Valor positivo nas linhas de dose; `0` nas linhas de concentração |
| `DV` | Variável dependente, isto é, a concentração usada no modelo farmacocinético. | Ver as observações específicas abaixo |
| `MDV` | Indicador de ausência de observação da variável dependente (*missing dependent variable*). | `0` = `DV` é uma observação; `1` = linha sem concentração observada, normalmente uma dose |
| `EVID` | Tipo do evento no NONMEM. | `0` = observação; `1` = administração de dose |
| `II ` | Intervalo entre doses (*interdose interval*). O nome da coluna contém um espaço ao final. | Horas; `12` nas linhas que iniciam administração repetida e `0` nas demais |
| `ADDL` | Número de doses adicionais a serem geradas após a dose registrada. | Inteiro; `0` quando não há doses adicionais codificadas |
| `CMT` | Compartimento do modelo farmacocinético associado ao evento. | `1` = compartimento de administração/absorção; `2` = compartimento central, no qual a concentração é observada |

### Particularidades de `DV`

- Em `tacrolimus.csv`, `DV` está armazenado como o **logaritmo natural da concentração de tacrolimus em sangue total**, originalmente medida em ng/mL. Para retornar à escala original, use `concentracao = exp(DV)`. O arquivo contém 1.102 linhas de observação (`MDV = 0`) e 494 linhas de dose (`MDV = 1`).
- Em `mpa.csv`, `DV` representa diretamente a **concentração plasmática de MPA**, medida em µg/mL. O arquivo contém 1.071 linhas de observação e 456 linhas de dose.
- O valor `DV = 0` em uma linha de dose não representa uma concentração medida. A seleção das concentrações deve ser feita com `MDV = 0` e `EVID = 0`.
- Em `mpa.csv`, existem 21 linhas classificadas como observações com `DV = 0`. Elas devem ser verificadas antes de serem interpretadas como concentrações reais, pois o artigo informa que não houve concentrações abaixo do limite de quantificação.

## Características do receptor e do doador

| Campo | Descrição | Unidade/codificação |
|---|---|---|
| `AGE` | Idade do receptor do transplante. | Anos |
| `SEX` | Sexo do receptor. | `0` = masculino; `1` = feminino |
| `WGT` | Peso do receptor na ocasião registrada. | kg |
| `HGT` | Altura do receptor. | cm |
| `IMC` | Índice de massa corporal. | kg/m²; aparece arredondado para número inteiro |
| `DTY` | Tipo de doador. | `0` = doador falecido/cadavérico; `1` = doador vivo |
| `DAGE` | Idade do doador. | Anos |
| `CISQ` | Tempo de isquemia fria do enxerto. | Provavelmente horas; `9999` indica dado ausente ou não aplicável |
| `DYL` | Tempo de diálise anterior ao transplante. | A unidade não é informada no CSV; os valores são compatíveis com meses. `9999` indica dado ausente |
| `DYLC` | Indicador derivado da informação de diálise; existe apenas em `lr_input.csv`. | `0` quando `DYL = 0`; `1` quando há tempo de diálise positivo; `9999` = ausente |

## Função renal, imunologia e desfechos clínicos

| Campo | Descrição | Unidade/codificação |
|---|---|---|
| `GF` | Taxa de filtração glomerular estimada pelo método MDRD. | mL/min; `9999` deve ser tratado como ausente |
| `GFC` | Categoria da função renal. | Codificação pretendida: `1` = falência renal/estágio terminal; `2` = redução grave; `3` = redução moderada; `4` = redução leve; `5` = função ≥ 90 mL/min. Há registros que não coincidem perfeitamente com `GF`; para análises quantitativas, prefira `GF` e valide as exceções |
| `IFC` | Indicador geral de ocorrência de infecção na ocasião. | `0` = não; `1` = sim; `9999` = ausente |
| `OIFC` | Indicador de outras infecções, separado dos indicadores específicos de CMV e BKV. | `0` = não; `1` = sim |
| `CMV` | Ocorrência de infecção por citomegalovírus. | `0` = não; `1` = sim |
| `BKV` | Ocorrência de infecção por BK vírus. | `0` = não; `1` = sim; há um valor ausente em `lr_input.csv` |
| `DM` | Ocorrência/presença de diabetes mellitus durante o estudo. | `0` = não; `1` = sim |
| `AR` | Ocorrência de rejeição aguda do enxerto renal. | `0` = não; `1` = sim; presente em `tacrolimus.csv` e `lr_input.csv` |
| `LYM` | Medida de linfócitos usada como covariável clínica. | A unidade não é fornecida nos arquivos nem explicitada no artigo; não assumir células/µL ou porcentagem sem confirmação no protocolo original |
| `NTA` | Campo binário adicional não definido no artigo, no apêndice dos modelos ou no cabeçalho dos CSVs. | Valores `0`, `1` e `9999`; seu significado deve ser confirmado com os autores antes do uso analítico |

`IFC`, `OIFC` e `NTA` são descritos com cautela porque o material suplementar não apresenta um dicionário formal das siglas. A interpretação de `IFC` e `OIFC` é sustentada pelo padrão dos dados e pelas variáveis clínicas descritas no artigo; `NTA` permanece sem definição segura.

## Biomarcador e resposta em `lr_input.csv`

| Campo | Descrição | Unidade/codificação |
|---|---|---|
| `AR` | Resposta binária usada no modelo logístico: ocorrência de rejeição aguda. | `0` = ausência; `1` = ocorrência |
| `M155` | Expressão relativa de miR155-5p em sedimento urinário, obtida por qPCR. O artigo descreve a expressão relativa por `2^(-ΔCq)`, de modo que valores maiores indicam maior expressão. | Valor positivo = medida disponível; `0` = marcador estrutural em linhas sem medida; `9999` = ausente |

Embora `lr_input.csv` tenha 1.525 linhas, o código NONMEM fornecido pelos autores ignora `M155 = 0` e `M155 = 9999`. Após essa exclusão, restam **183 observações utilizáveis de miR155-5p**. Portanto, as 1.525 linhas não devem ser tratadas como 1.525 medidas independentes.

## Cuidados para análise estatística

1. Substitua `9999` por valor ausente (`NA`) antes de calcular médias, correlações ou ajustar modelos.
2. Remova o espaço final de `II `, por exemplo com `df.columns = df.columns.str.strip()` em pandas.
3. Para concentrações farmacocinéticas, filtre `MDV = 0` e `EVID = 0`.
4. Desfaça o logaritmo de `DV` em `tacrolimus.csv` quando precisar da concentração em ng/mL.
5. Não considere as linhas como observações independentes: há medidas repetidas do mesmo paciente. Modelos longitudinais ou de efeitos mistos devem usar `#ID` para representar essa dependência.
6. Variáveis fixas do paciente, como `AGE`, `SEX`, `DTY` e `DAGE`, aparecem repetidas em muitas linhas.
7. Para estudar linfócitos, confirme primeiro a unidade e a natureza de `LYM`. Para relacioná-los à exposição aos medicamentos, será necessário resumir as curvas concentração–tempo por paciente e ocasião, por exemplo por AUC ou concentração de vale, antes de integrá-las às medidas clínicas.

## Origem dos arquivos

- `tacrolimus.csv`: [S1 Table — Tacrolimus PK model dataset](https://doi.org/10.1371/journal.pone.0245880.s005)
- `mpa.csv`: [S2 Table — MPA PK model dataset](https://doi.org/10.1371/journal.pone.0245880.s006)
- `lr_input.csv`: [S3 Table — Logistic regression model dataset](https://doi.org/10.1371/journal.pone.0245880.s007)
- Código dos modelos NONMEM: [S1 Appendix](https://doi.org/10.1371/journal.pone.0245880.s008)

## Licença e citação

O artigo foi publicado em acesso aberto sob licença Creative Commons Attribution. Ao reutilizar os dados, cite o trabalho original de Quintairos et al. (2021) e consulte as condições apresentadas pela PLOS ONE.
