"""Preparação dos dados. Execute com: python data_preparation.py."""

import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import numpy as np


def normalizar_colunas(df):
    """Normaliza os nomes das colunas."""
    df = df.copy()

    df.columns = [
        unicodedata.normalize("NFKD", col)
        .encode("ascii", "ignore")
        .decode("utf-8")
        .strip()
        .lower()
        for col in df.columns
    ]

    df.columns = [re.sub(r"[^a-z0-9_]", "_", col) for col in df.columns]
    df.columns = [re.sub(r"_+", "_", col).strip("_") for col in df.columns]

    return df


# ---------------------------------------------------------
# Leitura dos dados
# ---------------------------------------------------------


def read_data():

    print("\n[1/6] Lendo bases de dados...")

    lr_input_df = (
        normalizar_colunas(pd.read_parquet("data/lr_input.parquet"))
        .sort_values(["time", "id"])
        .reset_index(drop=True)
    )

    print(
        f"  lr_input carregado: "
        f"{len(lr_input_df)} linhas | "
        f"{lr_input_df['id'].nunique()} pacientes"
    )

    mpa_df = (
        normalizar_colunas(pd.read_parquet("data/mpa.parquet"))
        .sort_values(["time", "id"])
        .reset_index(drop=True)
    )

    print(
        f"  MPA carregado: "
        f"{len(mpa_df)} linhas | "
        f"{mpa_df['id'].nunique()} pacientes"
    )

    tacrolimus_df = (
        normalizar_colunas(pd.read_parquet("data/tacrolimus.parquet"))
        .sort_values(["time", "id"])
        .reset_index(drop=True)
    )

    print(
        f"  Tacrolimus carregado: "
        f"{len(tacrolimus_df)} linhas | "
        f"{tacrolimus_df['id'].nunique()} pacientes"
    )

    return lr_input_df, mpa_df, tacrolimus_df


# ---------------------------------------------------------
# Merge
# ---------------------------------------------------------


def exclude_occ5(data):
    """Exclui a ocasião 5 antes de qualquer merge, preenchimento ou interpolação."""
    if "occ" not in data.columns:
        raise ValueError("Coluna occ obrigatória para excluir OCC = 5")
    return data.loc[pd.to_numeric(data["occ"], errors="raise") != 5].copy()


def merge_db(lr_input_df, mpa_df, tacrolimus_df):

    print("\n[2/6] Realizando merge das bases...")

    lr_input_df, mpa_df, tacrolimus_df = (
        exclude_occ5(frame) for frame in (lr_input_df, mpa_df, tacrolimus_df)
    )

    print("  Associando dados de Tacrolimus...")

    df_merge = pd.merge_asof(
        lr_input_df,
        tacrolimus_df[["id", "time", "amt", "dv", "mdv"]],
        on="time",
        by="id",
        direction="nearest",
        suffixes=("", "_tac"),
        tolerance=5,
    )

    print("  Associando dados de MPA...")

    df_merge = pd.merge_asof(
        df_merge,
        mpa_df[["id", "time", "amt", "dv", "mdv"]],
        on="time",
        by="id",
        direction="nearest",
        suffixes=("", "_map"),
        tolerance=5,
    )

    df_merge = df_merge.drop(columns=["occ", "imc", "ar", "nta"])

    print(
        f"  Merge concluído: "
        f"{df_merge.shape[0]} linhas x "
        f"{df_merge.shape[1]} colunas"
    )

    return df_merge


# ---------------------------------------------------------
# Tratamento das flags
# ---------------------------------------------------------


def translating_flags(merged_df):

    print("\n[3/6] Tratando flags e valores ausentes...")

    # Valores antes do tratamento
    print(f"  NaNs em dv antes: {merged_df['dv'].isna().sum()}")
    print(f"  NaNs em dv_map antes: {merged_df['dv_map'].isna().sum()}")

    # Colocando NaN com base nas colunas de labels
    merged_df.loc[merged_df["mdv"] == 1, "dv"] = np.nan
    merged_df.loc[merged_df["mdv_map"] == 1, "dv_map"] = np.nan

    merged_df.loc[merged_df["lym"] == 9999, "lym"] = np.nan
    merged_df.loc[merged_df["dylc"] == 9999, "dyl"] = np.nan
    merged_df.loc[merged_df["cisq"] == 9999, "cisq"] = np.nan
    merged_df.loc[merged_df["gf"] == 9999, "gf"] = np.nan
    merged_df.loc[merged_df["ifc"] == 9999, "ifc"] = np.nan
    merged_df.loc[merged_df["m155"] == 9999, "m155"] = np.nan

    # Dose ausente = 0
    merged_df["amt_map"] = merged_df["amt_map"].fillna(0)
    merged_df["amt"] = merged_df["amt"].fillna(0)

    # Removendo colunas de flags
    merged_df = merged_df.drop(columns=["mdv", "mdv_map", "dylc"])

    # Renomeando Tacrolimus
    merged_df = merged_df.rename(columns={"amt": "amt_tac", "dv": "dv_tac"})

    # Dados estáticos por paciente
    colunas_estaticas = [
        "age",
        "sex",
        "hgt",
        "dty",
        "dage",
        "cisq",
        "dyl",
    ]

    print("  Propagando dados estáticos dentro de cada paciente...")

    merged_df[colunas_estaticas] = merged_df.groupby("id")[colunas_estaticas].transform(
        lambda x: x.ffill().bfill()
    )

    print("  Tratamento das flags concluído.")

    return merged_df


# ---------------------------------------------------------
# Interpolação
# ---------------------------------------------------------


def ts_interpolation(df_final):

    print("\n[4/6] Interpolando séries temporais...")

    colunas_temporais = ["dv_tac", "dv_map", "dm", "m155"]

    df_final = df_final.sort_values(["id", "time"]).copy()

    print("  NaNs antes da interpolação:")

    for coluna in colunas_temporais:
        print(f"    {coluna}: {df_final[coluna].isna().sum()}")

    df_final[colunas_temporais] = df_final.groupby("id")[colunas_temporais].transform(
        lambda x: x.interpolate(method="linear", limit_direction="both")
    )

    print("  NaNs após interpolação:")

    for coluna in colunas_temporais:
        print(f"    {coluna}: {df_final[coluna].isna().sum()}")

    # -----------------------------------------------------
    # Pacientes sem nenhum dado de MPA
    # -----------------------------------------------------

    ids_sem_dv_map = df_final.groupby("id")["dv_map"].apply(lambda x: x.isna().all())

    ids_sem_dv_map = ids_sem_dv_map[ids_sem_dv_map].index.to_list()

    print(f"  Pacientes sem nenhum valor de dv_map: " f"{len(ids_sem_dv_map)}")

    if ids_sem_dv_map:
        print(f"    IDs: {ids_sem_dv_map}")

    df_final = df_final[~df_final["id"].isin(ids_sem_dv_map)]

    # -----------------------------------------------------
    # Pacientes sem nenhum dado de Tacrolimus
    # -----------------------------------------------------

    ids_sem_dv_tac = df_final.groupby("id")["dv_tac"].apply(lambda x: x.isna().all())

    ids_sem_dv_tac = ids_sem_dv_tac[ids_sem_dv_tac].index.to_list()

    print(f"  Pacientes sem nenhum valor de dv_tac: " f"{len(ids_sem_dv_tac)}")

    if ids_sem_dv_tac:
        print(f"    IDs: {ids_sem_dv_tac}")

    df_final = df_final[~df_final["id"].isin(ids_sem_dv_tac)]

    print(
        f"  Após remoção: "
        f"{len(df_final)} linhas | "
        f"{df_final['id'].nunique()} pacientes"
    )

    return df_final


# ---------------------------------------------------------
# Limpeza final
# ---------------------------------------------------------


def data_cleaning(final_df):

    print("\n[5/6] Criando base limpa...")

    colunas_obrigatorias = ["m155", "ifc", "cisq", "dyl", "lym", "dm"]

    antes = len(final_df)

    clean_df = final_df.dropna(subset=colunas_obrigatorias)

    depois = len(clean_df)

    print(f"  Linhas antes: {antes}")
    print(f"  Linhas depois: {depois}")
    print(f"  Linhas removidas: {antes - depois}")
    print(f"  Pacientes restantes: " f"{clean_df['id'].nunique()}")

    return clean_df


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------


def dataset_metadata(df):
    """Resume pacientes e ausências sem incluir identificadores individuais."""
    missing = df.isna()
    total_missing = int(missing.sum().sum())
    rows_with_missing = missing.any(axis=1)
    return {
        "linhas": len(df),
        "colunas": len(df.columns),
        "pacientes": int(df["id"].nunique()),
        "linhas_sem_id": int(df["id"].isna().sum()),
        "linhas_duplicadas": int(df.duplicated().sum()),
        "missing_values": total_missing,
        "missing_percentual": 100 * total_missing / df.size if df.size else 0.0,
        "linhas_com_missing": int(rows_with_missing.sum()),
        "pacientes_com_missing": int(df.loc[rows_with_missing, "id"].nunique()),
        "por_coluna": {
            column: {
                "tipo": str(df[column].dtype),
                "missing_values": int(missing[column].sum()),
                "missing_percentual": (
                    float(missing[column].mean() * 100) if len(df) else 0.0
                ),
            }
            for column in df.columns
        },
    }


def main():

    print("=" * 60)
    print("INICIANDO PREPARAÇÃO DOS DADOS")
    print("=" * 60)

    lr_input_df, mpa_df, tacrolimus_df = read_data()
    metadata = {
        "versao": 2,
        "gerado_em_utc": datetime.now(timezone.utc).isoformat(),
        "definicao_missing": (
            "Valores reconhecidos por pandas.isna (NaN, None, NA, NaT). "
            "Flags e sentinelas como 9999 não contam como missing antes de "
            "sua tradução. O tratamento também preenche valores ausentes. "
            "Totais de missing entre etapas podem envolver colunas diferentes."
        ),
        "antes": {
            "lr_input": dataset_metadata(lr_input_df),
            "mpa": dataset_metadata(mpa_df),
            "tacrolimus": dataset_metadata(tacrolimus_df),
        },
        "etapas": {},
    }

    metadata["exclusao_occ5"] = {
        "momento": "Antes do merge, preenchimento e interpolação",
        "linhas_removidas": {
            name: len(frame) - len(exclude_occ5(frame))
            for name, frame in (
                ("lr_input", lr_input_df), ("mpa", mpa_df),
                ("tacrolimus", tacrolimus_df),
            )
        },
    }
    merged_df = merge_db(lr_input_df, mpa_df, tacrolimus_df)

    metadata["etapas"]["apos_merge"] = dataset_metadata(merged_df)

    treated_df = translating_flags(merged_df)
    metadata["etapas"]["apos_flags_e_preenchimento"] = dataset_metadata(treated_df)

    final_df = ts_interpolation(treated_df)

    clean_df = data_cleaning(final_df)
    metadata["depois"] = {
        "full_processed": dataset_metadata(final_df),
        "clean_processed": dataset_metadata(clean_df),
    }
    metadata["comparacao_com_lr_input"] = {
        name: {
            "linhas_removidas": len(lr_input_df) - len(df),
            "pacientes_removidos": int(
                lr_input_df.loc[~lr_input_df["id"].isin(df["id"]), "id"].nunique()
            ),
        }
        for name, df in (("full_processed", final_df), ("clean_processed", clean_df))
    }

    print("\n[6/6] Salvando bases em formato Parquet...")

    final_df.to_parquet("data/full_processed.parquet", index=False)

    print("  Salvo: data/full_processed.parquet " f"({len(final_df)} linhas)")

    clean_df.to_parquet("data/clean_processed.parquet", index=False)

    print("  Salvo: data/clean_processed.parquet " f"({len(clean_df)} linhas)")

    metadata_path = Path("data/data_preparation_metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"  Metadados salvos: {metadata_path}")

    print("\n" + "=" * 60)
    print("PREPARAÇÃO DOS DADOS CONCLUÍDA")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    main()
