#Mini-template Python (règles → flags → dataset ML)
import pandas as pd
import numpy as np

def safe_div(a, b):
    return a / np.where(b == 0, 1, b)

def apply_rules(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # --- Derived totals
    out["cout_total"] = (
        out["cout_total_prod"].fillna(0)
        + out["cout_total_act"].fillna(0)
        + out["cout_total_ex"].fillna(0)
        + out["cout_mise_en_observation"].fillna(0)
        + out["cout_evacuation"].fillna(0)
    )

    out["unit_cost_prod"] = safe_div(out["cout_total_prod"].fillna(0), out["quantite_total_prod"].fillna(0))
    out["unit_cost_act"]  = safe_div(out["cout_total_act"].fillna(0),  out["quantite_total_act"].fillna(0))
    out["unit_cost_ex"]   = safe_div(out["cout_total_ex"].fillna(0),   out["quantite_total_ex"].fillna(0))

    # --- FORM rules
    out["r_missing_identity"] = (
        (out["nom_patient_is_filled"] == 0)
        | (out["id_prescripteur_is_filled"] == 0)
        | (out["id_gerant_is_filled"] == 0)
    ).astype(int)

    out["r_age_invalid"] = ((out["age_patient"].isna()) | (out["age_patient"] <= 0) | (out["age_patient"] > 120)).astype(int)
    out["r_sex_invalid"] = (out["sex"].isna() | out["sex"].isin(["M", "F"])).astype(int)
    out["r_visit_date_missing"] = out["visit_date"].isna().astype(int)

    # --- DATE rules
    # (assure-toi que ces colonnes sont bien en datetime)
    out["date_entree"] = pd.to_datetime(out["date_entree"], errors="coerce")
    out["date_sortie"] = pd.to_datetime(out["date_sortie"], errors="coerce")
    out["visit_date"]  = pd.to_datetime(out["visit_date"], errors="coerce")

    stay_days = (out["date_sortie"] - out["date_entree"]).dt.days + 1
    out["stay_days_from_dates"] = stay_days

    out["r_date_sortie_before_entree"] = (
        out["date_entree"].notna() & out["date_sortie"].notna() & (out["date_sortie"] < out["date_entree"])
    ).astype(int)

    out["delta_duree"] = (stay_days - out["nbre_jours"]).abs()
    out["r_duree_incoherente"] = (
        out["date_entree"].notna() & out["date_sortie"].notna()
        & out["nbre_jours"].notna()
        & (out["delta_duree"] > 1)
    ).astype(int)

    out["r_visit_outside_stay"] = (
        out["visit_date"].notna() & out["date_entree"].notna() & out["date_sortie"].notna()
        & ((out["visit_date"] < out["date_entree"]) | (out["visit_date"] > out["date_sortie"]))
    ).astype(int)

    # --- QTY/COST consistency rules
    out["r_cost_without_qty_prod"] = ((out["cout_total_prod"] > 0) & (out["quantite_total_prod"] <= 0)).astype(int)
    out["r_cost_without_qty_act"]  = ((out["cout_total_act"]  > 0) & (out["quantite_total_act"]  <= 0)).astype(int)
    out["r_cost_without_qty_ex"]   = ((out["cout_total_ex"]   > 0) & (out["quantite_total_ex"]   <= 0)).astype(int)

    out["r_cost_negative"] = (
        (out["cout_total_prod"] < 0) | (out["cout_total_act"] < 0) | (out["cout_total_ex"] < 0)
        | (out["cout_mise_en_observation"] < 0) | (out["cout_evacuation"] < 0)
    ).astype(int)

    # --- Evac rules (simple)
    out["r_evac_cost_without_mode"] = ((out["cout_evacuation"] > 0) & (out["mode_sortie"].isna())).astype(int)

    # label binaire (si tu veux)
    out["y_non_valide"] = out["status_verification"].isin(["a_corriger", "rejetee"]).astype(int)

    return out
