from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler

from src.utils import get_logger, load_config

import numpy as np

logger = get_logger(__name__)

# Columns ( names of raw original.csv)

IDENTIFIER_COLS = [
    "id_student", "id_questionnaire", "id_student_original",
    "id_year", "id_grade", "id_class_group",
    "id_school", "id_student_16_19", "id_school_16_19",
]

AVAILABILITY_COLS = [
    "student_questionnaire", "principals_questionnaire",
    "family_questionnaire", "teachers_questionnaire",
    "census", "scores",
]

# Dropped: implied by other columns that are kept
REDUNDANT_COLS = [
    # immigration implied by place-of-birth cols (f5a, f5b, f5n)
    "inmigrant", "inmigrant2", "inmigrant_second_gen",
    # income quantile implied by f34 (monthly_household_income)
    "household_income_q",
    # household size duplicates f1n
    "nhousehold",
    # family structure implied by f31 (type_of_family_unit)
    "single_parent_household",
    # books duplicates f11 (extent_of_books_at_home)
    "books",
    # raw education codes – recoded versions (f3a, f3b) kept
    "mother_education", "father_education",
    # admin grouping metadata
    "p_group_criteria_alphabet", "p_group_criteria_gender",
    "p_group_criteria_language", "p_group_criteria_performance",
    "p_group_criteria_homogeneity", "p_group_criteria_heterogeneity",
    # survey weight – not meaningful for generation
    "weight",
    # misc columns only present in some years (MAR)
    "f8ta", "f8tm",   # age at start of schooling raw fields
    "f9a", "f9e", "f9g",  # individual resource items subsumed by aggregates
    "f13n",           # number of books (replaced by f11 extent)
    "f14c",           # other relative visits (sparse)
    "f22",            # hours of private tuition (very sparse)
    "f24a", "f24b",   # specific extracurricular items
    "f33a","f33b","f33c","f33d","f33e","f33f","f33g","f33h",  # detailed income breakdown
]

ALL_SCORE_COLS = [
    "score_MAT", "level_MAT",
    "score_LEN", "level_LEN",
    "score_ING", "level_ING",
]

#  Aggregation maps
 
FAMILY_AGG_MEAN = {
    "f_frequency_of_books_at_home":   ["f9b", "f9c", "f9h"],
    "f_frequency_of_tech_at_home":    ["f9d", "f9f"],
    "f_frequency_of_see_adult_read":  ["f12a", "f12b"],
    "f_extent_of_interest_in_interview": ["f15a", "f15b", "f15d", "f15f"],
    "f_frequency_of_support_at_home": ["f16a", "f16b", "f16c", "f16d", "f16e", "f16f"],
    "f_frequency_of_parent_involved_in_school_activities": ["f17a", "f17b", "f17c", "f17d"],
    "f_extent_of_family_satisfaction": ["f18a", "f18b", "f18e", "f18f", "f18g", "f18h"],
    "f_extent_of_teacher_satisfaction": ["f19a", "f19b", "f19c", "f19d", "f19e"],
}

# Source cols to drop after family aggregation
FAMILY_AGG_DROP = [
    "f9b","f9c","f9h","f9d","f9f",
    "f12a","f12b",
    "f15a","f15b","f15c","f15d","f15e","f15f",
    "f16a","f16b","f16c","f16d","f16e","f16f",
    "f17a","f17b","f17c","f17d",
    "f18a","f18b","f18c","f18d","f18e","f18f","f18g","f18h","f18i",
    "f19a","f19b","f19c","f19d","f19e",
]

TEACHER_AGG_MEAN = {
    "t_extent_of_evaluation_variety": [
        "p24a","p24b","p24c","p24d","p24e",
        "p24f","p24g","p24h","p24i","p24j","p24k",
    ],
    "t_extent_of_pfc_incidence":      ["p16a","p16c","p16g"],
    "t_extent_of_family_interest":    ["p29a","p29b","p29c","p29d","p29e"],
    "t_extent_of_family_support":     ["p30a","p30b","p30c"],
    "t_extent_of_work_facilitated_by_management":   ["p34b","p34d"],
    "t_extent_of_positive_relationships": ["p31d","p311a","p311b","p311c","p311e"],
    "t_extent_of_student_involvement_during_class":["p21a","p21b","p21c","p21d","p21e","p21f"],
    "t_extent_of_teaching_methods_variety":   ["p22a","p22d","p22e","p22f","p22g"],
    "t_extent_of_resource_variety":   ["p23a","p23b","p23c","p23d","p23e","p23f","p23g","p23h"],
    "t_extent_of_class_behaviour":    ["p12b","p12d"],
    "t_extent_of_opinion_on_school":  ["p32b"],
    "t_extent_of_good_work_by_non_teachers": ["p331d","p331e","p331f","p331g"],
    "t_extent_of_work_hampered":      ["p27b","p27c","p27d","p27e","p27f","p27g","p27h"],
}

TEACHER_AGG_SUM = {
    "t_number_of_individual_training_topics": [
        "p18a","p18b","p18c","p18d","p18e","p18f","p18g","p18h","p18i",
    ]
}

TEACHER_DROP = [
    "p24a","p24b","p24c","p24d","p24e","p24f","p24g","p24h","p24i","p24j","p24k",
    "p16a","p16b","p16c","p16d","p16e","p16f","p16g","p16h",
    "p27a","p27b","p27c","p27d","p27e","p27f","p27g","p27h",
    "p29a","p29b","p29c","p29d","p29e","p299d",
    "p30a","p30b","p30c",
    "p34a","p34b","p34c","p34d","p34e","p34f","p34g",
    "p31d","p311a","p311b","p311c","p311e","p311f","p311g","p311h",
    "p21a","p21b","p21c","p21d","p21e","p21f",
    "p22a","p22b","p22c","p22d","p22e","p22f","p22g",
    "p23a","p23b","p23c","p23d","p23e","p23f","p23g","p23h","p23i",
    "p12a","p12b","p12c","p12d",
    "p32a","p32b","p32c","p32d","p32e",
    "p331a","p331b","p331c","p331d","p331e","p331f","p331g","p331j",
]

# Cols to rename (raw to readable, from dataset code)
RENAME_MAP = {
    # student
    "a1":   "s_gender",
    "a4":   "s_birth_year",
    "repeater": "s_has_repeated",
    "a6nm": "s_frequency_of_skips",
    "country_iso_cnac": "s_birth_country",
    "country_iso_nac":  "s_nazionality_country",
    "weight": "s_weight",
    # family
    "f0":   "f_respondent",
    "f1n":  "f_number_of_people_in_household",
    "f2an": "f_mother_age",
    "f2bn": "f_father_age",
    "f3a":  "f_mother_education_level",
    "f3b":  "f_father_education_level",
    "f4a":  "f_mother_employment_status",
    "f4b":  "f_father_employment_status",
    "f5a":  "f_mother_place_of_birth",
    "f5b":  "f_father_place_of_birth",
    "f5n":  "f_student_place_of_birth",
    "f6":   "f_years_in_spanish_education",
    "f7":   "f_language_spoken_at_home",
    "f10n": "f_number_of_tech_at_home",
    "f11":  "f_extent_of_books_at_home",
    "start_schooling_age": "f_start_schooling_age",
    "f14a": "f_visits_in_school_by_mother",
    "f14b": "f_visits_in_school_by_father",
    "f20":  "f_has_been_recommended_school",
    "f21n": "f_number_of_homework_hours_a_week",
    "f23":  "f_parental_education_expectations",
    "f30":  "f_number_of_children_in_household",
    "f31":  "f_type_of_family_unit",
    "f34":  "f_monthly_household_income",
    "ESCS": "f_ESCS",
    "mother_occupation": "f_mother_occupation",
    "father_occupation": "f_father_occupation",
    # teacher
    "p2":   "t_gender",
    "p2n":  "t_age",
    "p3n":  "t_years_as_teacher",
    "p4n":  "t_years_in_school",
    "p6n":  "t_students_in_group",
    "p7fn": "t_students_disadvantaged",
    "p8an": "t_foreign_students_spanish",
    "p8bn": "t_foreign_students_not_spanish",
    "p10n": "t_teaching_hours_per_week",
    "p11":  "t_average_explanation_time",
    "p20":  "t_individual_training_incidence",
    "p25":  "t_seat_configuration",
    "p26":  "t_behaviour_problems_solution",
    "p28n": "t_meetings_with_families",
    "p141": "t_enrolled_in_school_training_plan",
    "p171n":"t_training_hours_last_six_years",
    "p172n":"t_training_hours_ceu_offer",
    "pfc":  "t_main_topic_of_pfc",
}

# raw int -> readable string
RECODE_MAP = {
    "f_respondent": {1: "MOTHER", 2: "FATHER", 3: "OTHER"},
    "f_mother_place_of_birth": {
        1: "CANARY_ISLANDS", 2: "SPAIN_NO_CANARY", 3: "ANOTHER_EU", 4: "ANOTHER_NON_EU"
    },
    "f_father_place_of_birth": {
        1: "CANARY_ISLANDS", 2: "SPAIN_NO_CANARY", 3: "ANOTHER_EU", 4: "ANOTHER_NON_EU"
    },
    "f_student_place_of_birth": {
        1: "CANARY_ISLANDS", 2: "SPAIN_NO_CANARY", 3: "ANOTHER_EU", 4: "ANOTHER_NON_EU"
    },
    "f_language_spoken_at_home": {1: "SPANISH", 2: "OTHER"},
    "f_type_of_family_unit": {
        1: "MOTHER_FATHER_CHILDREN", 2: "MOTHER_PARTNER_CHILDREN",
        3: "FATHER_PARTNER_CHILDREN", 4: "MOTHER_CHILDREN",
        5: "FATHER_CHILDREN", 6: "RELATIVES_CHILDREN", 7: "OTHERS",
    },
    "f_monthly_household_income": {
        1: "NO_INCOME", 2: "UP_TO_500", 3: "UP_TO_1000", 4: "UP_TO_1500",
        5: "UP_TO_2000", 6: "UP_TO_2500", 7: "UP_TO_3000", 8: "UP_TO_3500",
        9: "MORE_THAN_3500", 10: "NO_ANSWER",
    },
    "f_parental_education_expectations": {
        1: "4_ESO", 2: "INT_FP", 3: "BACH_ATO", 4: "UP_FP",
        5: "BACH_DEG", 9: "DONT_KNOW",
    },
    "f_visits_in_school_by_mother": {
        1: "NEVER", 2: "SOMETIMES", 3: "ONCE_PER_MONTH",
        4: "ONCE_PER_WEEK", 5: "DONT_KNOW",
    },
    "f_visits_in_school_by_father": {
        1: "NEVER", 2: "SOMETIMES", 3: "ONCE_PER_MONTH",
        4: "ONCE_PER_WEEK", 5: "DONT_KNOW",
    },
    "s_gender":   {1: "MALE", 2: "FEMALE"},
    "t_gender":   {1: "MALE", 2: "FEMALE"},
    "t_behaviour_problems_solution": {
        1: "PRINCIPAL", 2: "MANAGEMENT", 3: "CLASSMATES", 4: "INDIVIDUALLY"
    },
}

ESCS_BINS   = [-np.inf, -2, -1, 0, 1, 2, np.inf]
ESCS_LABELS = ["VERY_LOW", "LOW", "BELOW_AVG", "ABOVE_AVG", "HIGH", "VERY_HIGH"]

DEFAULT_NAN_THRESHOLD = 0.5


# Helper functions for aggregations
def _get_good_bad_agg(row, good_cols, bad_cols, max_degree_of_agreement=4):
    good_present = [c for c in good_cols if c in row.index]
    bad_present = [c for c in bad_cols if c in row.index]
    
    no_good = row[good_present].notna().sum()
    no_bad = row[bad_present].notna().sum()
    
    if no_good == 0 and no_bad == 0:
        return np.nan
    
    sum_good = row[good_present].sum()
    sum_bad = row[bad_present].sum()
    
    return (sum_good + (no_bad * (max_degree_of_agreement + 1)) - sum_bad) / (no_good + no_bad)


def _custom_binary_agg(series):
    binary = []
    for x in series:
        if x == 1:
            binary.append(1)
        elif x == 2:
            binary.append(0)
        else:
            binary.append(x)
    if any(pd.isna(v) for v in binary):
        return np.nan
    return sum(val * (2**idx) for idx, val in enumerate(reversed(binary)))


def _drop_if_present(df, cols, reason=""):
    to_drop = [c for c in cols if c in df.columns]
    if to_drop and reason:
        logger.info(f"  Dropping {len(to_drop)} {reason} cols")
    return df.drop(columns=to_drop)


def _aggregate_mean(df, agg_map):
    """For each entry in agg_map, create a new column as row-wise mean (skipna)."""
    for new_col, src_cols in agg_map.items():
        present = [c for c in src_cols if c in df.columns]
        if present:
            df[new_col] = df[present].mean(axis=1, skipna=True)
    return df


def _aggregate_sum(df, agg_map):
    """For each entry in agg_map, create a new column as row-wise sum (min_count=1)."""
    for new_col, src_cols in agg_map.items():
        present = [c for c in src_cols if c in df.columns]
        if present:
            df[new_col] = df[present].sum(axis=1, min_count=1)
    return df


def _normalize_col(series, old_min, old_max, new_min=0, new_max=1):
    return ((series - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min


def _normalize_extent_frequency_cols(df):
    """Normalize all extent_of_* and frequency_of_* cols to [0, 1]."""
    target_prefixes = (
        "extent_of_", "frequency_of_", 
        "f_extent_", "f_frequency_",
        "t_extent_", "t_frequency_",
        "s_extent_", "s_frequency_"
    )
    for col in df.columns:
        if any(col.startswith(p) for p in target_prefixes):
            col_min, col_max = df[col].min(), df[col].max()
            if col_max > col_min:  # avoid division by zero
                df[col] = _normalize_col(df[col], col_min, col_max)
    return df


def nan_summary(df):
    counts = df.isna().sum()
    fracs  = counts / len(df)
    return (
        pd.DataFrame({"nan_count": counts, "nan_fraction": fracs})
        .sort_values("nan_fraction", ascending=False)
        .query("nan_count > 0")
    )


def bin_escs(series):
    return pd.cut(series, bins=ESCS_BINS, labels=ESCS_LABELS)


def fit_feature_encoders(X_train: pd.DataFrame):
    """Fit OrdinalEncoder on categorical cols, StandardScaler on numerical."""
    cat_cols = X_train.select_dtypes(include="object").columns.tolist()
    num_cols = X_train.select_dtypes(include="number").columns.tolist()

    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    scaler = StandardScaler()

    if cat_cols:
        enc.fit(X_train[cat_cols])
    if num_cols:
        scaler.fit(X_train[num_cols])

    return enc, scaler, cat_cols, num_cols


def apply_feature_encoders(X: pd.DataFrame, enc, scaler, cat_cols, num_cols) -> pd.DataFrame:
    """Apply fitted OrdinalEncoder and StandardScaler. Returns a DataFrame."""
    X_out = X.copy()
    present_cat = [c for c in cat_cols if c in X_out.columns]
    present_num = [c for c in num_cols if c in X_out.columns]

    if present_cat:
        X_out[present_cat] = enc.transform(X_out[present_cat])
    if present_num:
        X_out[present_num] = scaler.transform(X_out[present_num])

    # Fill any remaining NaNs (sensitive cols not imputed in Goal 1) with 0
    X_out = X_out.fillna(0.0)
    return X_out.astype(float)


# Main pipeline 

def preprocess(
    df: pd.DataFrame,
    target_col= "level_MAT",
    nan_threshold = DEFAULT_NAN_THRESHOLD,
    drop_other_scores = True,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Preprocess original.csv into (X, y, sensitive).

    Steps
    -----
    1.  Drop identifiers, availability flags, redundant cols
    2.  Drop rows with >90% NaN
        2b. Student aggregations (Repo B style)
    3.  Family questionnaire aggregations
    4.  Teacher questionnaire aggregations & drops
    5.  Rename remaining columns to readable names
    6.  Recode categorical integers to strings & process boolean columns
    7.  Normalize extent_of_* / frequency_of_* to [0,1]
    8.  Drop columns above NaN threshold
    
    9.  Extract target and sensitive attribute
    """
    logger.info(f"Input shape: {df.shape}")
    df = df.copy()

    #structural drops
    logger.info("structural drops")
    df = _drop_if_present(df, IDENTIFIER_COLS,   "identifier")
    df = _drop_if_present(df, AVAILABILITY_COLS, "availability-flag")
    df = _drop_if_present(df, REDUNDANT_COLS,    "redundant")

    # drop rows with >90% missing
    logger.info(" row-level NaN filter")
    row_nan_frac = df.isna().mean(axis=1)
    n_before = len(df)
    df = df[row_nan_frac <= 0.9]
    logger.info(f"  Dropped {n_before - len(df)} rows with >90% NaN")

    #Student aggregations ( like code from the repo/thing of the data)
    logger.info(" student aggregations")
    student_mean_map = {
        "s_frequency_of_computer_usage": ["a8a", "a8b", "a8c"],
        "s_frequency_of_internet_usage": ["a9a", "a9b", "a9c", "a9d", "a9e", "a9g"],
        "s_frequency_of_work_with_teachers": ["a10a", "a10b", "a10c", "a10d", "a10e", "a10f", "a10g", "a10h", "a10i", "a10j"],
        "s_frequency_of_materials_in_class": ["a11a", "a11b", "a11c", "a11d", "a11e", "a11f", "a11h"],
        "s_frequency_of_evaluations": ["a12a", "a12b", "a12c", "a12d", "a12e", "a12f", "a12g", "a12h", "a12i"],
        "s_extent_of_teacher_performance": ["a15a", "a15b", "a15c", "a15d", "a15e", "a15f", "a15g", "a15h", "a15i", "a15j"],
        "s_extent_of_class_env": ["a16a", "a16b", "a16c", "a16d", "a16e", "a16f", "a16g", "a16h"]
    }
    for new_col, src_cols in student_mean_map.items():
        present = [c for c in src_cols if c in df.columns]
        if present:
            df[new_col] = df[present].mean(axis=1, skipna=True)

    student_mixed_map = {
        "s_extent_of_classmates_affinity": {
            "good": ["a14a", "a14b", "a14g", "a14h"],
            "bad": ["a14d", "a14e", "a14f"]
        },
        "s_extent_of_school_satisfaction": {
            "good": ["a17a", "a17b", "a17c", "a17e", "a17h"],
            "bad": ["a17d"]
        },
        "s_extent_of_math_affinity": {
            "good": ["a20a", "a20e"],
            "bad": ["a20b", "a20c", "a20d"]
        },
        "s_extent_of_reading_affinity": {
            "good": ["a21b"],
            "bad": ["a21c", "a21d", "a21e"]
        }
    }
    for new_col, groups in student_mixed_map.items():
        good_present = [c for c in groups["good"] if c in df.columns]
        bad_present = [c for c in groups["bad"] if c in df.columns]
        if good_present or bad_present:
            df[new_col] = df.apply(lambda r: _get_good_bad_agg(r, good_present, bad_present, 4), axis=1)

    student_agg_cols_to_drop = [
        "a8a", "a8b", "a8c",
        "a9a", "a9b", "a9c", "a9d", "a9e", "a9f", "a9g",
        "a10a", "a10b", "a10c", "a10d", "a10e", "a10f", "a10g", "a10h", "a10i", "a10j", "a10k", "a10l", "a10m", "a10n",
        "a11a", "a11b", "a11c", "a11d", "a11e", "a11f", "a11g", "a11h",
        "a12a", "a12b", "a12c", "a12d", "a12e", "a12f", "a12g", "a12h", "a12i",
        "a13a", "a13b", "a13c", "a13d",
        "a14a", "a14b", "a14c", "a14d", "a14e", "a14f", "a14g", "a14h",
        "a141g", "a144d", "a144h", "a166f", "a177d",
        "a15a", "a15b", "a15c", "a15d", "a15e", "a15f", "a15g", "a15h", "a15i", "a15j",
        "a16a", "a16b", "a16c", "a16d", "a16e", "a16f", "a16g", "a16h", "a16i", "a16j", "a16k", "a16l",
        "a17a", "a17b", "a17c", "a17d", "a17e", "a17f", "a17g", "a17h", "a171h",
        "a20a", "a20b", "a20c", "a20d", "a20e",
        "a21a", "a21b", "a21c", "a21d", "a21e", "a211a",
        "a22a", "a22b", "a22c", "a22d", "a222b",
        "a23a", "a23b", "a23c", "a23d", "a23e", "a23f", "a23g", "a23h", "a23i", "a23j", "a23k",
        "a24", "a40a", "a40b", "a40c", "a40d", "a111a", "a160k", "a162k", "a163k", "a166k"
    ]
    df = _drop_if_present(df, student_agg_cols_to_drop, "student-agg-source")

    # Family aggregations
    logger.info("family aggregations")
    df = _aggregate_mean(df, FAMILY_AGG_MEAN)
    df = _drop_if_present(df, FAMILY_AGG_DROP, "family-agg-source")

    #  Teacher aggregations
    logger.info(" teacher aggregations")
    # rescale inconsistent scale cols before aggregating
    for col in ["p331g", "p331j"]:
        if col in df.columns:
            df[col] = _normalize_col(df[col], old_min=1, old_max=5, new_min=1, new_max=4)

    # teacher drop logic directly from B to ensure parity
    pfc_topics = ["p15a","p15b","p15c","p15d","p15e","p15f","p15g","p15h","p15i"]
    class_problems = ["p26a","p26b","p26c","p26d"]
    cols_to_drop = ["p27a", "p16h", "p19", "p23i", "p32e", "p41d", "p41e", "p41f", "p41j", "p299d", "p331j"]
    cols_to_drop_dep = ["p12a", "p12c", "p13b", "p16d", "p16e", "p16b", "p16f", "p18c", "p18b", "p22b", "p22c", "p32d", "p32a", "p32c", "p34f", "p34e", "p34c", "p34a", "p34g", "p311h", "p311f", "p311g", "p331c", "p331b", "p331a"]
    
    df = _drop_if_present(df, pfc_topics + class_problems + cols_to_drop + cols_to_drop_dep, "teacher-pre-agg-drop")

    #drop whichever of p5/rep has more NaNs
    if "p5" in df.columns and "rep" in df.columns:
        drop_col = "p5" if df["p5"].isna().sum() > df["rep"].isna().sum() else "rep"
        df = df.drop(columns=[drop_col])
        logger.info(f"  Dropped redundant repeater col: {drop_col}")

    # perform teacher mean aggregations
    df = _aggregate_mean(df, TEACHER_AGG_MEAN)
    df = _drop_if_present(df, TEACHER_DROP, "teacher-agg-source/dep")

    # sum-based training indicators
    df = _aggregate_sum(df, TEACHER_AGG_SUM)
    p18_cols = ["p18a","p18b","p18c","p18d","p18e","p18f","p18g","p18h","p18i"]
    df = _drop_if_present(df, p18_cols, "p18-agg-source")

    # custom bitmask-based subjects taught indicator
    p9_cols = ["p9a","p9b","p9c","p9d","p9e","p9f"]
    if all(c in df.columns for c in p9_cols):
        df["t_number_of_subjects_taught"] = df.apply(lambda r: _custom_binary_agg(r[p9_cols]), axis=1)
    df = _drop_if_present(df, p9_cols, "p9-agg-source")

    # mixed satisfaction indicators with direction scaling
    p41_good = ["p41a", "p41b", "p41g", "p41h", "p41i"]
    p41_bad = ["p41c"]
    df["t_extent_of_satisfaction_job_and_school"] = df.apply(lambda r: _get_good_bad_agg(r, p41_good, p41_bad, 4), axis=1)
    df = _drop_if_present(df, p41_good + p41_bad, "p41-agg-source")

    p13_good = ["p13"]
    p13_bad = ["p13c"]
    df["t_extent_of_results_satisfaction"] = df.apply(lambda r: _get_good_bad_agg(r, p13_good, p13_bad, 4), axis=1)
    df = _drop_if_present(df, p13_good + p13_bad, "p13-agg-source")

    # rename
    logger.info("renaming columns")
    df = df.rename(columns={k: v for k, v in RENAME_MAP.items() if k in df.columns})

    #Recode categoricals
    logger.info("recoding categoricals")
    for col, mapping in RECODE_MAP.items():
        if col in df.columns:
            df[col] = df[col].map(lambda x: mapping.get(x, np.nan) if pd.notna(x) else np.nan)

    # boolean mapping
    for col in ["f_has_been_recommended_school", "t_enrolled_in_school_training_plan"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: False if x == 2 else (True if x == 1 else np.nan))

    if "s_has_repeated" in df.columns:
        df["s_has_repeated"] = df["s_has_repeated"].apply(lambda x: False if x == 1 else (True if x == 2 else np.nan))

    #Normalize extent/frequency cols
    logger.info(" normalizing extent/frequency columns")
    df = _normalize_extent_frequency_cols(df)

    #drop high-NaN columns
    logger.info(f" dropping columns above {nan_threshold:.0%} NaN threshold")
    nan_fracs = nan_summary(df)["nan_fraction"]
    high_nan  = nan_fracs[nan_fracs > nan_threshold].index.tolist()
    logger.info(f"  Dropping {len(high_nan)} columns")
    df = _drop_if_present(df, high_nan, "high-NaN")

    #extract target
    if target_col not in df.columns:
        raise ValueError(f"Target '{target_col}' not found. Available score cols: {[c for c in ALL_SCORE_COLS if c in df.columns]}")
    y  = df[target_col].copy()
    df = df.drop(columns=[target_col])

    if drop_other_scores:
        df = _drop_if_present(df, [c for c in ALL_SCORE_COLS if c != target_col], "score-leakage")

    #drop rows where target is NaN
    valid = y.notna()
    logger.info(f"  Dropping {(~valid).sum()} rows with NaN target")
    df, y = df[valid], y[valid]

    # extract sensitive attribute
    sensitive_col = "f_ESCS"
    sensitive = df[sensitive_col].copy() if sensitive_col in df.columns else pd.Series(np.nan, index=df.index, name=sensitive_col)

    logger.info(f"Done. X={df.shape}  target={target_col}  ESCS NaN={sensitive.isna().sum()}")
    return df, y, sensitive


def build_percentile_target(
    score_train: pd.Series,
    score_test: pd.Series,
    percentile = 75,
    column_name = "target_high_perf",
) -> tuple[pd.Series, pd.Series, float]:
    """
    Construct the binary "excellence" target by thresholding a continuous
    performance score.
    """
    threshold = float(np.percentile(score_train.dropna(), percentile))
    y_train = (score_train >= threshold).astype(int).rename(column_name)
    y_test  = (score_test  >= threshold).astype(int).rename(column_name)

    logger.info(
        f"Binary target '{column_name}': threshold={threshold:.3f} "
        f"(p{percentile} of train split only), "
        f"train positive rate={y_train.mean():.1%}, "
        f"test positive rate={y_test.mean():.1%}"
    )
    return y_train, y_test, threshold


@dataclass 
class DataSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    protected_train: pd.DataFrame
    protected_test: pd.DataFrame
    feature_names: list[str] = field(default_factory=list)
    target_name: str = ""
    protected_attrs: list[str] = field(default_factory=list)
    encoders: dict[str, Any] = field(default_factory=dict)
    scaler: Any = None


def save_split(split, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(split, f)
    logger.info(f"DataSplit saved to {path}.")


def load_split(path) -> DataSplit:
    with open(path, "rb") as f:
        split = pickle.load(f)
    logger.info(f"DataSplit loaded from {path}.")
    return split