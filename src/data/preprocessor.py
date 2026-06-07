from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.utils import get_logger, load_config

logger = get_logger(__name__)


import numpy as np
import pandas as pd
from src.utils import get_logger

logger = get_logger(__name__)

# ── Column taxonomy (raw original.csv names) ───────────────────────────────────

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

# Sensitive attributes (raw names in original.csv)
SENSITIVE_COLS = [
    "ESCS",
    "f34",           # monthly household income
    "f3a", "f3b",    # mother/father education level
    "mother_occupation", "father_occupation",
    "f4a", "f4b",    # mother/father employment status
    "f5a", "f5b", "f5n",  # place of birth
    "f7",            # language spoken at home
    "f31",           # type of family unit
    "a1",            # student gender
    "country_iso_cnac", "country_iso_nac",  # nationality
]

# ── Aggregation maps (raw column names) ────────────────────────────────────────
# Each entry: new_col_name -> list of source cols to average (skipna=True)

FAMILY_AGG_MEAN = {
    "frequency_of_books_at_home":   ["f9b", "f9c", "f9h"],
    "frequency_of_tech_at_home":    ["f9d", "f9f"],
    "frequency_of_see_adult_read":  ["f12a", "f12b"],
    "extent_of_interest_in_interview": ["f15a", "f15b", "f15d", "f15f"],
    "frequency_of_support_at_home": ["f16a", "f16b", "f16c", "f16d", "f16e", "f16f"],
    "frequency_of_parent_involved_in_school": ["f17a", "f17b", "f17c", "f17d"],
    "extent_of_family_satisfaction": ["f18a", "f18b", "f18e", "f18f", "f18g", "f18h"],
    "extent_of_teacher_satisfaction": ["f19a", "f19b", "f19c", "f19d", "f19e"],
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
    "extent_of_evaluation_variety": [
        "p24a","p24b","p24c","p24d","p24e",
        "p24f","p24g","p24h","p24i","p24j","p24k",
    ],
    "extent_of_pfc_incidence":      ["p16a","p16c","p16g"],
    "extent_of_family_interest":    ["p29a","p29b","p29c","p29d","p29e"],
    "extent_of_family_support":     ["p30a","p30b","p30c"],
    "extent_of_work_facilitated":   ["p34b","p34d"],
    "extent_of_positive_relationships": ["p31d","p311a","p311b","p311c","p311e"],
    "extent_of_student_involvement":["p21a","p21b","p21c","p21d","p21e","p21f"],
    "extent_of_teaching_methods":   ["p22a","p22d","p22e","p22f","p22g"],
    "extent_of_resource_variety":   ["p23a","p23b","p23c","p23d","p23e","p23f","p23g","p23h"],
    "extent_of_class_behaviour":    ["p12b","p12d"],
    "extent_of_opinion_on_school":  ["p32b"],
    "extent_of_good_work_by_non_teachers": ["p331d","p331e","p331f"],
    "extent_of_work_hampered":      ["p27b","p27c","p27d","p27e","p27f","p27g","p27h"],
    "extent_of_satisfaction":       ["p41a","p41b","p41c","p41g","p41h","p41i"],
    "extent_of_results_satisfaction":["p13","p13c"],
}

TEACHER_AGG_SUM = {
    "number_of_individual_training_topics": [
        "p18a","p18b","p18c","p18d","p18e","p18f","p18g","p18h","p18i",
    ],
    "number_of_subjects_taught": ["p9a","p9b","p9c","p9d","p9e","p9f"],
}

# Cols to drop after teacher aggregation (functional deps + high NaN + aggregated)
TEACHER_DROP = [
    # pfc topics
    "p15a","p15b","p15c","p15d","p15e","p15f","p15g","p15h","p15i",
    # class problem detail
    "p26a","p26b","p26c","p26d",
    # high NaN
    "p27a","p16h","p19","p23i","p32e","p41d","p41e","p41f","p41j","p299d","p331j",
    # functional dependencies
    "p12a","p12c","p13b","p16d","p16e","p16b","p16f","p18c","p18b",
    "p22b","p22c","p32d","p32a","p32c","p34f","p34e","p34c","p34a","p34g",
    "p311h","p311f","p311g","p331c","p331b","p331a",
    # scale-inconsistent cols (rescaled before agg, then drop originals)
    "p331g",
]

# Cols to rename (raw -> readable)
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

# Categorical recoding maps (raw int -> readable string)
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
ALL_SCORE_COLS = ["score_MAT","level_MAT","score_LEN","level_LEN","score_ING","level_ING"]



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
    target_prefixes = ("extent_of_", "frequency_of_", "f_extent_", "f_frequency_",
                       "t_extent_", "t_frequency_")
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




@dataclass #NOTE
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

def save_split(split: DataSplit, path):
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
















class Preprocessor:
    def __init__(self, config: dict | None = None) -> None:
        self.cfg = config or load_config()
        self._ds_cfg = self.cfg["dataset"]
        self.target_col = self._ds_cfg.get("target_column")
        self.protected_attrs = self._ds_cfg.get("protected_attrs", [])
        self.test_size = self._ds_cfg.get("test_size", 0.20)
        self.random_seed = self._ds_cfg.get("random_seed", 42)
        self._encoders: dict[str, LabelEncoder] = {}
        self._scaler: StandardScaler = None
        self._feature_names = []

    def fit_transform(self, df) -> DataSplit:
        df = df.copy()
        df = self._drop_duplicates(df)
        df = self._infer_and_validate_columns(df)
        df = self._encode_categoricals(df, fit=True)

        X, y, protected = self._split_xy(df)

        X_train, X_test, y_train, y_test, prot_train, prot_test = train_test_split(X, y, protected, test_size=self.test_size,
            random_state=self.random_seed,
            stratify=y,
        )

        X_train, X_test = self._scale(X_train, X_test, fit=True)
        split = DataSplit(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            protected_train=prot_train.reset_index(drop=True),
            protected_test=prot_test.reset_index(drop=True),
            feature_names=self._feature_names,
            target_name=self.target_col,
            protected_attrs=self.protected_attrs,
            encoders=self._encoders,
            scaler=self._scaler,
        )


        logger.info( # WE gotta log it up (in we gotta pump it up rhythm)
            f"Split: {len(X_train):,} train / {len(X_test):,} test rows "
            f"| {len(self._feature_names)} features."
        )
        
        return split
    

    def transform(self, df):
        if not self._encoders and self._scaler is None:
            raise RuntimeError("Preprocessor has not been fitted yet.")
        df = df.copy()
        df = self._encode_categoricals(df, fit=False)
        X, _, _ = self._split_xy(df)
        X_scaled, _ = self._scale(X, X, fit=False)
        return X_scaled
    
    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        
        logger.info(f"Preprocessor saved to {path}.")


    @classmethod
    def load(cls, path) -> "Preprocessor":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        
        logger.info(f"Preprocessor loaded from {path}.")
        
        return obj

    
    def _drop_duplicates(self, df):
        before = len(df)
        df = df.drop_duplicates()
        dropped = before - len(df)
        if dropped:
            logger.info(f"Dropped {dropped} duplicate rows.")
        return df

    def _encode_categoricals(self, df, *, fit):
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if self.target_col in cat_cols:
            cat_cols.remove(self.target_col)
        for col in cat_cols:
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self._encoders[col] = le
            else:
                le = self._encoders.get(col)
                if le is None:
                    raise KeyError(f"no fitted encoder for column '{col}'.")
                df[col] = le.transform(df[col].astype(str))
        return df


    def _split_xy(self, df) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        protected = df[self.protected_attrs] if self.protected_attrs else pd.DataFrame()
        drop_cols = [self.target_col] + self.protected_attrs
        X = df.drop(columns=[c for c in drop_cols if c in df.columns])
        y = df[self.target_col]
        self._feature_names = X.columns.tolist()
        return X, y, protected

    def _scale(self, X_train, X_test, *, fit: bool):
        num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
        if fit:
            self._scaler = StandardScaler()
            X_train[num_cols] = self._scaler.fit_transform(X_train[num_cols])
        else:
            X_train[num_cols] = self._scaler.transform(X_train[num_cols])
        X_test[num_cols] = self._scaler.transform(X_test[num_cols])
        return X_train.reset_index(drop=True), X_test.reset_index(drop=True)

