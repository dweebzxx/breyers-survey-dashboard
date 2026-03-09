"""Data loading and preprocessing for the Breyers survey dashboard."""

import pandas as pd
import numpy as np

LABEL_MAPS = {
    "Q1_Consent": {1: "Agree", 2: "Do not agree"},
    "Q2_PurchaseRecent": {1: "Yes", 2: "No"},
    "Q3_DecisionRole": {1: "I mostly decide", 2: "I share the decision"},
    "Q4_PurchaseFreq": {1: "Weekly", 2: "2-3x/month", 3: "Monthly", 4: "Less often"},
    "Q5_UsualChannel": {1: "Grocery store", 4: "Online delivery"},
    "Q7_BrandMostOften": {
        1: "Breyers", 2: "Ben & Jerry's", 3: "Häagen-Dazs",
        4: "Halo Top", 5: "Enlightened", 6: "Nick's",
        7: "Store brand", 8: "Local/regional brand", 9: "Other",
    },
    "Q8_AttrImportance_1": {1: "Not at all", 2: "Slightly", 3: "Moderately", 4: "Very", 5: "Extremely"},
    "Q8_AttrImportance_2": {1: "Not at all", 2: "Slightly", 3: "Moderately", 4: "Very", 5: "Extremely"},
    "Q8_AttrImportance_3": {1: "Not at all", 2: "Slightly", 3: "Moderately", 4: "Very", 5: "Extremely"},
    "Q8_AttrImportance_4": {1: "Not at all", 2: "Slightly", 3: "Moderately", 4: "Very", 5: "Extremely"},
    "Q8_AttrImportance_5": {1: "Not at all", 2: "Slightly", 3: "Moderately", 4: "Very", 5: "Extremely"},
    "Q8_AttrImportance_6": {1: "Not at all", 2: "Slightly", 3: "Moderately", 4: "Very", 5: "Extremely"},
    "Q8_AttrImportance_7": {1: "Not at all", 2: "Slightly", 3: "Moderately", 4: "Very", 5: "Extremely"},
    "Q9_Tradeoff": {1: "Higher protein", 2: "Lower sugar", 3: "Neither/taste"},
    "Q10_ActiveSeeking": {1: "Low/zero sugar", 2: "High-protein", 3: "Both", 4: "Neither"},
    "Q11_Appeal": {1: "Not at all", 2: "Slightly", 3: "Moderately", 4: "Very", 5: "Extremely"},
    "Q12_PurchaseIntent": {
        1: "Def. not buy", 2: "Prob. not buy", 3: "Might/might not",
        4: "Prob. buy", 5: "Def. buy",
    },
    "Q13_Replacement": {
        1: "Replace usual", 2: "In addition",
        3: "Replace different dessert", 4: "Not sure",
    },
    "Q13A_WhatReplaced": {
        1: "Regular Breyers", 2: "Another brand", 3: "Better-for-you brand",
        4: "Another dessert", 5: "Skip dessert", 6: "Not sure",
    },
    "Q14_InterestComparison": {
        1: "Much less", 2: "Somewhat less", 3: "About the same",
        4: "Somewhat more", 5: "Much more",
    },
    "Q15_AttentionCheck_1": {
        1: "Strongly disagree", 2: "Somewhat disagree", 3: "Neither",
        4: "Somewhat agree", 5: "Strongly agree",
    },
    "Q16_PurchaseLocation": {
        1: "Grocery store", 2: "Club/wholesale",
        3: "Online delivery", 4: "Convenience store",
    },
    "Q17a_Price399": {1: "Very unlikely", 2: "Unlikely", 3: "Neither", 4: "Likely", 5: "Very likely"},
    "Q17b_Price499": {1: "Very unlikely", 2: "Unlikely", 3: "Neither", 4: "Likely", 5: "Very likely"},
    "Q17c_Price599": {1: "Very unlikely", 2: "Unlikely", 3: "Neither", 4: "Likely", 5: "Very likely"},
    "Q17d_Price699": {1: "Very unlikely", 2: "Unlikely", 3: "Neither", 4: "Likely", 5: "Very likely"},
    "Q17e_Price799": {1: "Very unlikely", 2: "Unlikely", 3: "Neither", 4: "Likely", 5: "Very likely"},
    "Q19_ClubStore4Pack": {
        1: "Def. not", 2: "Prob. not", 3: "Might/might not",
        4: "Prob. would", 5: "Def. would",
    },
    "Q20_OnlineDelivery": {1: "Very unlikely", 2: "Unlikely", 3: "Neither", 4: "Likely", 5: "Very likely"},
    "Q21_DietFocus": {1: "Limit sugar", 2: "Increase protein", 3: "Both", 4: "Neither"},
    "Q22_HouseholdType": {
        1: "Live alone", 2: "Spouse/partner, no children",
        3: "Spouse/partner + children", 4: "Single parent", 6: "Other",
    },
    "Q23_Age": {1: "18-24", 2: "25-34", 3: "35-44", 4: "45-54", 5: "55-64", 6: "65+"},
    "Q24_Income": {
        1: "<$25k", 2: "$25-49k", 3: "$50-74k", 4: "$75-99k",
        5: "$100-149k", 6: "$150k+", 7: "Prefer not to say",
    },
    "ClaimCell": {1: "High Protein", 2: "Low Sugar", 3: "Both"},
}

Q6_BRAND_MAP = {
    1: "Breyers",
    2: "Ben & Jerry's",
    4: "Halo Top",
    5: "Enlightened",
    6: "Nick's",
    7: "Store brand",
    8: "Local/regional brand",
}

Q6_BRAND_COL_MAP = {
    1: "Q6_Brand_Breyers",
    2: "Q6_Brand_BenJerrys",
    4: "Q6_Brand_HaloTop",
    5: "Q6_Brand_Enlightened",
    6: "Q6_Brand_Nicks",
    7: "Q6_Brand_StoreBrand",
    8: "Q6_Brand_LocalRegional",
}

NUMERIC_COLS = [
    "Q1_Consent", "Q2_PurchaseRecent", "Q3_DecisionRole", "Q4_PurchaseFreq",
    "Q5_UsualChannel", "Q7_BrandMostOften",
    "Q8_AttrImportance_1", "Q8_AttrImportance_2", "Q8_AttrImportance_3",
    "Q8_AttrImportance_4", "Q8_AttrImportance_5", "Q8_AttrImportance_6",
    "Q8_AttrImportance_7", "Q8h_QualityCheck_Trap1",
    "Q9_Tradeoff", "Q10_ActiveSeeking", "Q11_Appeal", "Q12_PurchaseIntent",
    "Q13_Replacement", "Q13A_WhatReplaced", "Q14_InterestComparison",
    "Q15_AttentionCheck_1", "Q16_PurchaseLocation",
    "Q17a_Price399", "Q17b_Price499", "Q17c_Price599", "Q17d_Price699", "Q17e_Price799",
    "Q19_ClubStore4Pack", "Q20_OnlineDelivery",
    "Q21_DietFocus", "Q22_HouseholdType", "Q23_Age", "Q24_Income", "ClaimCell",
    "Progress", "Duration (in seconds)",
]

Q18_OUTLIER_CAP_PERCENTILE = 0.99

ATTR_LABELS = {
    "Q8_AttrImportance_1": "Taste",
    "Q8_AttrImportance_2": "Price",
    "Q8_AttrImportance_3": "Brand reputation",
    "Q8_AttrImportance_4": "Low/zero sugar",
    "Q8_AttrImportance_5": "High protein",
    "Q8_AttrImportance_6": "Clean ingredients",
    "Q8_AttrImportance_7": "Low calorie",
}


def load_data(filepath="breyers-survey-data-cleaned.csv") -> pd.DataFrame:
    """Load the survey CSV, skipping the question-text row (index 1)."""
    df = pd.read_csv(filepath, skiprows=[1], dtype=str)

    # Normalise blanks / literal "nan" → NaN
    df = df.replace({"": np.nan, "nan": np.nan, "NaN": np.nan})

    # Parse numeric survey columns
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Q18 as float, capped at 99th percentile
    if "Q18_PriceTooExpensive" in df.columns:
        df["Q18_PriceTooExpensive"] = pd.to_numeric(df["Q18_PriceTooExpensive"], errors="coerce")
        cap = df["Q18_PriceTooExpensive"].quantile(Q18_OUTLIER_CAP_PERCENTILE)
        df["Q18_PriceTooExpensive"] = df["Q18_PriceTooExpensive"].clip(upper=cap)

    # Expand Q6_BrandsBought multi-select → binary indicator columns
    for brand_code, col_name in Q6_BRAND_COL_MAP.items():
        df[col_name] = df["Q6_BrandsBought"].apply(
            lambda x: _has_brand(x, brand_code)
        ).astype("Int8")

    # ClaimCell label column
    df["ClaimCell_Label"] = df["ClaimCell"].map(LABEL_MAPS["ClaimCell"])

    return df


def _has_brand(val, code: int) -> int:
    """Return 1 if code appears in a comma-separated string, else 0."""
    if pd.isna(val):
        return 0
    try:
        codes = [int(c.strip()) for c in str(val).split(",") if c.strip()]
        return 1 if code in codes else 0
    except Exception:
        return 0


def get_label_maps() -> dict:
    """Return the full label-mapping dictionary."""
    return LABEL_MAPS


def apply_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with additional *_label columns for all coded fields."""
    df = df.copy()
    for col, mapping in LABEL_MAPS.items():
        if col in df.columns:
            df[f"{col}_label"] = df[col].map(mapping)
    return df
