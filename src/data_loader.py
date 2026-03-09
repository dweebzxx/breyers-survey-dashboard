"""
Data loader for Breyers Survey Dashboard.
Handles CSV loading, cleaning, and derived variable creation.
"""

import pandas as pd
from pathlib import Path
from typing import Tuple, Dict

# Define the path to the data file
DATA_PATH = Path(__file__).parent.parent / "data" / "breyers-survey-data-cleaned.csv"


def load_data() -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Load and clean the Breyers survey data.
    
    Returns:
        Tuple containing:
        - df: Cleaned DataFrame with respondent data
        - question_text: Dictionary mapping column names to full question text
    """
    # Read CSV with first row as header
    df_raw = pd.read_csv(DATA_PATH, header=0)
    
    # Extract Row 2 (question text) - this is index 0 after header parse
    question_row = df_raw.iloc[0]
    
    # Build question text dictionary
    question_text = {}
    for col in df_raw.columns:
        text = str(question_row[col])
        # Clean up the question text (remove newlines, extra spaces)
        text = ' '.join(text.split())
        question_text[col] = text
    
    # Drop Row 2 (question text row) from data
    df = df_raw.iloc[1:].copy()
    
    # Reset index
    df = df.reset_index(drop=True)
    
    # Define numeric columns that need type casting
    numeric_columns = [
        'Q1_Consent', 'Q2_PurchaseRecent', 'Q3_DecisionRole', 'Q4_PurchaseFreq',
        'Q5_UsualChannel', 'Q8_AttrImportance_1', 'Q8_AttrImportance_2',
        'Q8_AttrImportance_3', 'Q8_AttrImportance_4', 'Q8_AttrImportance_5',
        'Q8_AttrImportance_6', 'Q8_AttrImportance_7', 'Q8h_QualityCheck_Trap1',
        'Q9_Tradeoff', 'Q10_ActiveSeeking', 'Q11_Appeal', 'Q12_PurchaseIntent',
        'Q13_Replacement', 'Q13A_WhatReplaced', 'Q14_InterestComparison',
        'Q15_AttentionCheck_1', 'Q16_PurchaseLocation', 'Q17a_Price399',
        'Q17b_Price499', 'Q17c_Price599', 'Q17d_Price699', 'Q17e_Price799',
        'Q19_ClubStore4Pack', 'Q20_OnlineDelivery', 'Q21_DietFocus',
        'Q22_HouseholdType', 'Q23_Age', 'Q24_Income', 'ClaimCell'
    ]
    
    # Cast numeric columns to appropriate types
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Cast Q18_PriceTooExpensive to float (it's a continuous variable)
    if 'Q18_PriceTooExpensive' in df.columns:
        df['Q18_PriceTooExpensive'] = pd.to_numeric(df['Q18_PriceTooExpensive'], errors='coerce')
    
    # Create derived variables
    df = create_derived_variables(df)
    
    return df, question_text


def create_derived_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived variables for analysis.
    
    Args:
        df: DataFrame with raw survey data
        
    Returns:
        DataFrame with additional derived variables
    """
    # Top-2 Box Purchase Intent: 1 if Q12 is 4 or 5, else 0
    df['Top2Box_PI'] = (df['Q12_PurchaseIntent'] >= 4).astype(int)
    
    # Parse Q6_BrandsBought (comma-separated codes) into boolean columns
    df = parse_brands_bought(df)
    
    return df


def parse_brands_bought(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse the Q6_BrandsBought multi-select column into individual boolean columns.
    
    Args:
        df: DataFrame with Q6_BrandsBought column containing comma-separated codes
        
    Returns:
        DataFrame with additional boolean columns for each brand
    """
    # Brand code mapping
    brand_codes = {
        '1': 'Brand_Breyers',
        '2': 'Brand_BenJerrys',
        '4': 'Brand_HaloTop',
        '5': 'Brand_Enlightened',
        '6': 'Brand_Nicks',
        '7': 'Brand_StorePrivate',
        '8': 'Brand_LocalRegional'
    }
    
    # Initialize brand columns with False
    for brand_col in brand_codes.values():
        df[brand_col] = False
    
    # Parse each row
    for idx, row in df.iterrows():
        brands_str = str(row.get('Q6_BrandsBought', ''))
        if brands_str and brands_str != 'nan':
            # Split by comma and strip whitespace
            brand_list = [b.strip() for b in brands_str.split(',')]
            for code in brand_list:
                if code in brand_codes:
                    df.at[idx, brand_codes[code]] = True
    
    return df


def get_question_text(question_text: Dict[str, str], column: str) -> str:
    """
    Get the full question text for a column, with cleanup.
    
    Args:
        question_text: Dictionary mapping column names to question text
        column: Column name to look up
        
    Returns:
        Cleaned question text or column name if not found
    """
    text = question_text.get(column, column)
    
    # Remove the [Field-ConceptLabel] placeholder if present
    text = text.replace('[Field-ConceptLabel]', '').strip()
    text = text.replace('Thinking about the Breyers Better For You ice cream :', '').strip()
    
    # Clean up any leading colons or spaces
    text = text.lstrip(': ')
    
    return text if text else column


def filter_data(
    df: pd.DataFrame,
    concept_labels: list = None,
    diet_focus: list = None,
    age_groups: list = None
) -> pd.DataFrame:
    """
    Filter the DataFrame based on sidebar selections.
    
    Args:
        df: Full DataFrame
        concept_labels: List of ConceptLabel values to include (or None for all)
        diet_focus: List of Q21_DietFocus codes to include (or None for all)
        age_groups: List of Q23_Age codes to include (or None for all)
        
    Returns:
        Filtered DataFrame
    """
    filtered_df = df.copy()
    
    if concept_labels and len(concept_labels) > 0:
        filtered_df = filtered_df[filtered_df['ConceptLabel'].isin(concept_labels)]
    
    if diet_focus and len(diet_focus) > 0:
        filtered_df = filtered_df[filtered_df['Q21_DietFocus'].isin(diet_focus)]
    
    if age_groups and len(age_groups) > 0:
        filtered_df = filtered_df[filtered_df['Q23_Age'].isin(age_groups)]
    
    return filtered_df
