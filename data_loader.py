import pandas as pd
import numpy as np

def _map_categorical(df, col, mapping):
    if col in df.columns:
        pd.set_option('future.no_silent_downcasting', True)
        df[col] = df[col].map(mapping).fillna(df[col]).infer_objects(copy=False)
    return df

def _map_multiselect(val, mapping):
    if pd.isna(val):
        return val
    try:
        parts = str(val).split(',')
        labels = [mapping.get(p.strip(), p.strip()) for p in parts]
        return ', '.join(labels)
    except Exception:
        return val

def load_data(filepath='breyers-survey-data-cleaned.csv'):
    """
    Load and preprocess the survey data into a pandas DataFrame.
    """
    df = pd.read_csv(filepath, header=0, skiprows=[1])

    # Mapping logic based on breyers-survey-key.md

    # Q2_PurchaseRecent
    q2_map = {'1': 'Yes', '2': 'No'}
    df = _map_categorical(df, 'Q2_PurchaseRecent', q2_map)

    # Q3_DecisionRole
    q3_map = {'1': 'I mostly decide', '2': 'I share the decision'}
    df = _map_categorical(df, 'Q3_DecisionRole', q3_map)

    # Q4_PurchaseFreq
    q4_map = {1: 'Weekly', 2: '2-3 times per month', 3: 'Monthly', 4: 'Less often'}
    df = _map_categorical(df, 'Q4_PurchaseFreq', q4_map)

    # Q5_UsualChannel
    q5_map = {1: 'Grocery store', 4: 'Online delivery'}
    df = _map_categorical(df, 'Q5_UsualChannel', q5_map)

    # Q6_BrandsBought (multiselect)
    q6_map = {
        '1': 'Breyers',
        '2': "Ben & Jerry's",
        '4': 'Halo Top',
        '5': 'Enlightened',
        '6': "Nick's",
        '7': 'Store brand / private label',
        '8': 'Local or regional brand'
    }
    if 'Q6_BrandsBought' in df.columns:
        df['Q6_BrandsBought'] = df['Q6_BrandsBought'].apply(lambda x: _map_multiselect(x, q6_map))

    # Q8 Attribute Importances (1 to 5 scale)
    attr_map = {1: 'Not at all important', 2: 'Slightly important',
                3: 'Moderately important', 4: 'Very important', 5: 'Extremely important'}
    q8_cols = [f'Q8_AttrImportance_{i}' for i in range(1, 8)]
    for col in q8_cols:
        df = _map_categorical(df, col, attr_map)

    # Q9_Tradeoff
    q9_map = {1: 'Higher protein content', 2: 'Lower sugar content', 3: 'Neither - taste matters more'}
    df = _map_categorical(df, 'Q9_Tradeoff', q9_map)

    # Q10_ActiveSeeking
    q10_map = {1: 'I actively look for low/zero sugar ice cream',
               2: 'I actively look for high-protein ice cream',
               3: 'I actively look for both',
               4: 'Neither is a priority for me'}
    df = _map_categorical(df, 'Q10_ActiveSeeking', q10_map)

    # Concept Exposed - derived from ClaimCell
    # ClaimCell 1: High Protein
    # ClaimCell 2: Low Sugar
    # ClaimCell 3: Both
    claim_cell_map = {1: 'Higher protein', 2: 'Low or zero added sugar', 3: 'Both higher protein and low/zero sugar'}
    df = _map_categorical(df, 'ClaimCell', claim_cell_map)

    # Q11_Appeal
    appeal_map = {1: 'Not at all appealing', 2: 'Slightly appealing',
                  3: 'Moderately appealing', 4: 'Very appealing', 5: 'Extremely appealing'}
    df = _map_categorical(df, 'Q11_Appeal', appeal_map)

    # Q12_PurchaseIntent
    pi_map = {1: 'Definitely would not buy', 2: 'Probably would not buy',
              3: 'Might or might not buy', 4: 'Probably would buy', 5: 'Definitely would buy'}
    df = _map_categorical(df, 'Q12_PurchaseIntent', pi_map)

    # Q13_Replacement
    q13_map = {1: 'Replace your usual ice cream', 2: 'Be in addition to your usual ice cream',
               3: 'Replace a different dessert', 4: 'Not sure'}
    df = _map_categorical(df, 'Q13_Replacement', q13_map)

    # Q13A_WhatReplaced
    q13a_map = {1: 'Regular Breyers ice cream', 2: 'Another ice cream brand (not Breyers)',
                3: 'A better-for-you ice cream brand', 4: 'Another dessert (not ice cream)',
                5: 'I would skip dessert altogether', 6: 'Not sure'}
    df = _map_categorical(df, 'Q13A_WhatReplaced', q13a_map)

    # Q14_InterestComparison
    q14_map = {1: 'Much less interested', 2: 'Somewhat less interested',
               3: 'About the same', 4: 'Somewhat more interested', 5: 'Much more interested'}
    df = _map_categorical(df, 'Q14_InterestComparison', q14_map)

    # Q16_PurchaseLocation
    q16_map = {1: 'Grocery store', 2: 'Club/wholesale store',
               3: 'Online delivery', 4: 'Convenience store'}
    df = _map_categorical(df, 'Q16_PurchaseLocation', q16_map)

    # Q17a-e: Price Sensitivities
    price_sens_map = {1: 'Very unlikely', 2: 'Unlikely', 3: 'Neither likely nor unlikely',
                      4: 'Likely', 5: 'Very likely'}
    for col in ['Q17a_Price399', 'Q17b_Price499', 'Q17c_Price599', 'Q17d_Price699', 'Q17e_Price799']:
        df = _map_categorical(df, col, price_sens_map)

    # Q18_PriceTooExpensive (Convert to float, handle errors)
    if 'Q18_PriceTooExpensive' in df.columns:
        df['Q18_PriceTooExpensive'] = pd.to_numeric(df['Q18_PriceTooExpensive'], errors='coerce')

    # Q19_ClubStore4Pack
    q19_map = {1: 'Definitely would not', 2: 'Probably would not', 3: 'Might or might not',
               4: 'Probably would', 5: 'Definitely would'}
    df = _map_categorical(df, 'Q19_ClubStore4Pack', q19_map)

    # Q20_OnlineDelivery
    q20_map = {1: 'Very unlikely', 2: 'Unlikely', 3: 'Neither likely nor unlikely',
               4: 'Likely', 5: 'Very likely'}
    df = _map_categorical(df, 'Q20_OnlineDelivery', q20_map)

    # Q21_DietFocus
    q21_map = {1: 'I try to limit sugar in my diet', 2: 'I try to increase protein in my diet',
               3: 'I try to do both', 4: 'Neither of these is a focus for me'}
    df = _map_categorical(df, 'Q21_DietFocus', q21_map)

    # Q22_HouseholdType
    q22_map = {1: 'Live alone', 2: 'Live with spouse/partner (no children)',
               3: 'Live with spouse/partner and children under 18',
               4: 'Live with children under 18 (single parent)', 6: 'Other'}
    df = _map_categorical(df, 'Q22_HouseholdType', q22_map)

    # Q23_Age
    q23_map = {1: '18-24', 2: '25-34', 3: '35-44', 4: '45-54', 5: '55-64', 6: '65+'}
    df = _map_categorical(df, 'Q23_Age', q23_map)

    # Q24_Income
    q24_map = {1: 'Less than $25,000', 2: '$25,000-$49,999', 3: '$50,000-$74,999',
               4: '$75,000-$99,999', 5: '$100,000-$149,999', 6: '$150,000 or more', 7: 'Prefer not to say'}
    df = _map_categorical(df, 'Q24_Income', q24_map)

    # Extract clean raw numeric inputs for statistical tests
    # Load raw again to just copy over numeric raw
    df_raw = pd.read_csv(filepath, header=0, skiprows=[1])
    # Keep Q11, Q12, Q14, Q18 as numeric copies if available, with _Num suffix
    numeric_cols = ['Q11_Appeal', 'Q12_PurchaseIntent', 'Q14_InterestComparison', 'ClaimCell']
    for c in numeric_cols:
        if c in df_raw.columns:
            df[f'{c}_Num'] = pd.to_numeric(df_raw[c], errors='coerce')
    for c in ['Q17a_Price399', 'Q17b_Price499', 'Q17c_Price599', 'Q17d_Price699', 'Q17e_Price799']:
        if c in df_raw.columns:
            df[f'{c}_Num'] = pd.to_numeric(df_raw[c], errors='coerce')

    return df
