"""
Label mappings for Breyers Survey Dashboard.
All value codes mapped to human-readable labels from breyers-survey-key.md
"""

# Concept/Claim Cell Labels (ClaimCell column)
CONCEPT_LABEL = {
    1: "with low or zero added sugar",
    2: "with higher protein",
    3: "with higher protein and low or zero added sugar"
}

# Q4: Purchase Frequency
PURCHASE_FREQ = {
    1: "Weekly",
    2: "2-3 times per month",
    3: "Monthly",
    4: "Less often"
}

# Q11: Appeal (5-point scale)
APPEAL = {
    1: "Not at all appealing",
    2: "Slightly appealing",
    3: "Moderately appealing",
    4: "Very appealing",
    5: "Extremely appealing"
}

# Q12: Purchase Intent - CORRECTED per user feedback
# Uses likelihood scale, NOT "definitely would buy" scale
PURCHASE_INTENT = {
    1: "Very unlikely",
    2: "Unlikely",
    3: "Neither likely nor unlikely",
    4: "Likely",
    5: "Very likely"
}

# Top 2 Box Purchase Intent (derived variable)
TOP2BOX = {
    0: "Bottom 3 Box",
    1: "Top 2 Box (Likely/Very Likely)"
}

# Q13: Replacement behavior
REPLACEMENT = {
    1: "Replace your usual ice cream",
    2: "Be in addition to your usual ice cream",
    3: "Replace a different dessert",
    4: "Not sure"
}

# Q13A: What would be replaced
WHAT_REPLACED = {
    1: "Regular Breyers ice cream",
    2: "Another ice cream brand (not Breyers)",
    3: "A better-for-you ice cream brand (e.g., Halo Top, Enlightened, Nick's)",
    4: "Another dessert (not ice cream)",
    5: "I would skip dessert altogether",
    6: "Not sure"
}

# Q14: Interest Comparison
INTEREST_COMPARISON = {
    1: "Much less interested",
    2: "Somewhat less interested",
    3: "About the same",
    4: "Somewhat more interested",
    5: "Much more interested"
}

# Q16: Purchase Location
PURCHASE_LOCATION = {
    1: "Grocery store",
    2: "Club/wholesale store (e.g., Costco, Sam's)",
    3: "Online delivery",
    4: "Convenience store"
}

# Q17a-e: Price Likelihood (same scale as Q12)
PRICE_LIKELIHOOD = {
    1: "Very unlikely",
    2: "Unlikely",
    3: "Neither likely nor unlikely",
    4: "Likely",
    5: "Very likely"
}

# Q19: Club Store 4-Pack
CLUB_STORE = {
    1: "Definitely would not",
    2: "Probably would not",
    3: "Might or might not",
    4: "Probably would",
    5: "Definitely would"
}

# Q20: Online Delivery Likelihood
ONLINE_DELIVERY = {
    1: "Very unlikely",
    2: "Unlikely",
    3: "Neither likely nor unlikely",
    4: "Likely",
    5: "Very likely"
}

# Q21: Diet Focus
DIET_FOCUS = {
    1: "Limit sugar",
    2: "Increase protein",
    3: "Both",
    4: "Neither"
}

# Q22: Household Type
HOUSEHOLD_TYPE = {
    1: "Live alone",
    2: "Live with spouse/partner (no children)",
    3: "Live with spouse/partner and children under 18",
    4: "Live with children under 18 (single parent)",
    6: "Other"
}

# Q23: Age
AGE = {
    1: "18-24",
    2: "25-34",
    3: "35-44",
    4: "45-54",
    5: "55-64",
    6: "65+"
}

# Q24: Income
INCOME = {
    1: "Less than $25,000",
    2: "$25,000-$49,999",
    3: "$50,000-$74,999",
    4: "$75,000-$99,999",
    5: "$100,000-$149,999",
    6: "$150,000 or more",
    7: "Prefer not to say"
}

# Q8: Attribute Importance Labels
ATTR_IMPORTANCE = {
    1: "Taste",
    2: "Price",
    3: "Brand reputation",
    4: "Low/zero sugar",
    5: "High protein",
    6: "Short/clean ingredient list",
    7: "Low calorie content"
}

# Importance Scale (for Q8)
IMPORTANCE_SCALE = {
    1: "Not at all important",
    2: "Slightly important",
    3: "Moderately important",
    4: "Very important",
    5: "Extremely important"
}

# Q6: Brands Bought (multi-select codes)
BRANDS_BOUGHT = {
    1: "Breyers",
    2: "Ben & Jerry's",
    4: "Halo Top",
    5: "Enlightened",
    6: "Nick's",
    7: "Store brand / private label",
    8: "Local or regional brand"
}

# Q9: Tradeoff
TRADEOFF = {
    1: "Higher protein content",
    2: "Lower sugar content",
    3: "Neither - taste matters more"
}

# Q10: Active Seeking
ACTIVE_SEEKING = {
    1: "I actively look for low/zero sugar ice cream",
    2: "I actively look for high-protein ice cream",
    3: "I actively look for both",
    4: "Neither is a priority for me"
}

# Scale Footnotes for Charts - CORRECTED Q12 footnote
SCALE_FOOTNOTES = {
    "Q11_Appeal": "Scale: 1 = Not at all appealing, 5 = Extremely appealing",
    "Q12_PurchaseIntent": "Scale: 1 = Very unlikely, 5 = Very likely",
    "Q8_AttrImportance": "Scale: 1 = Not at all important, 5 = Extremely important",
    "Q17_Price": "Scale: 1 = Very unlikely, 5 = Very likely",
    "Q14_InterestComparison": "Scale: 1 = Much less interested, 5 = Much more interested",
    "Top2Box_PI": "Top 2 Box = Likely (4) or Very Likely (5) on Purchase Intent"
}

# Price points for Q17 variables
PRICE_POINTS = {
    "Q17a_Price399": "$3.99",
    "Q17b_Price499": "$4.99",
    "Q17c_Price599": "$5.99",
    "Q17d_Price699": "$6.99",
    "Q17e_Price799": "$7.99"
}
