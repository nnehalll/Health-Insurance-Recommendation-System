import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
import random

# Load model and data
with open("final_model.pkl", "rb") as f:
    data_dict = pickle.load(f)

model = data_dict['model']
data = data_dict['data']
encoders = data_dict['encoders']
scaler = data_dict['scaler']
X = data_dict['X']
age_df = data_dict['age_df']
additional_prop_df = data_dict['additional_prop_df']
additional_subtype_df = data_dict['additional_subtype_df']

# Load Policy Descriptions
policy_df = pd.read_excel("Insurance.xlsx", sheet_name="Policy")
policy_cat_desc = dict(zip(policy_df["PolicyCategory"], policy_df["PolicyCategoryDescription"]))
policy_plan_desc = dict(zip(policy_df["PolicyPlanType"], policy_df["PolicyPlanTypeDescription"]))

st.title("🩺 Health Insurance Recommendation System")

# Step 1: Number of Insured Members
num_members = st.number_input("Enter number of insured members", min_value=1, max_value=20, value=1)

# Step 2: Member Details
st.subheader("👨‍👩‍👧‍👦 Enter Member Details")
relationship_options = [
    'SELF', 'SPOUSE', 'SON', 'DAUGHTER', 'FATHER', 'MOTHER', 'BROTHER', 'SISTER',
    'FATHER IN LAW', 'MOTHER IN LAW', 'SISTER IN LAW', 'BROTHER IN LAW', 'GRAND SON',
    'GRAND DAUGHTER', 'GRAND DOUGHTER', 'GRAND FATHER', 'GRAND MOTHER', 'SON IN LAW',
    'DAUGHTER IN LAW'
]

members = []

for i in range(num_members):
    st.markdown(f"### 👤 Member {i + 1}")
    first_name = st.text_input("First Name", key=f"fname_{i}")
    middle_name = st.text_input("Middle Name (Optional)", key=f"mname_{i}")
    last_name = st.text_input("Last Name", key=f"lname_{i}")
    relationship = st.selectbox("Relationship", options=relationship_options, key=f"rel_{i}")

    # Default DOB logic
    default_age = 30 if relationship == "SELF" else 20
    default_dob = datetime.today() - timedelta(days=default_age * 365)

    dob = st.date_input("Birthdate (DD/MM/YYYY)", value=default_dob, format="DD/MM/YYYY", key=f"dob_{i}")
    birth_date_str = dob.strftime("%d/%m/%Y")
    age = (datetime.today().date() - dob).days // 365

    gender = st.selectbox("Gender", options=["Male", "Female", "Other"], key=f"gender_{i}")
    smoked = st.radio("Has smoked in the past 12 months?", ["Yes", "No"], key=f"smoke_{i}")
    tobacco = st.radio("Has consumed tobacco in the past 12 months?", ["Yes", "No"], key=f"tob_{i}")

    members.append({
        'first_name': first_name,
        'middle_name': middle_name,
        'last_name': last_name,
        'relationship': relationship,
        'age': age,
        'gender': gender,
        'smoked': smoked,
        'tobacco': tobacco,
        'dob_str': birth_date_str
    })

# Step 3: Policy Category
cat_options = sorted(list(set(encoders['PolicyCategory'].classes_) - {"All"}))
category_descriptions = [f"{cat} - {policy_cat_desc.get(cat, '')}" for cat in cat_options]
selected_cat = st.selectbox("Select Policy Category", ["All"] + category_descriptions)
policy_category = selected_cat.split(" - ")[0] if " - " in selected_cat else selected_cat

# Step 4: Policy Plan Type
plan_options = sorted(list(set(encoders['PolicyPlanType'].classes_) - {"All"}))
plan_descriptions = [f"{plan} - {policy_plan_desc.get(plan, '')}" for plan in plan_options]
selected_plan = st.selectbox("Select Policy Plan Type", ["All"] + plan_descriptions)
policy_plan = selected_plan.split(" - ")[0] if " - " in selected_plan else selected_plan

# Step 5: Sum Insured
suminsured_options = sorted([5000, 30000, 50000, 100000, 150000, 200000, 250000, 300000, 350000,
    400000, 450000, 500000, 550000, 600000, 650000, 700000, 750000, 800000, 850000, 875000, 
    900000, 1000000, 1200000, 1500000, 2000000, 2500000, 3000000, 4000000, 5000000, 10000000])
suminsured = st.selectbox("Select Sum Insured", options=["All"] + suminsured_options)

# Format Output - Clean line-by-line version
def format_policy_output(policy):
    lines = []

    # Policy Name
    lines.append(f"🛡️ Policy Name: {policy.get('PolicyName', 'Unnamed')}")

    # Company Name
    lines.append(f"🏢 Company Name: {policy.get('CompanyName', 'Unknown')}")

    # Sum Insured
    coverage = policy.get('SumInsured', [])
    if isinstance(coverage, list) and coverage:
        coverage_str = ', '.join([f"₹{int(c):,}" for c in sorted(set(coverage))])
        lines.append(f"💰 Sum Insured Options: {coverage_str}")

    # Key Features
    key_features = str(policy.get('PolicyPlanTypeDescription', '')).strip()
    if key_features.lower() != 'nan' and key_features:
        lines.append(f"✨ Key Features: {key_features}")

    # Zone
    zone = policy.get('ZoneName')
    if zone and str(zone).strip().lower() != "nan":
        lines.append(f"📍 Zone Coverage: {zone}")

    # Additional Benefits (line-by-line)
    benefits = policy.get('Benefits')
    if isinstance(benefits, str) and benefits.strip().lower() != "not available":
        benefit_lines = [line.strip() for line in benefits.replace('•', '').split('\n') if line.strip()]
        if benefit_lines:
            lines.append("🎁 Additional Benefits:")
            lines.extend(benefit_lines)  # Each benefit in a new line

    # Cashless Network Hospitals
    hospitals = policy.get('Cashless Network Hospitals')
    if hospitals:
        try:
            hospitals = int(hospitals)
            lines.append(f"🏥 Cashless Network Hospitals: {hospitals:,}")
        except:
            lines.append(f"🏥 Cashless Network Hospitals: {str(hospitals)}")

    # Website
    website = policy.get('Website Link')
    if website and website.strip().lower() != "not available":
        lines.append(f"🌐 Website: {website}")

    return "\n\n".join(lines)

# Recommend Policies
if st.button("🎯 Get Recommended Policies"):
    # Encode and transform user input
    try:
        input_df = pd.DataFrame([{
            'MinNoOfPerson': num_members,
            'MaxNoOfPerson': num_members,
            'PolicyCategory': encoders['PolicyCategory'].transform([policy_category])[0],
            'PolicyPlanType': encoders['PolicyPlanType'].transform([policy_plan])[0]
        }])
        input_scaled = input_df.copy()
        input_scaled[['MinNoOfPerson', 'MaxNoOfPerson']] = scaler.transform(input_scaled[['MinNoOfPerson', 'MaxNoOfPerson']])

        # Get recommendations
        distances, indices = model.kneighbors(input_scaled)
        recommended_policies = data.iloc[indices[0]].copy()
    except Exception as e:
        st.error(f"Error generating recommendations: {e}")
        recommended_policies = pd.DataFrame()

    if recommended_policies.empty:
        st.warning("⚠️ Sorry! No matching policies found.")
    else:
        count = recommended_policies.shape[0]
        st.subheader(f"🔍 Top Recommended Policies: {count} found")
        for _, row in recommended_policies.iterrows():
            st.markdown(format_policy_output(row))
            st.markdown("---")
