import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle

# Load Excel file
file_path = "Insurance.xlsx"
policy_df = pd.read_excel(file_path, sheet_name="Policy")
age_df = pd.read_excel(file_path, sheet_name="Age&Relationship")
suminsured_df = pd.read_excel(file_path, sheet_name="SumInsured")
additional_prop_df = pd.read_excel(file_path, sheet_name="AdditionalProperties")
additional_subtype_df = pd.read_excel(file_path, sheet_name="AdditionalPropertySubtype")
benefits_df = pd.read_excel(file_path, sheet_name="AdditionalBenefit")
zone_df = pd.read_excel(file_path, sheet_name="Zone")

# Merge SumInsured
suminsured_grouped = suminsured_df.groupby("PolicyTypeGUID")['SumInsured'].apply(list).reset_index()
features = pd.merge(policy_df, suminsured_grouped, on="PolicyTypeGUID", how="left")

# Merge Age & Relationship constraints
age_grouped = age_df.groupby("PolicyTypeGUID").apply(lambda x: pd.Series({
    'Relationships': list(x['Relationship'].unique()),
    'AgeMinimum': x['AgeMinimum'].min(),
    'AgeMaximum': x['AgeMaximum'].max()
})).reset_index()
features = pd.merge(features, age_grouped, on="PolicyTypeGUID", how="left")

# Extract Additional Properties: Gender & Personal Habits
def extract_property(policy_id, prop_type):
    prop = additional_prop_df[(additional_prop_df['PolicyTypeGUID'] == policy_id) & 
                              (additional_prop_df['Property'] == prop_type)]
    if prop.empty:
        return None
    prop_id = prop['AdditionalPropertyID'].values[0]
    subtypes = additional_subtype_df[(additional_subtype_df['PolicyTypeGUID'] == policy_id) & 
                                     (additional_subtype_df['AdditionalPropertyID'] == prop_id)]
    return list(subtypes['Description'].unique())

features['AllowedGenders'] = features['PolicyTypeGUID'].apply(lambda x: extract_property(x, 'Gender'))
features['AllowedHabits'] = features['PolicyTypeGUID'].apply(lambda x: extract_property(x, 'Personal Habits'))

# Merge Benefits
benefit_grouped = benefits_df.groupby("PolicyTypeGUID").apply(
    lambda x: "\n".join([f"{row['BenefitType']}: {row['Description']}" for _, row in x.iterrows()])
).reset_index(name='Benefits')
features = pd.merge(features, benefit_grouped, on="PolicyTypeGUID", how="left")

# Merge Zone info
zone_grouped = zone_df.groupby("PolicyTypeGUID")['Description'].apply(lambda x: ", ".join(x.dropna())).reset_index(name='ZoneName')
features = pd.merge(features, zone_grouped, on="PolicyTypeGUID", how="left")

# Fill missing Policy Category & Plan Type
features['PolicyCategory'] = features['PolicyCategory'].fillna('All')
features['PolicyPlanType'] = features['PolicyPlanType'].fillna('All')

# Encode categorical columns
cat_cols = ['PolicyCategory', 'PolicyPlanType']
encoders = {col: LabelEncoder().fit(features[col].astype(str)) for col in cat_cols}
for col in cat_cols:
    features[col] = encoders[col].transform(features[col].astype(str))

# Prepare model input features
X = features[['MinNoOfPerson', 'MaxNoOfPerson', 'PolicyCategory', 'PolicyPlanType']].copy()

# Normalize Min/Max number of persons
scaler = StandardScaler()
X[['MinNoOfPerson', 'MaxNoOfPerson']] = scaler.fit_transform(X[['MinNoOfPerson', 'MaxNoOfPerson']])

# Train Nearest Neighbors model
model = NearestNeighbors(n_neighbors=5, metric='euclidean')
model.fit(X)

# Save the model and all metadata
with open("final_model.pkl", "wb") as f:
    pickle.dump({
        'model': model,
        'data': features,
        'encoders': encoders,
        'scaler': scaler,
        'X': X,
        'age_df': age_df,
        'additional_prop_df': additional_prop_df,
        'additional_subtype_df': additional_subtype_df
    }, f)

print("\u2705 Model training complete. Saved as final_model.pkl")

# Recommendation function using additional filters
def get_recommendations(user_input, model_data, top_n=5, min_n=3):
    model = model_data['model']
    features = model_data['data']
    encoders = model_data['encoders']
    scaler = model_data['scaler']
    X = model_data['X']

    # Prepare input vector
    input_vector = np.array([[user_input['MinNoOfPerson'],
                              user_input['MaxNoOfPerson'],
                              encoders['PolicyCategory'].transform([user_input['PolicyCategory']])[0],
                              encoders['PolicyPlanType'].transform([user_input['PolicyPlanType']])[0]]])
    input_vector[:, :2] = scaler.transform(input_vector[:, :2])

    # Get nearest neighbors
    distances, indices = model.kneighbors(input_vector, n_neighbors=10)
    candidates = features.iloc[indices[0]].copy()

    # Apply soft filters
    def policy_matches(policy):
        if user_input['Relationship'] not in policy['Relationships']:
            return False
        if not (policy['AgeMinimum'] <= user_input['Age'] <= policy['AgeMaximum']):
            return False
        if policy['AllowedGenders'] and user_input['Gender'] not in policy['AllowedGenders']:
            return False
        if policy['AllowedHabits'] and user_input['Habit'] not in policy['AllowedHabits']:
            return False
        if policy['ZoneName'] and user_input['Zone'] not in policy['ZoneName']:
            return False
        if policy['SumInsured'] and user_input['SumInsured'] not in policy['SumInsured']:
            return False
        return True

    filtered = candidates[candidates.apply(policy_matches, axis=1)]

    if len(filtered) < min_n:
        filtered = candidates.head(top_n)

    return filtered.head(top_n)
