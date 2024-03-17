#%%
import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load and prepare the dataset
new_file_path = './mutated_dataset.xlsx'
new_data = pd.read_excel(new_file_path, header=0)
new_data_cleaned = new_data.dropna(axis=1, how='all')

# Function to apply mutations to the reference gene sequence
def apply_mutation_to_gene(ref_seq, mutation):
    mutation_match = re.match(r'([A-Z])(\d+)([A-Z])$', mutation)
    if mutation_match:
        initial_nuc, position, new_nuc = mutation_match.groups()
        position = int(position) - 1
        if position < len(ref_seq) and ref_seq[position] == initial_nuc:
            return ref_seq[:position] + new_nuc + ref_seq[position+1:]
    return ref_seq

# Parse the protein changes and apply them to the reference gene sequence
amino_acids = "MLLLARCLLLVLVSSLLVCSGLACGPGRGFGKRRHPKKLTPLAYKQFIPNVAEKTLGASGRYEGKISRNSERFKELTPNYNPDIIFKDEENTGADRLMTQRCKDKLNALAISVMNQWPGVKLRVTEGWDEDGHHSEESLHYEGRAVDITTSDRDRSKYGMLARLAVEAGFDWVYYESKAHIHCSVKAENSVAAKSGGCFPGSATVHLEQGGTKLVKDLSPGDRVLAADDQGRLLYSDFLTFLDRDDGAKKVFYVIETREPRERLLLTAAHLLFVAPHNDSATGEPEASSGSGPPSGGALGPRALFASRVRPGQRVYVVAERDGDRRLLPAAVHSVTLSEEAAGAYAPLTAQGTILINRVLASCYAVIEEHSWAHRAFAPFRLAHALLAALAPARTDRGGDSGGGDRGGGGGRVALTAPGAADAPGAGATAGIHWYSQLLYQIGTWLLDSEALHPLGMAVKSS"
protein_changes = new_data_cleaned.iloc[:, 1].fillna('')
new_data_cleaned['Modified Gene Sequence'] = protein_changes.apply(lambda x: apply_mutation_to_gene(amino_acids, x))
gene_data = new_data_cleaned

# Load the datasets
mutated_dataset = pd.read_excel('mutated_dataset.xlsx')
combined_study_data = pd.read_excel('combined_study_clinical_data(2).xlsx')

# Combine the datasets
combined_data = pd.concat([mutated_dataset, combined_study_data], ignore_index=True)
print(combined_data)

# Select the required features, excluding 'Overall Survival Status'
selected_features = combined_data[['Sample ID', 'Protein Change', 'Mutation Type', 'Sex', 'Overall Survival (Months)']]

# Process 'Protein Change' column to identify protein position
def process_protein_change(protein_change):
    match = re.match(r'([A-Za-z]+)(\d+)([A-Za-z]*)(fs\*\d+)?', protein_change)
    if match:
        position = match.group(2)
        return position
    else:
        return None

# Extract Protein Position using regex within a lambda function
selected_features['Protein Position'] = selected_features['Protein Change'].apply(lambda x: re.findall(r'\d+', x)[0] if pd.notnull(x) else None)

# Filter the data based on 'Mutation Type' to include only specified mutation types
filtered_data = selected_features[selected_features['Mutation Type'].isin(['Frame_Shift_Del', 'Frame_Shift_Ins', 'Missense_Mutation'])]

# If you need to convert 'Mutation Type' into a numerical format for the model, you can use one-hot encoding or factorize
# Here's an example using pd.get_dummies() for one-hot encoding
mutation_type_dummies = pd.get_dummies(filtered_data['Mutation Type'], prefix='Mutation_Type')

# Concatenate the one-hot encoded DataFrame with your filtered_data DataFrame
filtered_data = pd.concat([filtered_data, mutation_type_dummies], axis=1)

# Convert 'Sex' into numerical values
filtered_data['Sex'] = filtered_data['Sex'].map({'Male': 0, 'Female': 1})

# Drop rows with NaNs in 'Overall Survival (Months)' to ensure 'y' contains no NaNs
filtered_data = filtered_data.dropna(subset=['Overall Survival (Months)'])

#Prepare the feature matrix (X) and target variable (y)
X = filtered_data.drop(['Sample ID', 'Protein Change', 'Mutation Type', 'Overall Survival (Months)'], axis=1)
y = filtered_data['Overall Survival (Months)']

# 80-20 train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Predict survival rates on the training data
y_pred_train = rf_model.predict(X_train)


# Calculate the statistical measures
mae = mean_absolute_error(y_train, y_pred_train)
mse = mean_squared_error(y_train, y_pred_train)
r2 = r2_score(y_train, y_pred_train)
r2p = r2*100
# Plot actual vs. predicted survival rates for the training data
plt.figure(figsize=(12, 8), dpi=300)
scatter_plot = plt.scatter(y_train, y_pred_train, alpha=0.6, label='Regression Model Predictions') 
plt.xlabel('Actual Survival Rates (Months)', fontsize=14)
plt.ylabel('Predicted Survival Rates (Months)', fontsize=14)
plt.title('Actual vs. Predicted Survival Rates on Training Data', fontsize=16)
ideal_line, = plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=4, label='Ideal Prediction Line')

text_location = (0.02, 0.98)
fontsize = 12
horizontalalignment = 'left'
verticalalignment = 'top'

# Create a text string with the statistical values
stats_str = (f"Mean Absolute Error: {mae:.2f}\n"
             f"Mean Squared Error: {mse:.2f}\n"
             f"R-squared: {r2:.5f} = {r2p:.3f}%")

# Add the text to the plot with a semi-transparent background for readability
plt.text(text_location[0], text_location[1], stats_str, fontsize=fontsize,
         horizontalalignment=horizontalalignment, verticalalignment=verticalalignment,
         transform=plt.gca().transAxes,  # Use the axes coordinates, not data coordinates
         bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='gray', alpha=0.8))

plt.legend(loc='upper right', fontsize=12)
plt.tight_layout()
plt.grid()
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.show()


# Extract feature importances and prepare for the heatmap
importances = rf_model.feature_importances_
feature_names = X.columns

# Create a dictionary mapping feature names to their importances
feature_importance_dict = dict(zip(feature_names, importances))
# Sort the features by their importance
sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
updated_feature_names = [
    (re.sub(r'^Mutation_Type_', '', feature).replace('_', ' ').replace('Protein Position', 'Mutation Position'), importance) 
    for feature, importance in sorted_features
]

# Create a DataFrame for the sorted and updated feature importances
importance_df = pd.DataFrame(updated_feature_names, columns=['Feature', 'Relative Importance']).set_index('Feature')

# Plot the heatmap with updated labels and larger annotations
plt.figure(figsize=(12, 8), dpi=300)  # Increased figure size for a poster
ax = sns.heatmap(importance_df.T, cmap='viridis', annot=True, annot_kws={"size": 14}) 
plt.title('Feature Importances', fontsize=16)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("")
plt.show()



#%%

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# Define a pipeline that first scales the features and then applies the random forest
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestRegressor(random_state=42))
])

# Parameters of the model
param_grid = {
    'rf__max_depth': [10, 20, None],
    'rf__max_features': ['sqrt', 'log2'],
    'rf__min_samples_split': [8, 10, 12],
    'rf__min_samples_leaf': [3, 4, 5],
    'rf__n_estimators': [100, 200, 300]
}

# Create a GridSearchCV object
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, 
                           cv=3, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')

# Fit the model
grid_search.fit(X_train, y_train)

# Print the best parameters and the best score
print(f'Best parameters found: {grid_search.best_params_}')
best_model = grid_search.best_estimator_

# Predict using the best model
y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)

# Evaluate the model using R-squared score
r2_train = best_model.score(X_train, y_train)
r2_test = best_model.score(X_test, y_test)
print(f'Improved R-squared Value on Training Data: {r2_train:.2f}')
print(f'Improved R-squared Value on Test Data: {r2_test:.2f}')

# Plot actual vs. predicted survival rates for the training data
plt.figure(figsize=(12, 8))  # Increased figure size for readability in a poster
plt.scatter(y_train, y_pred_train, label='Training Data')
plt.scatter(y_test, y_pred_test, label='Test Data', color='r')
plt.xlabel('Actual Survival Rates', fontsize=14)
plt.ylabel('Predicted Survival Rates', fontsize=14)
plt.title('Actual vs. Predicted Survival Rates', fontsize=16)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.legend(fontsize=12)

# Calculate statistical measurements
mae = mean_absolute_error(y_test, y_pred_test)
mse = mean_squared_error(y_test, y_pred_test)

# Add statistical values to the graph
plt.text(0.02, 0.98, f'MAE: {mae:.2f}\nMSE: {mse:.2f}\nR²: {r2_test:.2f}', fontsize=12,
         verticalalignment='top', horizontalalignment='left', transform=plt.gca().transAxes,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

plt.tight_layout()
plt.show()

# %%
