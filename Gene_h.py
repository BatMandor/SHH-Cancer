#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix, r2_score
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re


new_file_path = './mutated_dataset.xlsx'
new_data = pd.read_excel(new_file_path, header=0)
new_data_cleaned = new_data.dropna(axis=1, how='all')


# Function to apply mutations to the reference gene sequence
def apply_mutation_to_gene(ref_seq, mutation):
    mutation_match = re.match(r'([A-Z])(\d+)([A-Z])$', mutation)
    if mutation_match:
        initial_nuc, position, new_nuc = mutation_match.groups()
        position = int(position) - 1  # Convert so 0 index is the first position
        #print(mutation_match.groups())
        #It seems that the reference gene nucleotide is not matching with the initial nucleotide according to the excel file
        if position < len(ref_seq) and ref_seq[position] == initial_nuc:
            modified_seq = ref_seq[:position] + new_nuc + ref_seq[position+1:]
            return modified_seq
        else:
            # print('expected gene not present at position')
            return ref_seq
        
    else:
        # Return original sequence for complex mutations
        return ref_seq

#Parse the protein changes and apply them to the reference gene sequence
amino_acids = "MLLLARCLLLVLVSSLLVCSGLACGPGRGFGKRRHPKKLTPLAYKQFIPNVAEKTLGASGRYEGKISRNSERFKELTPNYNPDIIFKDEENTGADRLMTQRCKDKLNALAISVMNQWPGVKLRVTEGWDEDGHHSEESLHYEGRAVDITTSDRDRSKYGMLARLAVEAGFDWVYYESKAHIHCSVKAENSVAAKSGGCFPGSATVHLEQGGTKLVKDLSPGDRVLAADDQGRLLYSDFLTFLDRDDGAKKVFYVIETREPRERLLLTAAHLLFVAPHNDSATGEPEASSGSGPPSGGALGPRALFASRVRPGQRVYVVAERDGDRRLLPAAVHSVTLSEEAAGAYAPLTAQGTILINRVLASCYAVIEEHSWAHRAFAPFRLAHALLAALAPARTDRGGDSGGGDRGGGGGRVALTAPGAADAPGAGATAGIHWYSQLLYQIGTWLLDSEALHPLGMAVKSS"
protein_changes = new_data_cleaned.iloc[:, 1].fillna('')  # 2nd column for protein changes
modified_gene_sequences = [apply_mutation_to_gene(amino_acids, change) for change in protein_changes]

#Add the modified sequences as a new column in the DataFrame
new_data_cleaned['Modified Gene Sequence'] = modified_gene_sequences
gene_data = new_data_cleaned

# print(gene_data.shape)
# print(gene_data['Modified Gene Sequence'])

# Load the datasets
mutated_file_path = './mutated_dataset.xlsx'
nonmutated_file_path = './combined_study_clinical_data(2).xlsx'
mutated_data = pd.read_excel(mutated_file_path, header=0)
nonmutated_data = pd.read_excel(nonmutated_file_path, header=0)

# Add "Has Mutation" feature and combine the datasets
mutated_data['Has Mutation'] = 1
nonmutated_data['Has Mutation'] = 0
combined_data = pd.concat([mutated_data, nonmutated_data], ignore_index=True)

# Convert 'Overall Survival Status' from string to numeric
combined_data['Overall Survival Status'] = combined_data['Overall Survival Status'].map({'0:LIVING': 0, '1:DECEASED': 1})

# Apply the function to extract positions only for the mutated dataset
def extract_position(protein_change):
    if pd.isnull(protein_change):
        return np.nan
    match = re.match(r'[A-Za-z](\d+)[A-Za-z]', protein_change)
    if match:
        return int(match.group(1))
    return np.nan

combined_data['Protein Change Position'] = combined_data['Protein Change'].apply(extract_position)

# One-hot encoding for categorical columns
categorical_columns = ['Mutation Type', 'Sex', 'Ethnicity Category', 'Race Category']
combined_data = pd.get_dummies(combined_data, columns=categorical_columns)

# Drop rows where 'Overall Survival (Months)' is NaN before defining X and y
combined_data = combined_data.dropna(subset=['Overall Survival (Months)'])

#Dataset only containing mutation locations
mutation_loc_data = pd.get_dummies(combined_data['Protein Change Position'])

features_to_exclude = ['Sample ID', 'Protein Change', 'Protein Change Position'] 
 # Update or add to this list as necessary
# X = combined_data.drop(features_to_exclude + ['Overall Survival (Months)', 'Overall Survival Status'], axis=1)
X = mutation_loc_data
y = combined_data['Overall Survival (Months)']

# Split, train, and evaluate the model as before
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
# print(f'Mean Squared Error: {mse}')
# print(f'R-squared: {r2}')

# Feature Importances
feature_importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(y=feature_importances.head(10), x=feature_importances.head(10).index)
plt.title('Top 10 Feature Importances')
plt.ylabel('Relative Importance')
plt.xlabel('Features')
plt.show()

# Scatter plot for actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
# plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)  # Diagonal line representing perfect predictions
plt.xlabel('Actual Survival Months')
plt.ylabel('Predicted Survival Months')
plt.title('Actual vs. Predicted Survival Months')
plt.show()
