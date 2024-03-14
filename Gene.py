#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#%%

new_file_path = './mutated_dataset.xlsx'
new_data = pd.read_excel(new_file_path, header=0)
new_data_cleaned = new_data.dropna(axis=1, how='all')

#%%
# Function to apply mutations to the reference gene sequence
def apply_mutation_to_gene(ref_seq, mutation):
    mutation_match = re.match(r'([A-Z])(\d+)([A-Z])$', mutation)
    if mutation_match:
        initial_nuc, position, new_nuc = mutation_match.groups()
        position = int(position) - 1  # Convert so 0 index is the first position
        print(mutation_match.groups())
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

#%%
#Parse the protein changes and apply them to the reference gene sequence
amino_acids = "MLLLARCLLLVLVSSLLVCSGLACGPGRGFGKRRHPKKLTPLAYKQFIPNVAEKTLGASGRYEGKISRNSERFKELTPNYNPDIIFKDEENTGADRLMTQRCKDKLNALAISVMNQWPGVKLRVTEGWDEDGHHSEESLHYEGRAVDITTSDRDRSKYGMLARLAVEAGFDWVYYESKAHIHCSVKAENSVAAKSGGCFPGSATVHLEQGGTKLVKDLSPGDRVLAADDQGRLLYSDFLTFLDRDDGAKKVFYVIETREPRERLLLTAAHLLFVAPHNDSATGEPEASSGSGPPSGGALGPRALFASRVRPGQRVYVVAERDGDRRLLPAAVHSVTLSEEAAGAYAPLTAQGTILINRVLASCYAVIEEHSWAHRAFAPFRLAHALLAALAPARTDRGGDSGGGDRGGGGGRVALTAPGAADAPGAGATAGIHWYSQLLYQIGTWLLDSEALHPLGMAVKSS"
print(amino_acids)
protein_changes = new_data_cleaned.iloc[:, 1].fillna('')  # 2nd column for protein changes
modified_gene_sequences = [apply_mutation_to_gene(amino_acids, change) for change in protein_changes]

#Add the modified sequences as a new column in the DataFrame
new_data_cleaned['Modified Gene Sequence'] = modified_gene_sequences
gene_data = new_data_cleaned

print(gene_data.shape)
print(gene_data['Modified Gene Sequence'])


# %%
import pandas as pd
from sklearn.model_selection import train_test_split
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# Load the datasets
mutated_dataset = pd.read_excel('mutated_dataset.xlsx')
combined_study_data = pd.read_excel('combined_study_clinical_data(2).xlsx')

# Combine the datasets
combined_data = pd.concat([mutated_dataset, combined_study_data], ignore_index=True)
print(combined_data)
# Select the required features
selected_features = combined_data[['Sample ID', 'Protein Change', 'Mutation Type', 'Sex', 'Overall Survival (Months)', 'Overall Survival Status']]

# Process 'Protein Change' column to identify protein position and frameshift mutations
def process_protein_change(protein_change):
    match = re.match(r'([A-Za-z]+)(\d+)([A-Za-z]*)(fs\*\d+)?', protein_change)
    if match:
        position = match.group(2)
        frameshift = match.group(4) is not None
        return position, frameshift
    else:
        return None, False

# Extract Protein Position using regex within a lambda function
selected_features['Protein Position'] = selected_features['Protein Change'].apply(lambda x: re.findall(r'\d+', x)[0] if pd.notnull(x) else None)
# Filter the data based on 'Mutation Type' to include only specified mutation types
filtered_data = selected_features[selected_features['Mutation Type'].isin(['Frame_Shift_Del', 'Frame_Shift_Ins', 'Missense_Mutation'])]

# If you need to convert 'Mutation Type' into a numerical format for the model, you can use one-hot encoding or factorize
# Here's an example using pd.get_dummies() for one-hot encoding
mutation_type_dummies = pd.get_dummies(filtered_data['Mutation Type'], prefix='Mutation_Type')

# Concatenate the one-hot encoded DataFrame with your filtered_data DataFrame
filtered_data = pd.concat([filtered_data, mutation_type_dummies], axis=1)

# Now 'filtered_data' includes one-hot encoded columns for 'Mutation Type', along with other selected features

# Filter the data based on 'Mutation Type'
filtered_data = selected_features[selected_features['Mutation Type'].isin(['Frame_Shift_Del', 'Frame_Shift_Ins', 'Missense_Mutation'])]

# Convert 'Sex' into numerical values
filtered_data['Sex'] = filtered_data['Sex'].map({'Male': 0, 'Female': 1})

# Split the data into features (X) and the target variable (y)
X = filtered_data.drop(['Overall Survival (Months)', 'Overall Survival Status'], axis=1)
y = filtered_data['Overall Survival (Months)']
print(X, y)
# 80-20 train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Assuming 'X_train' and 'y_train' are already defined from the previous code

# Train a Random Forest model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Predict survival rates on the training data
y_pred_train = rf_model.predict(X_train)

# Plot actual vs. predicted survival rates for the training data
plt.scatter(y_train, y_pred_train)
plt.xlabel('Actual Survival Rates')
plt.ylabel('Predicted Survival Rates')
plt.title('Actual vs. Predicted Survival Rates on Training Data')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=4)
plt.show()

# Calculate and print statistical measurements
mae = mean_absolute_error(y_train, y_pred_train)
mse = mean_squared_error(y_train, y_pred_train)
r2 = r2_score(y_train, y_pred_train)

print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'R-squared Value: {r2}')


# %%
