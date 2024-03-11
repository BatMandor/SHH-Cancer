#%%
import pandas as pd
import re
from Bio import SeqIO
from lifelines import CoxPHFitter

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

