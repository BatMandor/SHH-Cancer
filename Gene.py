#%%
import pandas as pd
import re
from Bio import SeqIO

#%%
#Reference gene
gene_file_path = './gene.fna'
for seq_record in SeqIO.parse(gene_file_path, "fasta"):
    reference_gene_sequence = str(seq_record.seq)
print(reference_gene_sequence)
#Mutation data
new_file_path = './variables_adult.xlsx'
new_data = pd.read_excel(new_file_path, header=0)
new_data_cleaned = new_data.dropna(axis=1, how='all')

#%%
# Function to apply mutations to the reference gene sequence
def apply_mutation_to_gene(ref_seq, mutation):
    # Regular expression to parse the mutation format (e.g., A123C)
    mutation_match = re.match(r'([A-Z])(\d+)([A-Z])$', mutation)
    if mutation_match:
        initial_nuc, position, new_nuc = mutation_match.groups()
        position = int(position) - 1  # Convert so 0 index is the first position
        # print(mutation_match.groups())
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

# def dna_to_amino_acids(dna_sequence):
#     # Define the genetic code
#     genetic_code = {
#         'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
#         'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
#         'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
#         'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',                 
#         'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
#         'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
#         'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
#         'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
#         'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
#         'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
#         'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
#         'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
#         'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
#         'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
#         'TAC':'Y', 'TAT':'Y', 'TAA':'_', 'TAG':'_',
#         'TGC':'C', 'TGT':'C', 'TGA':'_', 'TGG':'W',
#     }

#     # Convert the DNA sequence to uppercase
#     dna_sequence = dna_sequence.upper()

#     # Group the sequence into codons
#     codons = [dna_sequence[i:i+3] for i in range(0, len(dna_sequence), 3) if len(dna_sequence[i:i+3]) == 3]

#     # Convert codons to amino acids
#     amino_acids = ''.join([genetic_code[codon] for codon in codons if codon in genetic_code])

#     return amino_acids



#%%
#Parse the protein changes and apply them to the reference gene sequence
# amino_acids = dna_to_amino_acids("")
amino_acids = 
print(amino_acids)
protein_changes = new_data_cleaned.iloc[:, 3].fillna('')  # 4th column for protein changes
modified_gene_sequences = [apply_mutation_to_gene(amino_acids, change) for change in protein_changes]

#Add the modified sequences as a new column in the DataFrame
new_data_cleaned['Modified Gene Sequence'] = modified_gene_sequences
gene_data = new_data_cleaned

print(gene_data.shape)
# print(gene_data['Modified Gene Sequence'])
# for mutated in gene_data['Modified Gene Sequence']:
#     print(mutated == reference_gene_sequence)

# %%
