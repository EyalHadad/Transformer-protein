import pandas as pd
from sklearn.model_selection import train_test_split

def parse_fasta(fasta_file):
    sequences = []
    labels = []
    with open(fasta_file, 'r') as file:
        current_sequence = []
        current_label = None
        for line in file:
            if line.startswith('>'):
                if current_sequence and current_label:
                    sequences.append(''.join(current_sequence))
                    labels.append(current_label)
                header = line[1:].strip()
                if 'ribosomal' in header:
                    current_label = 'ribosomal'
                elif 'kinase' in header:
                    current_label = 'kinase'
                elif 'ligase' in header:
                    current_label = 'ligase'
                else:
                    current_label = None
                current_sequence = []
            else:
                if current_label:
                    current_sequence.append(line.strip())
        # Add the last sequence if there is one
        if current_sequence and current_label:
            sequences.append(''.join(current_sequence))
            labels.append(current_label)
    return sequences, labels

# File path to your FASTA file
fasta_file = 'uniprot_sprot.fasta'

# Parse the FASTA file and filter sequences
sequences, labels = parse_fasta(fasta_file)

# Create a DataFrame
df = pd.DataFrame({'sequence': sequences, 'label': labels})

# Split the dataset into train and test sets (80-20 ratio)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

# Write the train and test sets to CSV files
train_df.to_csv('train_dataset.csv', index=False)
test_df.to_csv('test_dataset.csv', index=False)

print("Data split and CSV files created successfully.")
