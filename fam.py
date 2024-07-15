from collections import Counter


def parse_fasta_headers(fasta_file):
    headers = []
    with open(fasta_file, 'r') as file:
        for line in file:
            if line.startswith('>'):
                headers.append(line[1:].strip())  # Strip the '>' and any leading/trailing whitespace

    return headers


def count_words_in_headers(headers):
    word_counter = Counter()
    for header in headers:
        words = header.split()
        word_counter.update(words)

    return word_counter


# File path to your FASTA file
fasta_file = 'uniprot_sprot.fasta'

# Parse the headers from the FASTA file
headers = parse_fasta_headers(fasta_file)

# Count the words in the headers
word_counts = count_words_in_headers(headers)

# Print the most common words and their counts
print(word_counts.most_common())
