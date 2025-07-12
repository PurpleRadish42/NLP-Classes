from Bio import pairwise2
from nltk.metrics import edit_distance
import nltk

def compute_edit_distance(s1, s2):
    """Compute Levenshtein edit distance using NLTK."""
    return edit_distance(s1, s2)

def perform_alignment(seq1, seq2, match=1, mismatch=-1, gap_open=-2, gap_extend=-2):
    """Use Biopython to perform Needleman-Wunsch global alignment."""
    alignments = pairwise2.align.globalms(seq1, seq2, match, mismatch, gap_open, gap_extend)
    aligned_seq1, aligned_seq2, score, _, _ = alignments[0]

    # Generate a match line
    match_line = ""
    for a, b in zip(aligned_seq1, aligned_seq2):
        if a == b:
            match_line += "|"
        elif a == "-" or b == "-":
            match_line += " "
        else:
            match_line += "."

    # Display the alignment
    print("\nAligned Sequences:")
    print(aligned_seq1)
    print(match_line)
    print(aligned_seq2)
    print(f"\nAlignment Score: {score}")

def menu():
    while True:
        print("\nMenu:")
        print("1. Compute Edit Distance")
        print("2. Perform Sequence Alignment")
        print("3. Exit")
        choice = input("Enter your choice (1/2/3): ").strip()

        if choice == "1":
            s1 = input("Enter the first string: ").strip()
            s2 = input("Enter the second string: ").strip()
            dist = compute_edit_distance(s1, s2)
            print(f"\nEdit distance between '{s1}' and '{s2}' is: {dist}")
        elif choice == "2":
            seq1 = input("Enter sequence A: ").strip().upper()
            seq2 = input("Enter sequence B: ").strip().upper()
            perform_alignment(seq1, seq2)
        elif choice == "3":
            print("Exiting program. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    # Ensure NLTK data is available
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    menu()
