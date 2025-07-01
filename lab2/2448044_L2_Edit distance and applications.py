import nltk
from nltk.metrics import edit_distance

def compute_edit_distance(s1, s2):
    """Compute the Levenshtein edit distance between two strings using NLTK."""
    return edit_distance(s1, s2)

def needleman_wunsch(seq1, seq2, match_score=1, mismatch_score=-1, gap_penalty=-2):
    """Perform global sequence alignment (Needleman–Wunsch) on two sequences."""
    len1, len2 = len(seq1), len(seq2)
    # Initialize scoring matrix
    score = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    for i in range(1, len1 + 1):
        score[i][0] = score[i-1][0] + gap_penalty
    for j in range(1, len2 + 1):
        score[0][j] = score[0][j-1] + gap_penalty

    # Fill the scoring matrix
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            match = score[i-1][j-1] + (match_score if seq1[i-1] == seq2[j-1] else mismatch_score)
            delete = score[i-1][j] + gap_penalty
            insert = score[i][j-1] + gap_penalty
            score[i][j] = max(match, delete, insert)

    # Traceback to build alignment
    align1, align2 = "", ""
    i, j = len1, len2
    while i > 0 and j > 0:
        current = score[i][j]
        if current == score[i-1][j-1] + (match_score if seq1[i-1] == seq2[j-1] else mismatch_score):
            align1 = seq1[i-1] + align1
            align2 = seq2[j-1] + align2
            i -= 1
            j -= 1
        elif current == score[i-1][j] + gap_penalty:
            align1 = seq1[i-1] + align1
            align2 = "-" + align2
            i -= 1
        else:
            align1 = "-" + align1
            align2 = seq2[j-1] + align2
            j -= 1

    # Finish tracing up to the matrix origin
    while i > 0:
        align1 = seq1[i-1] + align1
        align2 = "-" + align2
        i -= 1
    while j > 0:
        align1 = "-" + align1
        align2 = seq2[j-1] + align2
        j -= 1

    return align1, align2

def main():
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
            print(f"Edit distance between '{s1}' and '{s2}' is {dist}.")
        elif choice == "2":
            seq1 = input("Enter sequence A: ").strip().upper()
            seq2 = input("Enter sequence B: ").strip().upper()
            align1, align2 = needleman_wunsch(seq1, seq2)
            print("\nAligned Sequences:")
            print(align1)
            print(align2)
        elif choice == "3":
            print("Exiting program. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    main()
