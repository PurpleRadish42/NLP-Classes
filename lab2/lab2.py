import streamlit as st
from Bio import pairwise2
from nltk.metrics import edit_distance
import nltk

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Set page title
st.title("NLP Lab 2: Edit Distance and Applications")
st.write("R Abhijit Srivathsan - 2448044")

# Create tabs for different functions
tab1, tab2 = st.tabs(["Edit Distance", "Sequence Alignment"])

# Tab 1: Edit Distance
with tab1:
    st.header("Compute Edit Distance")
    st.write("Calculate the Levenshtein distance between two strings")
    
    # Input fields
    string1 = st.text_input("Enter first string:", key="edit_str1")
    string2 = st.text_input("Enter second string:", key="edit_str2")
    
    # Calculate button
    if st.button("Calculate Edit Distance"):
        if string1 and string2:
            distance = edit_distance(string1, string2)
            st.success(f"Edit distance between '{string1}' and '{string2}' is: **{distance}**")
        else:
            st.warning("Please enter both strings")

# Tab 2: Sequence Alignment
with tab2:
    st.header("Perform Sequence Alignment")
    st.write("Perform Needleman-Wunsch global alignment on DNA/protein sequences")
    
    # Input fields
    seq1 = st.text_input("Enter sequence A:", key="seq1").strip().upper()
    seq2 = st.text_input("Enter sequence B:", key="seq2").strip().upper()
    
    # Parameters (optional - with defaults)
    st.subheader("Alignment Parameters (optional)")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        match_score = st.number_input("Match", value=1, key="match")
    with col2:
        mismatch_score = st.number_input("Mismatch", value=-1, key="mismatch")
    with col3:
        gap_open = st.number_input("Gap Open", value=-2, key="gap_open")
    with col4:
        gap_extend = st.number_input("Gap Extend", value=-2, key="gap_extend")
    
    # Perform alignment button
    if st.button("Perform Alignment"):
        if seq1 and seq2:
            # Perform alignment
            alignments = pairwise2.align.globalms(seq1, seq2, match_score, mismatch_score, gap_open, gap_extend) # type: ignore
            
            if alignments:
                # Get the first (best) alignment
                aligned_seq1, aligned_seq2, score, start, end = alignments[0]
                
                # Generate match line
                match_line = ""
                for a, b in zip(aligned_seq1, aligned_seq2):
                    if a == b:
                        match_line += "|"
                    elif a == "-" or b == "-":
                        match_line += " "
                    else:
                        match_line += "."
                
                # Display results
                st.subheader("Alignment Results:")
                st.text("Aligned Sequences:")
                
                # Use monospace font for alignment display
                st.code(aligned_seq1 + "\n" + match_line + "\n" + aligned_seq2)
                
                st.info(f"Alignment Score: **{score}**")
            else:
                st.error("No alignment found")
        else:
            st.warning("Please enter both sequences")

# # Add footer with instructions
# st.markdown("---")
# st.markdown("### How to use:")
# st.markdown("1. **Edit Distance**: Enter two strings to calculate their edit distance")
# st.markdown("2. **Sequence Alignment**: Enter two sequences (DNA/protein) to perform global alignment")
# st.markdown("3. You can adjust alignment parameters or use the defaults")