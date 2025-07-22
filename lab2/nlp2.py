import streamlit as st
from Bio import pairwise2
from nltk.metrics import edit_distance
import nltk

# Configure page
st.set_page_config(
    page_title="NLP Lab 2: Edit Distance & Sequence Alignment",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Card styling for sections */
    .info-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    /* Custom button styling */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f0f2f6;
        border-radius: 25px;
        color: #262730;
        font-weight: bold;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Input field styling */
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        padding: 10px;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
    }
    
    /* Code block styling */
    .stCode {
        background: #1e1e1e;
        border-radius: 10px;
        border: none;
    }
    
    /* Metric styling */
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        text-align: center;
        border-left: 4px solid #667eea;
    }
    
    /* Success/info box styling */
    .stSuccess, .stInfo {
        border-radius: 10px;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Main header
st.markdown("""
<div class="main-header">
    <h1>🧬 NLP Lab 2: Edit Distance & Sequence Alignment</h1>
    <p style="margin: 0; font-size: 1.2em; opacity: 0.9;">R Abhijit Srivathsan - 2448044</p>
</div>
""", unsafe_allow_html=True)

# Sidebar with information
with st.sidebar:
    st.markdown("### 📚 About This Tool")
    st.markdown("""
    This application provides two key NLP/Bioinformatics functions:
    
    **🔍 Edit Distance**
    - Calculates Levenshtein distance
    - Useful for spell checking, DNA analysis
    - Measures string similarity
    
    **🧬 Sequence Alignment**
    - Needleman-Wunsch algorithm
    - Global sequence alignment
    - DNA/Protein sequence comparison
    """)
    
    st.markdown("---")
    st.markdown("### 💡 Quick Tips")
    st.markdown("""
    - Use uppercase for DNA sequences (A, T, G, C)
    - For proteins, use standard amino acid codes
    - Adjust scoring parameters for different alignment types
    - Higher match scores favor exact matches
    """)

# Create tabs with icons
tab1, tab2 = st.tabs(["🔍 Edit Distance", "🧬 Sequence Alignment"])

# Tab 1: Edit Distance
with tab1:
    # st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.markdown("### Compute Edit Distance")
    st.markdown("Calculate the **Levenshtein distance** between two strings - the minimum number of single-character edits needed to transform one string into another.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Create two columns for inputs
    col1, col2 = st.columns(2)
    
    with col1:
        string1 = st.text_input("🔤 Enter first string:", 
                               placeholder="e.g., kitten", 
                               key="edit_str1",
                               help="Enter any text string")
    
    with col2:
        string2 = st.text_input("🔤 Enter second string:", 
                               placeholder="e.g., sitting", 
                               key="edit_str2",
                               help="Enter any text string")
    
    # Center the button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        calculate_btn = st.button("🚀 Calculate Edit Distance", use_container_width=True)
    
    # Calculate and display results
    if calculate_btn:
        if string1.strip() and string2.strip():
            distance = edit_distance(string1.strip(), string2.strip())
            
            # Create metrics display
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="metric-container">
                    <h3 style="color: #667eea; margin: 0;">Edit Distance</h3>
                    <h1 style="color: #333; margin: 0;">{distance}</h1>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                similarity = max(0, (max(len(string1), len(string2)) - distance) / max(len(string1), len(string2)) * 100)
                st.markdown(f"""
                <div class="metric-container">
                    <h3 style="color: #667eea; margin: 0;">Similarity</h3>
                    <h1 style="color: #333; margin: 0;">{similarity:.1f}%</h1>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                max_length = max(len(string1), len(string2))
                st.markdown(f"""
                <div class="metric-container">
                    <h3 style="color: #667eea; margin: 0;">Max Length</h3>
                    <h1 style="color: #333; margin: 0;">{max_length}</h1>
                </div>
                """, unsafe_allow_html=True)
            
            st.success(f"✅ Edit distance between **'{string1}'** and **'{string2}'** is **{distance}** edits")
            
            # Show interpretation
            if distance == 0:
                st.info("🎉 Perfect match! The strings are identical.")
            elif distance <= 2:
                st.info("🟢 Very similar strings - minimal differences.")
            elif distance <= 5:
                st.info("🟡 Moderately similar strings.")
            else:
                st.info("🔴 Quite different strings - many edits needed.")
                
        else:
            st.warning("⚠️ Please enter both strings to calculate edit distance")

# Tab 2: Sequence Alignment
with tab2:
    # st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.markdown("### Perform Sequence Alignment")
    st.markdown("Perform **Needleman-Wunsch global alignment** on DNA or protein sequences to find optimal alignments.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Sequence inputs
    col1, col2 = st.columns(2)
    
    with col1:
        seq1 = st.text_input("🧬 Enter sequence A:", 
                            placeholder="e.g., ATCGATCG", 
                            key="seq1",
                            help="Enter DNA (ATGC) or protein sequence").strip().upper()
    
    with col2:
        seq2 = st.text_input("🧬 Enter sequence B:", 
                            placeholder="e.g., ATCGATCG", 
                            key="seq2",
                            help="Enter DNA (ATGC) or protein sequence").strip().upper()
    
    # Parameters section with expander
    with st.expander("🔧 Alignment Parameters", expanded=False):
        st.markdown("Adjust scoring parameters for different alignment types:")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            match_score = st.number_input("✅ Match Score", 
                                        value=2, 
                                        min_value=-10, 
                                        max_value=10,
                                        key="match",
                                        help="Score for matching characters")
        with col2:
            mismatch_score = st.number_input("❌ Mismatch Score", 
                                           value=-1, 
                                           min_value=-10, 
                                           max_value=10,
                                           key="mismatch",
                                           help="Score for mismatched characters")
        with col3:
            gap_open = st.number_input("🔓 Gap Open", 
                                     value=-2, 
                                     min_value=-10, 
                                     max_value=10,
                                     key="gap_open",
                                     help="Penalty for opening a gap")
        with col4:
            gap_extend = st.number_input("📏 Gap Extend", 
                                       value=-1, 
                                       min_value=-10, 
                                       max_value=10,
                                       key="gap_extend",
                                       help="Penalty for extending a gap")
    
    # Center the alignment button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        align_btn = st.button("🚀 Perform Alignment", use_container_width=True)
    
    # Perform alignment and display results
    if align_btn:
        if seq1 and seq2:
            with st.spinner("🔄 Computing optimal alignment..."):
                # Perform alignment
                alignments = pairwise2.align.globalms(seq1, seq2, match_score, mismatch_score, gap_open, gap_extend) # type: ignore
                
                if alignments:
                    # Get the best alignment
                    aligned_seq1, aligned_seq2, score, start, end = alignments[0]
                    
                    # Generate match line with different symbols
                    match_line = ""
                    matches = 0
                    total_positions = 0
                    
                    for a, b in zip(aligned_seq1, aligned_seq2):
                        if a == b and a != '-':
                            match_line += "|"
                            matches += 1
                        elif a == "-" or b == "-":
                            match_line += " "
                        else:
                            match_line += "•"
                        total_positions += 1
                    
                    # Calculate statistics
                    identity = (matches / len(seq1.replace('-', '')) * 100) if seq1 else 0
                    gaps = aligned_seq1.count('-') + aligned_seq2.count('-')
                    
                    # Display alignment results
                    st.markdown("### 🎯 Alignment Results")
                    
                    # Metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.markdown(f"""
                        <div class="metric-container">
                            <h4 style="color: #667eea; margin: 0;">Alignment Score</h4>
                            <h2 style="color: #333; margin: 0;">{score:.1f}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="metric-container">
                            <h4 style="color: #667eea; margin: 0;">Identity</h4>
                            <h2 style="color: #333; margin: 0;">{identity:.1f}%</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                        <div class="metric-container">
                            <h4 style="color: #667eea; margin: 0;">Matches</h4>
                            <h2 style="color: #333; margin: 0;">{matches}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col4:
                        st.markdown(f"""
                        <div class="metric-container">
                            <h4 style="color: #667eea; margin: 0;">Gaps</h4>
                            <h2 style="color: #333; margin: 0;">{gaps}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Alignment visualization
                    st.markdown("### 📊 Sequence Alignment Visualization")
                    
                    # Split long alignments into chunks for better display
                    chunk_size = 80
                    alignment_chunks = []
                    
                    for i in range(0, len(aligned_seq1), chunk_size):
                        chunk1 = aligned_seq1[i:i+chunk_size]
                        chunk2 = aligned_seq2[i:i+chunk_size]
                        chunk_match = match_line[i:i+chunk_size]
                        alignment_chunks.append((chunk1, chunk_match, chunk2))
                    
                    for i, (chunk1, chunk_match, chunk2) in enumerate(alignment_chunks):
                        pos_start = i * chunk_size + 1
                        pos_end = min((i + 1) * chunk_size, len(aligned_seq1))
                        
                        st.markdown(f"**Position {pos_start}-{pos_end}:**")
                        
                        # Create alignment display with line numbers
                        alignment_display = f"Seq A: {chunk1}\n       {chunk_match}\nSeq B: {chunk2}"
                        st.code(alignment_display, language=None)
                    
                    # Legend
                    st.markdown("""
                    **Legend:**
                    - `|` = Match
                    - `•` = Mismatch  
                    - ` ` = Gap
                    """)
                    
                    # Quality assessment
                    if identity >= 80:
                        st.success("🎉 Excellent alignment! High sequence similarity.")
                    elif identity >= 60:
                        st.info("✅ Good alignment with moderate similarity.")
                    elif identity >= 40:
                        st.warning("⚠️ Fair alignment - sequences show some similarity.")
                    else:
                        st.error("❌ Poor alignment - sequences are quite different.")
                        
                else:
                    st.error("❌ No alignment found. Please check your sequences and parameters.")
        else:
            st.warning("⚠️ Please enter both sequences to perform alignment")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🔬 <strong>NLP Lab 2</strong> | Built with Streamlit & Biopython
</div>
""", unsafe_allow_html=True)