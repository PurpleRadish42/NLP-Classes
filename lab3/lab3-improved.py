import streamlit as st
import nltk
from nltk.util import ngrams
from nltk import FreqDist, pos_tag, word_tokenize
from collections import defaultdict, Counter
import math
import spacy
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re

# Configure page with enhanced styling
st.set_page_config(
    page_title="Advanced NLP Lab 3: N-gram, POS, NER Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🔬"
)

# Custom CSS for better styling with dark mode support
st.markdown("""
<style>
    /* Main header - works in both light and dark mode */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white !important;
    }
    
    /* Metric cards - adaptive colors */
    .metric-card {
        background: var(--background-color, #f8f9fa);
        color: var(--text-color, #000000);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    
    /* Dark mode detection and adaptive styling */
    @media (prefers-color-scheme: dark) {
        .metric-card {
            background: #2d3748 !important;
            color: #ffffff !important;
        }
        .formula-box {
            background: #2d3748 !important;
            color: #ffffff !important;
            border: 1px solid #4a5568 !important;
        }
        .explanation-box {
            background: #2c5282 !important;
            color: #ffffff !important;
        }
        .warning-box {
            background: #744210 !important;
            color: #ffffff !important;
        }
    }
    
    /* Check for Streamlit dark theme */
    [data-testid="stAppViewContainer"][class*="dark"] .metric-card,
    .stApp[class*="dark"] .metric-card {
        background: #2d3748 !important;
        color: #ffffff !important;
    }
    
    [data-testid="stAppViewContainer"][class*="dark"] .formula-box,
    .stApp[class*="dark"] .formula-box {
        background: #2d3748 !important;
        color: #ffffff !important;
        border: 1px solid #4a5568 !important;
    }
    
    [data-testid="stAppViewContainer"][class*="dark"] .explanation-box,
    .stApp[class*="dark"] .explanation-box {
        background: #2c5282 !important;
        color: #ffffff !important;
    }
    
    [data-testid="stAppViewContainer"][class*="dark"] .warning-box,
    .stApp[class*="dark"] .warning-box {
        background: #744210 !important;
        color: #ffffff !important;
    }
    
    /* Default light mode styles */
    .metric-card {
        background: #f8f9fa;
        color: #000000;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    
    .formula-box {
        background: #f0f2f6;
        color: #000000;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e1e5e9;
        margin: 1rem 0;
        font-family: 'Courier New', monospace;
    }
    
    .explanation-box {
        background: #e8f4f8;
        color: #000000;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: #fff3cd;
        color: #000000;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    
    .stTab {
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Ensure text is always visible */
    .metric-card *, .formula-box *, .explanation-box *, .warning-box * {
        color: inherit !important;
    }
</style>

<script>
// JavaScript to detect dark mode and apply styles dynamically
(function() {
    function updateTheme() {
        const isDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
        const streamlitDark = document.querySelector('[data-testid="stAppViewContainer"]')?.classList.contains('dark') || 
                             document.querySelector('.stApp')?.classList.contains('dark');
        
        const root = document.documentElement;
        if (isDark || streamlitDark) {
            root.style.setProperty('--background-color', '#2d3748');
            root.style.setProperty('--text-color', '#ffffff');
        } else {
            root.style.setProperty('--background-color', '#f8f9fa');
            root.style.setProperty('--text-color', '#000000');
        }
    }
    
    // Update on load
    updateTheme();
    
    // Listen for theme changes
    if (window.matchMedia) {
        window.matchMedia('(prefers-color-scheme: dark)').addListener(updateTheme);
    }
    
    // Observe for Streamlit theme changes
    const observer = new MutationObserver(updateTheme);
    observer.observe(document.body, { attributes: true, subtree: true, attributeFilter: ['class'] });
})();
</script>
""", unsafe_allow_html=True)

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        return True
    except:
        return False

# Load spacy model with custom NER for Indian names
@st.cache_resource
def load_spacy_model():
    try:
        nlp = spacy.load("en_core_web_sm")
        # Add custom patterns for Indian names and places
        ruler = nlp.add_pipe("entity_ruler", before="ner")
        patterns = [
            {"label": "PERSON", "pattern": [{"TEXT": {"REGEX": r"^[A-Z][a-z]*jit$"}}]},  # Names ending in 'jit'
            {"label": "PERSON", "pattern": [{"TEXT": {"REGEX": r"^[A-Z][a-z]*deep$"}}]}, # Names ending in 'deep'
            {"label": "PERSON", "pattern": [{"TEXT": {"REGEX": r"^[A-Z][a-z]*raj$"}}]},  # Names ending in 'raj'
            {"label": "GPE", "pattern": [{"TEXT": {"REGEX": r"^Bangalore$"}}]},
            {"label": "GPE", "pattern": [{"TEXT": {"REGEX": r"^Chennai$"}}]},
            {"label": "GPE", "pattern": [{"TEXT": {"REGEX": r"^Mumbai$"}}]},
            {"label": "GPE", "pattern": [{"TEXT": {"REGEX": r"^Delhi$"}}]},
        ]
        ruler.add_patterns(patterns)
        return nlp
    except:
        st.error("Please install spacy model: python -m spacy download en_core_web_sm")
        return None

# Initialize
download_success = download_nltk_data()
nlp_spacy = load_spacy_model()

# Helper functions
def preprocess_text(text, case_sensitive=True):
    """Enhanced preprocessing with options"""
    if not case_sensitive:
        text = text.lower()
    # Remove extra whitespace but preserve sentence structure
    text = ' '.join(text.split())
    return text

def get_tokens(text, case_sensitive=True):
    """Enhanced tokenization with preprocessing"""
    processed_text = preprocess_text(text, case_sensitive)
    tokens = word_tokenize(processed_text)
    return ['<s>'] + tokens + ['</s>']

def train_ngram_model(tokens, n=2, smoothing='none', alpha=1.0):
    """Enhanced n-gram model with smoothing options"""
    model = defaultdict(lambda: defaultdict(lambda: 0))
    vocab_size = len(set(tokens))
    
    # Count n-grams
    for ngram in ngrams(tokens, n):
        prefix, word = tuple(ngram[:-1]), ngram[-1]
        model[prefix][word] += 1
    
    # Apply smoothing and normalize
    for prefix in model:
        if smoothing == 'laplace':
            total = float(sum(model[prefix].values()) + alpha * vocab_size)
            for word in set(tokens):  # Add all vocab words
                model[prefix][word] = (model[prefix][word] + alpha) / total
        else:
            total = float(sum(model[prefix].values()))
            for word in model[prefix]:
                model[prefix][word] /= total
    
    return model, vocab_size

def calculate_perplexity_detailed(model, test_tokens, vocab_size, smoothing='none', alpha=1.0):
    """Calculate perplexity with detailed breakdown"""
    N = len(test_tokens) - 1  # Don't count <s>
    log_prob = 0
    word_probs = []
    
    for i in range(1, len(test_tokens)):
        prefix = tuple(test_tokens[i-1:i])
        word = test_tokens[i]
        
        if smoothing == 'laplace':
            count = model[prefix].get(word, 0)
            prefix_count = sum(model[prefix].values()) if prefix in model else 0
            prob = (count + alpha) / (prefix_count + alpha * vocab_size)
        else:
            prob = model[prefix].get(word, 1e-6)
        
        word_probs.append((word, prob))
        log_prob += math.log(prob)
    
    perplexity = math.exp(-log_prob / N)
    return perplexity, word_probs, log_prob

def get_pos_description(tag):
    """Get human-readable description of POS tags"""
    pos_descriptions = {
        'CC': 'Coordinating conjunction',
        'CD': 'Cardinal number',
        'DT': 'Determiner',
        'EX': 'Existential there',
        'FW': 'Foreign word',
        'IN': 'Preposition/subordinating conjunction',
        'JJ': 'Adjective',
        'JJR': 'Adjective, comparative',
        'JJS': 'Adjective, superlative',
        'LS': 'List item marker',
        'MD': 'Modal',
        'NN': 'Noun, singular',
        'NNS': 'Noun, plural',
        'NNP': 'Proper noun, singular',
        'NNPS': 'Proper noun, plural',
        'PDT': 'Predeterminer',
        'POS': 'Possessive ending',
        'PRP': 'Personal pronoun',
        'PRP': 'Possessive pronoun',
        'RB': 'Adverb',
        'RBR': 'Adverb, comparative',
        'RBS': 'Adverb, superlative',
        'RP': 'Particle',
        'SYM': 'Symbol',
        'TO': 'to',
        'UH': 'Interjection',
        'VB': 'Verb, base form',
        'VBD': 'Verb, past tense',
        'VBG': 'Verb, gerund/present participle',
        'VBN': 'Verb, past participle',
        'VBP': 'Verb, non-3rd person singular present',
        'VBZ': 'Verb, 3rd person singular present',
        'WDT': 'Wh-determiner',
        'WP': 'Wh-pronoun',
        'WP': 'Possessive wh-pronoun',
        'WRB': 'Wh-adverb'
    }
    return pos_descriptions.get(tag, 'Unknown tag')

# Main header
st.markdown("""
<div class="main-header">
    <h1>NLP Lab 3: N-gram Models, POS Tagging & Named Entity Recognition</h1>
    <p><strong>R Abhijit Srivathsan - 2448044</strong></p>
</div>
""", unsafe_allow_html=True)

# Sidebar with theory
with st.sidebar:
    st.markdown("## 📚 NLP Concepts")
    
    with st.expander("🔤 N-gram Models"):
        st.markdown("""
        **N-gram models** predict the next word based on the previous n-1 words:
        - **Unigram**: P(w) - word frequency
        - **Bigram**: P(w₂|w₁) - conditional probability
        - **Trigram**: P(w₃|w₁,w₂) - context of 2 words
        
        **Smoothing techniques** handle unseen n-grams:
        - Laplace smoothing adds α to all counts
        """)
    
    with st.expander("📊 Perplexity"):
        st.markdown("""
        **Perplexity** measures how well a model predicts text:
        
        PP(W) = 2^(-1/N ∑ log₂ P(wᵢ|w₁...wᵢ₋₁))
        
        - Lower perplexity = better model
        - Perplexity of N means the model is as confused as if it had to choose uniformly among N possibilities
        """)
    
    with st.expander("🏷️ POS Tagging"):
        st.markdown("""
        **Part-of-Speech tagging** assigns grammatical categories:
        - Uses statistical models (HMM, CRF)
        - Context-dependent disambiguation
        - Penn Treebank tagset (NN, VB, JJ, etc.)
        """)
    
    with st.expander("👤 Named Entity Recognition"):
        st.markdown("""
        **NER** identifies entities in text:
        - **BIO tagging**: B-egin, I-nside, O-utside
        - Entity types: PERSON, ORG, GPE, etc.
        - Uses neural networks (BiLSTM-CRF, BERT)
        """)

# Create enhanced tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔤 Unigram Analysis", 
    "🔗 Bigram Analysis", 
    "📊 N-gram & Perplexity", 
    "🏷️ POS Tagging", 
    "👤 Named Entity Recognition"
])

# Tab 1: Enhanced Unigram Analysis
with tab1:
    st.header("🔤 Unigram Probability Analysis")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        text1 = st.text_area("Enter text for unigram analysis:", 
                           value="I love living in Bangalore", 
                           key="unigram_text", height=100)
    with col2:
        case_sensitive = st.checkbox("Case sensitive", value=True, key="unigram_case")
        show_formula = st.checkbox("Show mathematical formulation", value=True)
    
    if show_formula:
        st.markdown("""
        <div class="formula-box">
        <strong>Unigram Probability Formula:</strong><br>
        P(w) = Count(w) / Total_Words<br><br>
        <strong>Maximum Likelihood Estimation (MLE)</strong><br>
        Each word probability is estimated from relative frequency in corpus
        </div>
        """, unsafe_allow_html=True)
    
    if st.button("🔍 Analyze Unigrams", type="primary"):
        if text1:
            tokens = get_tokens(text1, case_sensitive)
            freq = FreqDist(tokens)
            total = len(tokens)
            
            # Create results dataframe
            results = []
            for word in sorted(set(tokens)):
                count = freq[word]
                prob = count / total
                results.append({
                    'Word': word,
                    'Count': count,
                    'Probability': prob,
                    'Log Probability': math.log(prob),
                    '-Log Probability': -math.log(prob)
                })
            
            df = pd.DataFrame(results)
            
            # Display results
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.subheader("📊 Unigram Statistics")
                st.dataframe(df, use_container_width=True)
                
                # Additional statistics
                st.markdown(f"""
                <div class="metric-card">
                <strong>Corpus Statistics:</strong><br>
                • Total tokens: {total}<br>
                • Unique tokens: {len(set(tokens))}<br>
                • Type-token ratio: {len(set(tokens))/total:.3f}<br>
                • Entropy: {-sum(p * math.log2(p) for p in df['Probability']):.3f} bits
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Visualization
                fig = px.bar(df, x='Word', y='Probability', 
                           title="Unigram Probability Distribution",
                           color='Probability', color_continuous_scale='viridis')
                fig.update_layout(xaxis_tickangle=-45, height=400)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Please enter some text")

# Tab 2: Enhanced Bigram Analysis
with tab2:
    st.header("🔗 Bigram Probability Analysis")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        text2 = st.text_area("Enter text for bigram analysis:", 
                           value="I love living in Bangalore", 
                           key="bigram_text", height=100)
    with col2:
        case_sensitive2 = st.checkbox("Case sensitive", value=True, key="bigram_case")
        auto_analyze = st.checkbox("Auto-analyze all consecutive pairs", value=True)
    
    st.markdown("""
    <div class="formula-box">
    <strong>Bigram Probability Formula:</strong><br>
    P(w₂|w₁) = Count(w₁, w₂) / Count(w₁)<br><br>
    <strong>Chain Rule Application:</strong><br>
    P(sentence) = ∏ P(wᵢ|wᵢ₋₁)
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        w1 = st.text_input("First word (w₁):", value="I" if not auto_analyze else "")
    with col2:
        w2 = st.text_input("Second word (w₂):", value="love" if not auto_analyze else "")
    
    if st.button("🔍 Analyze Bigrams", type="primary"):
        if text2:
            tokens = get_tokens(text2, case_sensitive2)
            bigrams = list(ngrams(tokens, 2))
            bigram_freq = FreqDist(bigrams)
            unigram_freq = FreqDist(tokens)
            
            if auto_analyze:
                # Analyze all consecutive pairs in the sentence
                st.subheader("📈 All Consecutive Bigram Probabilities")
                
                results = []
                total_log_prob = 0
                
                for i in range(len(tokens) - 1):
                    w1_curr = tokens[i]
                    w2_curr = tokens[i + 1]
                    
                    bigram_count = bigram_freq[(w1_curr, w2_curr)]
                    unigram_count = unigram_freq[w1_curr]
                    prob = bigram_count / unigram_count if unigram_count != 0 else 0
                    log_prob = math.log(prob) if prob > 0 else float('-inf')
                    total_log_prob += log_prob
                    
                    results.append({
                        'Position': f"{i} → {i+1}",
                        'Bigram': f"({w1_curr}, {w2_curr})",
                        'P(w₂|w₁)': f"P({w2_curr}|{w1_curr})",
                        'Count(w₁,w₂)': bigram_count,
                        'Count(w₁)': unigram_count,
                        'Probability': prob,
                        'Log Probability': log_prob if log_prob != float('-inf') else 'undefined'
                    })
                
                df_results = pd.DataFrame(results)
                st.dataframe(df_results, use_container_width=True)
                
                # Sentence probability
                sentence_prob = math.exp(total_log_prob) if total_log_prob != float('-inf') else 0
                st.markdown(f"""
                <div class="metric-card">
                <strong>Sentence Analysis:</strong><br>
                • Total log probability: {total_log_prob:.6f}<br>
                • Sentence probability: {sentence_prob:.2e}<br>
                • Perplexity: {math.exp(-total_log_prob/len(tokens)) if total_log_prob != float('-inf') else 'undefined':.2f}
                </div>
                """, unsafe_allow_html=True)
                
                # Visualization
                if len(results) > 1:
                    fig = px.bar(df_results, x='Position', y='Probability',
                               title="Bigram Probabilities in Sequence",
                               hover_data=['Bigram'])
                    st.plotly_chart(fig, use_container_width=True)
            
            # Manual bigram analysis
            if w1 and w2:
                st.subheader(f"🎯 Specific Bigram Analysis: P({w2}|{w1})")
                
                if not case_sensitive2:
                    w1, w2 = w1.lower(), w2.lower()
                
                bigram_count = bigram_freq[(w1, w2)]
                unigram_count = unigram_freq[w1]
                prob = bigram_count / unigram_count if unigram_count != 0 else 0
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Count(w₁, w₂)", bigram_count)
                with col2:
                    st.metric("Count(w₁)", unigram_count)
                with col3:
                    st.metric("P(w₂|w₁)", f"{prob:.6f}")
        else:
            st.warning("⚠️ Please enter some text")

# Tab 3: Enhanced N-gram and Perplexity
with tab3:
    st.header("📊 N-gram Model Training & Perplexity Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        train_text = st.text_area("Training corpus:", 
                                value="I love machine learning. Machine learning is fascinating. Learning never stops.",
                                key="train_text", height=120)
    with col2:
        test_text = st.text_area("Test corpus:", 
                               value="I love learning",
                               key="test_text", height=120)
    
    # Model parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        n_value = st.selectbox("N-gram order:", [2, 3], index=0)
    with col2:
        smoothing = st.selectbox("Smoothing:", ["none", "laplace"], index=0)
    with col3:
        alpha = st.slider("Alpha (for Laplace):", 0.1, 2.0, 1.0, 0.1)
    
    if st.button("🚀 Train Model & Analyze", type="primary"):
        if train_text and test_text:
            # Train model
            train_tokens = get_tokens(train_text, True)
            model, vocab_size = train_ngram_model(train_tokens, n=n_value, smoothing=smoothing, alpha=alpha)
            
            # Calculate perplexity
            test_tokens = get_tokens(test_text, True)
            perplexity, word_probs, log_prob = calculate_perplexity_detailed(
                model, test_tokens, vocab_size, smoothing, alpha
            )
            
            # Display results
            st.subheader("🎯 Model Performance")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Perplexity", f"{perplexity:.2f}")
            with col2:
                st.metric("Log Probability", f"{log_prob:.2f}")
            with col3:
                st.metric("Vocab Size", vocab_size)
            with col4:
                st.metric("Test Length", len(test_tokens)-1)
            
            # Perplexity explanation
            st.markdown(f"""
            <div class="explanation-box">
            <strong>🧠 Perplexity Interpretation:</strong><br><br>
            
            <strong>Your model's perplexity: {perplexity:.2f}</strong><br><br>
            
            <strong>What this means:</strong><br>
            • On average, your model is as confused as if it had to choose uniformly among {perplexity:.0f} possibilities<br>
            • Lower perplexity = better model performance<br>
            • Perplexity of 1.0 would mean perfect prediction<br><br>
            
            <strong>Benchmarks:</strong><br>
            • Perplexity < 100: Very good for small vocabulary<br>
            • Perplexity 100-300: Reasonable for medium vocabulary<br>
            • Perplexity > 500: Poor model performance<br><br>
            
            <strong>Formula:</strong> PP(W) = 2^(-1/N ∑ log₂ P(wᵢ|context))
            </div>
            """, unsafe_allow_html=True)
            
            # Word-by-word analysis
            st.subheader("🔍 Word-by-Word Probability Breakdown")
            
            word_df = pd.DataFrame([
                {
                    'Word': word,
                    'Probability': prob,
                    'Log₂ Probability': math.log2(prob),
                    'Surprise': -math.log2(prob),
                    'Contribution to Perplexity': -math.log2(prob) / len(word_probs)
                }
                for word, prob in word_probs
            ])
            
            st.dataframe(word_df, use_container_width=True)
            
            # Visualization
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("Word Probabilities", "Surprise Values"),
                specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
            )
            
            fig.add_trace(
                go.Bar(x=word_df['Word'], y=word_df['Probability'], name="Probability"),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(x=word_df['Word'], y=word_df['Surprise'], name="Surprise", marker_color='red'),
                row=2, col=1
            )
            
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Model details
            with st.expander("🔧 Model Details & N-gram Counts"):
                st.write("**Learned N-grams:**")
                ngram_data = []
                for prefix, words in model.items():
                    for word, prob in words.items():
                        if prob > 0:
                            ngram_data.append({
                                'Context': ' '.join(prefix),
                                'Next Word': word,
                                'Probability': prob
                            })
                
                ngram_df = pd.DataFrame(ngram_data).sort_values('Probability', ascending=False)
                st.dataframe(ngram_df, use_container_width=True)
        else:
            st.warning("⚠️ Please enter both training and test text")

# Tab 4: Enhanced POS Tagging
with tab4:
    st.header("🏷️ Advanced Part-of-Speech Tagging")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        text4 = st.text_area("Enter text for POS analysis:", 
                           value="Abhijit loves studying NLP at Bangalore University.",
                           key="pos_text", height=100)
    with col2:
        case_sensitive4 = st.checkbox("Case sensitive", value=False, key="pos_case")
        show_details = st.checkbox("Show detailed analysis", value=True)
    
    if st.button("🔍 Analyze POS Tags", type="primary"):
        if text4:
            processed_text = preprocess_text(text4, case_sensitive4)
            tokens = word_tokenize(processed_text)
            pos_tags = pos_tag(tokens)
            
            # Create detailed analysis
            pos_data = []
            tag_counts = Counter()
            
            for i, (word, tag) in enumerate(pos_tags):
                tag_counts[tag] += 1
                pos_data.append({
                    'Position': i + 1,
                    'Word': word,
                    'POS Tag': tag,
                    'Description': get_pos_description(tag),
                    'Word Length': len(word),
                    'Is Capitalized': word[0].isupper() if word else False
                })
            
            df_pos = pd.DataFrame(pos_data)
            
            # Display results
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.subheader("📊 POS Tagging Results")
                st.dataframe(df_pos, use_container_width=True)
            
            with col2:
                # POS distribution
                fig = px.pie(values=list(tag_counts.values()), 
                           names=list(tag_counts.keys()),
                           title="POS Tag Distribution")
                st.plotly_chart(fig, use_container_width=True)
            
            if show_details:
                # Tag statistics
                st.subheader("📈 Detailed POS Statistics")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Tokens", len(tokens))
                with col2:
                    st.metric("Unique POS Tags", len(tag_counts))
                with col3:
                    st.metric("Most Common Tag", max(tag_counts, key=tag_counts.get))
                
                # Linguistic features
                st.markdown("""
                <div class="explanation-box">
                <strong>🔍 Linguistic Analysis:</strong><br>
                • <strong>Lexical Diversity:</strong> Ratio of unique words to total words<br>
                • <strong>Syntactic Complexity:</strong> Variety of POS tags used<br>
                • <strong>Content vs Function Words:</strong> Nouns/Verbs vs Articles/Prepositions
                </div>
                """, unsafe_allow_html=True)
                
                # Tag frequency chart
                tag_df = pd.DataFrame(list(tag_counts.items()), columns=['POS Tag', 'Frequency'])
                fig = px.bar(tag_df, x='POS Tag', y='Frequency', 
                           title="POS Tag Frequency Distribution")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Please enter some text")

# Tab 5: Enhanced NER
with tab5:
    st.header("👤 Advanced Named Entity Recognition")
    
    if nlp_spacy is None:
        st.error("⚠️ SpaCy model not loaded. Please install: `python -m spacy download en_core_web_sm`")
    else:
        col1, col2 = st.columns([2, 1])
        with col1:
            text5 = st.text_area("Enter text for NER analysis:", 
                               value="Abhijit Srivathsan works at Google in Bangalore. He studied at IIT Delhi with Priya Sharma.",
                               key="ner_text", height=100)
        with col2:
            case_sensitive5 = st.checkbox("Case sensitive", value=False, key="ner_case")
            show_confidence = st.checkbox("Show confidence scores", value=True)
        
        if st.button("🔍 Extract Named Entities", type="primary"):
            if text5:
                processed_text = preprocess_text(text5, case_sensitive5)
                doc = nlp_spacy(processed_text)
                
                # Token-level analysis
                st.subheader("🏷️ Token-Level NER Tags (BIO Format)")
                
                token_data = []
                for token in doc:
                    entity_type = f"{token.ent_iob_}-{token.ent_type_}" if token.ent_type_ else "O"
                    token_data.append({
                        'Token': token.text,
                        'BIO Tag': entity_type,
                        'Entity Type': token.ent_type_ or "None",
                        'POS': token.pos_,
                        'Lemma': token.lemma_,
                        'Is Alpha': token.is_alpha,
                        'Is Stop': token.is_stop
                    })
                
                df_tokens = pd.DataFrame(token_data)
                st.dataframe(df_tokens, use_container_width=True)
                
                # Entity-level analysis
                st.subheader("👥 Extracted Entities")
                
                if doc.ents:
                    entity_data = []
                    for ent in doc.ents:
                        entity_data.append({
                            'Entity': ent.text,
                            'Label': ent.label_,
                            'Description': spacy.explain(ent.label_),
                            'Start': ent.start_char,
                            'End': ent.end_char,
                            'Confidence': getattr(ent, 'confidence', 'N/A') if show_confidence else 'N/A'
                        })
                    
                    df_entities = pd.DataFrame(entity_data)
                    st.dataframe(df_entities, use_container_width=True)
                    
                    # Entity visualization
                    entity_counts = Counter([ent['Label'] for ent in entity_data])
                    if entity_counts:
                        fig = px.bar(x=list(entity_counts.keys()), 
                                   y=list(entity_counts.values()),
                                   title="Entity Type Distribution",
                                   labels={'x': 'Entity Type', 'y': 'Count'})
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Highlighted text
                    st.subheader("📝 Text with Highlighted Entities")
                    html = spacy.displacy.render(doc, style="ent", jupyter=False)
                    st.components.v1.html(html, height=200, scrolling=True)
                    
                else:
                    st.info("No named entities found in the text.")
                
                # NER Statistics
                st.subheader("📊 NER Analysis Statistics")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Tokens", len(doc))
                with col2:
                    st.metric("Named Entities", len(doc.ents))
                with col3:
                    entity_density = len(doc.ents) / len(doc) if len(doc) > 0 else 0
                    st.metric("Entity Density", f"{entity_density:.2%}")
                
                # Additional NER insights
                st.markdown("""
                <div class="explanation-box">
                <strong>🧠 NER Analysis Insights:</strong><br><br>
                
                <strong>BIO Tagging Scheme:</strong><br>
                • <strong>B-</strong>: Beginning of an entity<br>
                • <strong>I-</strong>: Inside/continuation of an entity<br>
                • <strong>O</strong>: Outside any entity<br><br>
                
                <strong>Common Entity Types:</strong><br>
                • <strong>PERSON</strong>: People, including fictional characters<br>
                • <strong>ORG</strong>: Companies, agencies, institutions<br>
                • <strong>GPE</strong>: Countries, cities, states (Geo-Political Entities)<br>
                • <strong>DATE</strong>: Absolute or relative dates or periods<br>
                • <strong>MONEY</strong>: Monetary values, including unit
                </div>
                """, unsafe_allow_html=True)
                
                # Custom entity patterns info
                with st.expander("🔧 Custom NER Patterns for Indian Names/Places"):
                    st.markdown("""
                    **Enhanced Recognition Patterns:**
                    - Names ending in 'jit' (e.g., Abhijit, Ranjit)
                    - Names ending in 'deep' (e.g., Pradeep, Sandeep)
                    - Names ending in 'raj' (e.g., Raj, Suraj)
                    - Indian cities: Bangalore, Chennai, Mumbai, Delhi
                    
                    **Note:** The model has been enhanced with custom entity ruler patterns 
                    to better recognize Indian names and places that might be missed by 
                    the default English model.
                    """)
            else:
                st.warning("⚠️ Please enter some text")

# Helper function for POS descriptions
