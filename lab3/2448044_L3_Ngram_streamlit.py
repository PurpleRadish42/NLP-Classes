import streamlit as st
import nltk
from nltk.util import ngrams
from nltk import FreqDist, pos_tag, word_tokenize
from collections import defaultdict
import math
import spacy

# Configure page
st.set_page_config(
    page_title="NLP Lab 3: N-gram, POS, NER",
    # layout="wide",
    # initial_sidebar_state="expanded"
)

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')

# Load spacy model
@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

# Initialize
download_nltk_data()
nlp_spacy = load_spacy_model()

# Helper functions
def get_tokens(text):
    return ['<s>'] + word_tokenize(text) + ['</s>']

def train_ngram_model(tokens, n=2):
    model = defaultdict(lambda: defaultdict(lambda: 0))
    for ngram in ngrams(tokens, n):
        prefix, word = tuple(ngram[:-1]), ngram[-1]
        model[prefix][word] += 1
    for prefix in model:
        total = float(sum(model[prefix].values()))
        for word in model[prefix]:
            model[prefix][word] /= total # type: ignore
    return model

# Set page title
st.title("NLP Lab 3: N-gram, POS, NER")
st.write("R Abhijit Srivathsan - 2448044")

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Unigram Prob", "Bigram Prob", "N-gram & Perplexity", "POS Tagging", "NER"])

# Tab 1: Unigram Probability
with tab1:
    st.header("Unigram Probability")
    text1 = st.text_area("Enter text:", key="unigram_text")
    
    if st.button("Calculate Unigram Probabilities"):
        if text1:
            tokens = get_tokens(text1)
            freq = FreqDist(tokens)
            total = len(tokens)
            
            st.subheader("Results:")
            for word in sorted(set(tokens)):
                prob = freq[word] / total
                st.write(f"P({word}) = {prob:.6f}")
        else:
            st.warning("Please enter some text")

# Tab 2: Bigram Probability
with tab2:
    st.header("Bigram Probability")
    text2 = st.text_area("Enter text:", key="bigram_text")
    col1, col2 = st.columns(2)
    with col1:
        w1 = st.text_input("First word:")
    with col2:
        w2 = st.text_input("Second word:")
    
    if st.button("Calculate Bigram Probability"):
        if text2 and w1 and w2:
            tokens = get_tokens(text2)
            bigrams = list(ngrams(tokens, 2))
            bigram_freq = FreqDist(bigrams)
            unigram_freq = FreqDist(tokens)
            
            prob = bigram_freq[(w1, w2)] / unigram_freq[w1] if unigram_freq[w1] != 0 else 0
            st.success(f"P({w2}|{w1}) = {prob:.6f}")
        else:
            st.warning("Please enter text and both words")

# Tab 3: N-gram Model and Perplexity
with tab3:
    st.header("Train N-gram Model and Calculate Perplexity")
    train_text = st.text_area("Enter training text:", key="train_text")
    test_text = st.text_area("Enter test text for perplexity:", key="test_text")
    
    if st.button("Train Model & Calculate Perplexity"):
        if train_text and test_text:
            # Train model
            train_tokens = get_tokens(train_text)
            model = train_ngram_model(train_tokens, n=2)
            
            # Calculate perplexity
            test_tokens = get_tokens(test_text)
            N = len(test_tokens)
            log_prob = 0
            
            for i in range(1, len(test_tokens)):
                prefix = tuple(test_tokens[i-1:i])
                word = test_tokens[i]
                prob = model[prefix].get(word, 1e-6)
                log_prob += math.log(prob)
            
            ppl = math.exp(-log_prob / N)
            st.info(f"Perplexity = {ppl:.2f}")
        else:
            st.warning("Please enter both training and test text")

# Tab 4: POS Tagging
with tab4:
    st.header("Part-of-Speech Tagging")
    text4 = st.text_area("Enter text:", key="pos_text")
    
    if st.button("Tag POS"):
        if text4:
            tokens = word_tokenize(text4)
            pos_tags = pos_tag(tokens)
            
            st.subheader("POS Tags:")
            for word, tag in pos_tags:
                st.write(f"{word} → {tag}")
        else:
            st.warning("Please enter some text")

# Tab 5: Named Entity Recognition
with tab5:
    st.header("Named Entity Recognition (NER)")
    text5 = st.text_area("Enter text:", key="ner_text")
    
    if st.button("Extract Named Entities"):
        if text5:
            doc = nlp_spacy(text5)
            
            st.subheader("Named Entities:")
            for token in doc:
                entity_type = f"{token.ent_iob_}-{token.ent_type_}" if token.ent_type_ else "O"
                st.write(f"{token.text} → {entity_type}")
        else:
            st.warning("Please enter some text")

# # Footer
# st.markdown("---")
# st.markdown("### Instructions:")
# st.markdown("1. **Unigram Prob**: Shows probability of each word in the text")
# st.markdown("2. **Bigram Prob**: Shows conditional probability P(word2|word1)")
# st.markdown("3. **N-gram & Perplexity**: Trains a bigram model and calculates perplexity")
# st.markdown("4. **POS Tagging**: Tags each word with its part of speech")
# st.markdown("5. **NER**: Identifies named entities in the text")