import streamlit as st
import nltk
from collections import defaultdict, Counter
import math
import pandas as pd
import spacy

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)
        nltk.download('brown', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('maxent_ne_chunker', quiet=True)
        nltk.download('words', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        return True
    except Exception as e:
        st.error(f"Error downloading NLTK data: {str(e)}")
        return False

# Load spaCy model
@st.cache_resource
def load_spacy_model():
    try:
        nlp = spacy.load("en_core_web_sm")
        return nlp
    except:
        st.error("Please install spaCy model: python -m spacy download en_core_web_sm")
        return None

# N-gram Language Model Class
class NgramLanguageModel:
    def __init__(self, n=2):
        self.n = n
        self.ngram_counts = defaultdict(int)
        self.context_counts = defaultdict(int)
        self.vocabulary = set()
        
    def train(self, text):
        # Tokenize text
        tokens = ['<s>'] + nltk.word_tokenize(text.lower()) + ['</s>']
        self.vocabulary.update(tokens)
        
        # Count n-grams
        for i in range(len(tokens) - self.n + 1):
            ngram = tuple(tokens[i:i + self.n])
            context = ngram[:-1]
            self.ngram_counts[ngram] += 1
            self.context_counts[context] += 1
            
    def get_probability(self, ngram):
        ngram = tuple(ngram)
        context = ngram[:-1]
        
        if context not in self.context_counts:
            return 0.0
            
        # Add-one smoothing
        vocab_size = len(self.vocabulary)
        ngram_count = self.ngram_counts.get(ngram, 0) + 1
        context_count = self.context_counts[context] + vocab_size
        
        return ngram_count / context_count
    
    def calculate_perplexity(self, test_text):
        tokens = ['<s>'] + nltk.word_tokenize(test_text.lower()) + ['</s>']
        log_prob_sum = 0
        count = 0
        
        for i in range(len(tokens) - self.n + 1):
            ngram = tuple(tokens[i:i + self.n])
            prob = self.get_probability(ngram)
            if prob > 0:
                log_prob_sum += math.log2(prob)
                count += 1
                
        if count == 0:
            return float('inf')
            
        avg_log_prob = log_prob_sum / count
        perplexity = 2 ** (-avg_log_prob)
        return perplexity

# Streamlit App
def main():
    st.title("NLP Lab 3: N-gram, POS, and NER")
    st.markdown("---")
    
    # Download NLTK data
    if not download_nltk_data():
        st.error("Failed to download NLTK data automatically.")
        st.info("Please run these commands in your terminal:")
        st.code("""
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('brown')
nltk.download('wordnet')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('punkt_tab')
        """)
        
        if st.button("Retry Download"):
            st.rerun()
    
    # Sidebar for navigation
    task = st.sidebar.selectbox(
        "Select Task",
        ["Q1: N-gram Language Model", "Q2: POS Tagging", "Q3: Named Entity Recognition"]
    )
    
    if task == "Q1: N-gram Language Model":
        st.header("Q1: Estimating N-gram Probability")
        
        # Part a: Unigram lookup table
        st.subheader("a) Unigram Probability Lookup Table")
        
        sample_text = st.text_area(
            "Enter training text:",
            "I want english food. I like green food. Sam likes english food.",
            height=100
        )
        
        if st.button("Generate Unigram Table"):
            tokens = nltk.word_tokenize(sample_text.lower())
            unigram_counts = Counter(tokens)
            total_count = sum(unigram_counts.values())
            
            # Create lookup table
            lookup_table = pd.DataFrame([
                {"Word": word, "Count": count, "Probability": count/total_count}
                for word, count in unigram_counts.items()
            ])
            lookup_table = lookup_table.sort_values("Count", ascending=False)
            
            st.dataframe(lookup_table)
        
        # Part b: Bigram probability calculation
        st.subheader("b) Bigram Probability Calculation")
        
        col1, col2 = st.columns(2)
        with col1:
            word1 = st.text_input("First word:", "Sam")
        with col2:
            word2 = st.text_input("Second word:", "am")
            
        if st.button("Calculate Bigram Probability"):
            model = NgramLanguageModel(n=2)
            model.train(sample_text)
            
            # Calculate P(Sam/am)
            prob1 = model.get_probability([word1.lower(), word2.lower()])
            st.write(f"P({word1}/{word2}) = {prob1:.6f}")
            
            # Calculate P(green/like)
            prob2 = model.get_probability(["green", "like"])
            st.write(f"P(green/like) = {prob2:.6f}")
        
        # Part c: N-gram perplexity
        st.subheader("c) N-gram Model Perplexity")
        
        n_value = st.slider("Select n for n-gram model:", 1, 5, 2)
        test_text = st.text_input("Enter test text:", "I want food")
        
        if st.button("Calculate Perplexity"):
            # Load Brown corpus for training
            from nltk.corpus import brown
            train_text = " ".join(brown.words()[:5000])  # Use first 5000 words
            
            model = NgramLanguageModel(n=n_value)
            model.train(train_text)
            
            perplexity = model.calculate_perplexity(test_text)
            st.success(f"Perplexity of '{test_text}' with {n_value}-gram model: {perplexity:.2f}")
    
    elif task == "Q2: POS Tagging":
        st.header("Q2: Part-of-Speech Tagging")
        
        input_text = st.text_area(
            "Enter text for POS tagging:",
            "The quick brown fox jumps over the lazy dog.",
            height=100
        )
        
        if st.button("Tag Parts of Speech"):
            # Tokenize
            tokens = nltk.word_tokenize(input_text)
            
            # POS tagging
            pos_tags = nltk.pos_tag(tokens)
            
            # Display results
            st.subheader("POS Tags:")
            
            # Create a dataframe for better visualization
            pos_df = pd.DataFrame(pos_tags, columns=["Word", "POS Tag"])
            st.dataframe(pos_df)
            
            # Show tag meanings
            with st.expander("Common POS Tag Meanings"):
                st.markdown("""
                - **NN**: Noun, singular
                - **NNS**: Noun, plural
                - **VB**: Verb, base form
                - **VBZ**: Verb, 3rd person singular present
                - **JJ**: Adjective
                - **RB**: Adverb
                - **DT**: Determiner
                - **IN**: Preposition
                """)
    
    elif task == "Q3: Named Entity Recognition":
        st.header("Q3: Named Entity Recognition (NER)")
        
        # Load spaCy model
        nlp = load_spacy_model()
        if not nlp:
            return
        
        input_text = st.text_area(
            "Enter text for NER:",
            "John lives in New York",
            height=100
        )
        
        if st.button("Extract Named Entities"):
            # Process with spaCy
            doc = nlp(input_text)
            
            st.subheader("Named Entities (spaCy):")
            
            # Display entities
            entities = []
            for ent in doc.ents:
                entities.append({
                    "Text": ent.text,
                    "Label": ent.label_,
                    "Start": ent.start_char,
                    "End": ent.end_char
                })
            
            if entities:
                ent_df = pd.DataFrame(entities)
                st.dataframe(ent_df)
                
                # Show BIO tagging format
                st.subheader("BIO Tagging Format:")
                bio_tags = []
                for token in doc:
                    if token.ent_iob_ == 'O':
                        bio_tags.append((token.text, 'O'))
                    else:
                        bio_tags.append((token.text, f"{token.ent_iob_}-{token.ent_type_}"))
                
                bio_df = pd.DataFrame(bio_tags, columns=["Token", "BIO Tag"])
                st.dataframe(bio_df)
                
                # Visualize entities
                st.subheader("Entity Visualization:")
                from spacy import displacy
                html = displacy.render(doc, style="ent", jupyter=False)
                st.markdown(html, unsafe_allow_html=True)
            else:
                st.info("No named entities found in the text.")
            
            # Show entity type meanings
            with st.expander("Common Entity Types"):
                st.markdown("""
                - **PERSON**: People, including fictional
                - **LOC**: Locations (cities, countries, etc.)
                - **ORG**: Organizations, companies
                - **DATE**: Dates or periods
                - **GPE**: Geopolitical entities (countries, cities, states)
                """)

if __name__ == "__main__":
    main()