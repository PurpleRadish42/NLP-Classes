import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import random
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Word2Vec Dense Vectorization Lab",
    page_icon="🔤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1.5rem 0 1rem 0;
        color: #4a4a4a;
        border-bottom: 2px solid #667eea;
        padding-bottom: 0.5rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .info-box {
        background: rgba(102, 126, 234, 0.1);
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    @media (prefers-color-scheme: dark) {
        .section-header {
            color: #ffffff;
        }
    }
</style>
""", unsafe_allow_html=True)

# Training callback for Gensim
class TrainingCallback(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0
        self.losses = []
    
    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        if self.epoch == 0:
            current_loss = loss
        else:
            current_loss = loss - self.previous_loss
        self.losses.append(current_loss)
        self.previous_loss = loss
        self.epoch += 1

# Default corpus
DEFAULT_SENTENCES = [
    "NLP is fun and exciting",
    "We are learning natural language processing",
    "Machine learning powers modern NLP applications",
    "Natural language processing is a fascinating field",
    "Deep learning improves NLP performance",
    "We enjoy exploring text mining techniques",
    "AI is transforming language understanding",
    "Text mining reveals hidden patterns in data",
    "Word embeddings capture semantic relationships",
    "Neural networks excel at language tasks"
]

def preprocess_text(sentences):
    """Preprocess sentences into tokens"""
    processed = []
    for sentence in sentences:
        tokens = sentence.lower().split()
        processed.append(tokens)
    return processed

def create_vocabulary_stats(sentences):
    """Create vocabulary statistics"""
    tokens = [word for sentence in sentences for word in sentence]
    vocab_counter = Counter(tokens)
    
    df = pd.DataFrame([
        {"Word": word, "Frequency": count, "Relative Freq": count/len(tokens)}
        for word, count in vocab_counter.most_common()
    ])
    
    return df, vocab_counter, len(set(tokens))

def train_word2vec_models(sentences, vector_size=50, window=2, min_count=1, workers=1, epochs=5, sg=1):
    """Train Word2Vec model with specified architecture"""
    callback = TrainingCallback()
    
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg,  # 1 for Skip-gram, 0 for CBOW
        epochs=epochs,
        compute_loss=True,
        callbacks=[callback]
    )
    
    return model, callback.losses

def visualize_embeddings(model, method='PCA', title="Word Embeddings"):
    """Visualize word embeddings using PCA or t-SNE"""
    words = list(model.wv.key_to_index.keys())
    vectors = np.array([model.wv[word] for word in words])  # Convert to numpy array
    
    if method == 'PCA':
        reducer = PCA(n_components=2, random_state=42)
        reduced_vectors = reducer.fit_transform(vectors)
        explained_var = reducer.explained_variance_ratio_.sum()
    else:  # t-SNE
        # Fix: Ensure perplexity is valid for small datasets
        n_samples = vectors.shape[0]
        perplexity = min(30, max(1, n_samples - 1))
        reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        reduced_vectors = reducer.fit_transform(vectors)
        explained_var = None
    
    # Create interactive plot with Plotly
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=reduced_vectors[:, 0],
        y=reduced_vectors[:, 1],
        mode='markers+text',
        text=words,
        textposition='top center',
        marker=dict(
            size=10,
            color=np.random.rand(len(words)),
            colorscale='Viridis',
            line=dict(width=1, color='white')
        ),
        hovertemplate='<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f"{title} ({method})",
        xaxis_title=f"{method} Component 1",
        yaxis_title=f"{method} Component 2",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12),
        height=500
    )
    
    return fig, explained_var

def find_similar_words(model, word, topn=5):
    """Find similar words using the trained model"""
    try:
        similar = model.wv.most_similar(word, topn=topn)
        return similar
    except KeyError:
        return None

def compare_models_similarity(model1, model2, word, topn=5):
    """Compare similarity results between two models"""
    sim1 = find_similar_words(model1, word, topn)
    sim2 = find_similar_words(model2, word, topn)
    
    if sim1 is None or sim2 is None:
        return None, None
    
    df1 = pd.DataFrame(sim1, columns=['Word', 'Similarity'])
    df1['Model'] = 'Skip-gram'
    df2 = pd.DataFrame(sim2, columns=['Word', 'Similarity'])
    df2['Model'] = 'CBOW'
    
    return df1, df2

# Main App
def main():
    st.markdown('<h1 class="main-header">🔤 Word2Vec Dense Vectorization Lab</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <strong>Welcome to the Word2Vec Lab!</strong><br>
        This interactive application demonstrates dense word vectorization using both CBOW and Skip-gram architectures.
        Explore how different models learn word representations and compare their performance.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar Configuration
    st.sidebar.markdown("## 🛠️ Configuration")
    
    # Text Input Options
    input_method = st.sidebar.radio(
        "Choose input method:",
        ["Use default corpus", "Upload custom text", "Enter custom sentences"]
    )
    
    sentences = []
    if input_method == "Use default corpus":
        sentences = DEFAULT_SENTENCES
    elif input_method == "Upload custom text":
        uploaded_file = st.sidebar.file_uploader("Upload a text file", type=['txt'])
        if uploaded_file:
            content = uploaded_file.read().decode('utf-8')
            sentences = [line.strip() for line in content.split('\n') if line.strip()]
        else:
            sentences = DEFAULT_SENTENCES
    else:
        custom_text = st.sidebar.text_area(
            "Enter sentences (one per line):",
            value="\n".join(DEFAULT_SENTENCES[:5]),
            height=200
        )
        sentences = [line.strip() for line in custom_text.split('\n') if line.strip()]
    
    # Model Parameters
    st.sidebar.markdown("## 📊 Model Parameters")
    vector_size = st.sidebar.slider("Vector Size", 10, 200, 50, 10)
    window_size = st.sidebar.slider("Window Size", 1, 10, 2)
    min_count = st.sidebar.slider("Min Count", 1, 5, 1)
    epochs = st.sidebar.slider("Epochs", 1, 20, 5)
    
    # Visualization Options
    st.sidebar.markdown("## 📈 Visualization")
    viz_method = st.sidebar.selectbox("Dimensionality Reduction", ["PCA", "t-SNE"])
    
    # Process Data
    if sentences:
        processed_sentences = preprocess_text(sentences)
        vocab_df, vocab_counter, vocab_size = create_vocabulary_stats(processed_sentences)
        
        # Display Corpus Statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📝 Sentences</h3>
                <h2>{len(sentences)}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            total_tokens = sum(len(s) for s in processed_sentences)
            st.markdown(f"""
            <div class="metric-card">
                <h3>🔢 Total Tokens</h3>
                <h2>{total_tokens}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📚 Vocabulary Size</h3>
                <h2>{vocab_size}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            avg_sent_len = np.mean([len(s) for s in processed_sentences])
            st.markdown(f"""
            <div class="metric-card">
                <h3>📏 Avg Sentence Length</h3>
                <h2>{avg_sent_len:.1f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Vocabulary Analysis
        st.markdown('<h2 class="section-header">📊 Vocabulary Analysis</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Word Frequency Distribution")
            fig_bar = px.bar(
                vocab_df.head(15), 
                x='Word', 
                y='Frequency',
                title="Top 15 Most Frequent Words"
            )
            fig_bar.update_layout(height=400)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            st.subheader("Vocabulary Statistics")
            st.dataframe(vocab_df.head(10), use_container_width=True)
        
        # Train Models
        st.markdown('<h2 class="section-header">🤖 Model Training</h2>', unsafe_allow_html=True)
        
        if st.button("🚀 Train Models", type="primary"):
            with st.spinner("Training Skip-gram model..."):
                skipgram_model, sg_losses = train_word2vec_models(
                    processed_sentences, vector_size, window_size, min_count, 1, epochs, sg=1
                )
            
            with st.spinner("Training CBOW model..."):
                cbow_model, cbow_losses = train_word2vec_models(
                    processed_sentences, vector_size, window_size, min_count, 1, epochs, sg=0
                )
            
            # Store models in session state
            st.session_state.skipgram_model = skipgram_model
            st.session_state.cbow_model = cbow_model
            st.session_state.sg_losses = sg_losses
            st.session_state.cbow_losses = cbow_losses
            
            st.success("✅ Models trained successfully!")
        
        # Model Analysis (only if models are trained)
        if 'skipgram_model' in st.session_state and 'cbow_model' in st.session_state:
            st.markdown('<h2 class="section-header">📈 Training Analysis</h2>', unsafe_allow_html=True)
            
            # Training Loss Comparison
            col1, col2 = st.columns(2)
            
            with col1:
                if st.session_state.sg_losses and st.session_state.cbow_losses:
                    loss_df = pd.DataFrame({
                        'Epoch': range(1, len(st.session_state.sg_losses) + 1),
                        'Skip-gram': st.session_state.sg_losses,
                        'CBOW': st.session_state.cbow_losses
                    })
                    
                    fig_loss = px.line(
                        loss_df, 
                        x='Epoch', 
                        y=['Skip-gram', 'CBOW'],
                        title="Training Loss Comparison"
                    )
                    st.plotly_chart(fig_loss, use_container_width=True)
            
            with col2:
                st.subheader("Model Comparison")
                model_stats = pd.DataFrame({
                    'Metric': ['Vocabulary Size', 'Vector Dimensions', 'Window Size'],
                    'Skip-gram': [len(st.session_state.skipgram_model.wv), vector_size, window_size],
                    'CBOW': [len(st.session_state.cbow_model.wv), vector_size, window_size]
                })
                st.dataframe(model_stats, use_container_width=True)
            
            # Word Embeddings Visualization
            st.markdown('<h2 class="section-header">🎯 Word Embeddings Visualization</h2>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Skip-gram Embeddings")
                fig_sg, var_sg = visualize_embeddings(
                    st.session_state.skipgram_model, 
                    viz_method, 
                    "Skip-gram Word Embeddings"
                )
                st.plotly_chart(fig_sg, use_container_width=True)
                if var_sg:
                    st.info(f"Explained variance: {var_sg:.2%}")
            
            with col2:
                st.subheader("CBOW Embeddings")
                fig_cbow, var_cbow = visualize_embeddings(
                    st.session_state.cbow_model, 
                    viz_method, 
                    "CBOW Word Embeddings"
                )
                st.plotly_chart(fig_cbow, use_container_width=True)
                if var_cbow:
                    st.info(f"Explained variance: {var_cbow:.2%}")
            
            # Word Similarity Analysis
            st.markdown('<h2 class="section-header">🔍 Word Similarity Analysis</h2>', unsafe_allow_html=True)
            
            available_words = list(st.session_state.skipgram_model.wv.key_to_index.keys())
            selected_word = st.selectbox("Select a word to find similar words:", available_words)
            topn = st.slider("Number of similar words to show:", 3, 10, 5)
            
            if selected_word:
                col1, col2 = st.columns(2)
                
                # Skip-gram similarities
                with col1:
                    st.subheader("Skip-gram Similar Words")
                    sg_similar = find_similar_words(st.session_state.skipgram_model, selected_word, topn)
                    if sg_similar:
                        sg_df = pd.DataFrame(sg_similar, columns=['Word', 'Similarity'])
                        fig_sg_sim = px.bar(
                            sg_df, 
                            x='Similarity', 
                            y='Word',
                            orientation='h',
                            title=f"Words similar to '{selected_word}' (Skip-gram)"
                        )
                        fig_sg_sim.update_layout(height=300)
                        st.plotly_chart(fig_sg_sim, use_container_width=True)
                
                # CBOW similarities
                with col2:
                    st.subheader("CBOW Similar Words")
                    cbow_similar = find_similar_words(st.session_state.cbow_model, selected_word, topn)
                    if cbow_similar:
                        cbow_df = pd.DataFrame(cbow_similar, columns=['Word', 'Similarity'])
                        fig_cbow_sim = px.bar(
                            cbow_df, 
                            x='Similarity', 
                            y='Word',
                            orientation='h',
                            title=f"Words similar to '{selected_word}' (CBOW)"
                        )
                        fig_cbow_sim.update_layout(height=300)
                        st.plotly_chart(fig_cbow_sim, use_container_width=True)
                
                # Similarity Comparison Table
                if sg_similar and cbow_similar:
                    st.subheader("Model Comparison")
                    comparison_df = pd.DataFrame({
                        'Skip-gram Word': [item[0] for item in sg_similar],
                        'Skip-gram Similarity': [f"{item[1]:.3f}" for item in sg_similar],
                        'CBOW Word': [item[0] for item in cbow_similar],
                        'CBOW Similarity': [f"{item[1]:.3f}" for item in cbow_similar]
                    })
                    st.dataframe(comparison_df, use_container_width=True)
            
            # Vector Operations
            st.markdown('<h2 class="section-header">🧮 Vector Operations</h2>', unsafe_allow_html=True)
            
            st.markdown("""
            Perform word analogy operations like: **king - man + woman ≈ queen**
            """)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                word_a = st.selectbox("Word A (positive):", available_words, key="word_a")
            with col2:
                word_b = st.selectbox("Word B (negative):", available_words, key="word_b")
            with col3:
                word_c = st.selectbox("Word C (positive):", available_words, key="word_c")
            
            if st.button("🔍 Find Analogy"):
                try:
                    # Skip-gram analogy
                    sg_result = st.session_state.skipgram_model.wv.most_similar(
                        positive=[word_a, word_c], 
                        negative=[word_b], 
                        topn=3
                    )
                    
                    # CBOW analogy
                    cbow_result = st.session_state.cbow_model.wv.most_similar(
                        positive=[word_a, word_c], 
                        negative=[word_b], 
                        topn=3
                    )
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Skip-gram Results")
                        st.write(f"**{word_a} - {word_b} + {word_c} ≈**")
                        for word, sim in sg_result:
                            st.write(f"• {word} ({sim:.3f})")
                    
                    with col2:
                        st.subheader("CBOW Results")
                        st.write(f"**{word_a} - {word_b} + {word_c} ≈**")
                        for word, sim in cbow_result:
                            st.write(f"• {word} ({sim:.3f})")
                
                except Exception as e:
                    st.error(f"Error performing analogy: {str(e)}")
        
        # Educational Content
        st.markdown('<h2 class="section-header">📚 Understanding Word2Vec</h2>', unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["Skip-gram", "CBOW", "Comparison"])
        
        with tab1:
            st.markdown("""
            ### Skip-gram Architecture
            
            **Skip-gram** predicts context words given a center word.
            
            **How it works:**
            - Takes a center word as input
            - Predicts surrounding words within a window
            - Good for rare words and capturing detailed relationships
            
            **Training Process:**
            1. For each word in the corpus, use it as center word
            2. Predict each context word within the window
            3. Use negative sampling to make training efficient
            
            **Advantages:**
            - Better performance on rare words
            - Captures more nuanced semantic relationships
            - Generally better for larger datasets
            """)
        
        with tab2:
            st.markdown("""
            ### CBOW (Continuous Bag of Words)
            
            **CBOW** predicts a center word given its context words.
            
            **How it works:**
            - Takes context words as input (averaged)
            - Predicts the center word
            - Faster to train than Skip-gram
            
            **Training Process:**
            1. For each word, collect its context words within window
            2. Average the context word embeddings
            3. Predict the center word from this average
            
            **Advantages:**
            - Faster training
            - Better performance on frequent words
            - More stable gradients due to averaging
            """)
        
        with tab3:
            comparison_table = pd.DataFrame({
                'Aspect': [
                    'Training Speed', 'Rare Word Performance', 'Frequent Word Performance',
                    'Memory Usage', 'Semantic Accuracy', 'Syntactic Accuracy'
                ],
                'Skip-gram': [
                    'Slower', 'Better', 'Good', 
                    'Higher', 'Better', 'Good'
                ],
                'CBOW': [
                    'Faster', 'Good', 'Better',
                    'Lower', 'Good', 'Better'
                ]
            })
            
            st.markdown("""
            ### Skip-gram vs CBOW Comparison
            
            Both architectures learn dense word representations but with different approaches:
            """)
            
            st.dataframe(comparison_table, use_container_width=True)
            
            st.markdown("""
            **When to use Skip-gram:**
            - Large datasets
            - Focus on rare words
            - Need detailed semantic relationships
            
            **When to use CBOW:**
            - Smaller datasets
            - Focus on frequent words
            - Need faster training
            - Syntactic tasks
            """)

if __name__ == "__main__":
    main()