import nltk
from nltk.util import ngrams
from nltk import FreqDist, pos_tag, word_tokenize
from collections import defaultdict
import math
import spacy

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nlp_spacy = spacy.load("en_core_web_sm")

def get_tokens(text):
    return ['<s>'] + word_tokenize(text) + ['</s>']

def unigram_prob(tokens):
    freq = FreqDist(tokens)
    total = len(tokens)
    print("\nUnigram Probabilities:")
    for word in set(tokens):
        print(f"P({word}) = {freq[word]/total:.6f}")

def bigram_prob(tokens, w1, w2):
    bigrams = list(ngrams(tokens, 2))
    bigram_freq = FreqDist(bigrams)
    unigram_freq = FreqDist(tokens)
    prob = bigram_freq[(w1, w2)] / unigram_freq[w1] if unigram_freq[w1] != 0 else 0
    print(f"\nP({w2}|{w1}) = {prob:.6f}")

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

def perplexity(model, tokens, n=2):
    N = len(tokens)
    log_prob = 0
    for i in range(n-1, len(tokens)):
        prefix = tuple(tokens[i-n+1:i])
        word = tokens[i]
        prob = model[prefix].get(word, 1e-6)
        log_prob += math.log(prob)
    ppl = math.exp(-log_prob / N)
    print(f"\nPerplexity = {ppl:.2f}")

def pos_tagging(text):
    tokens = word_tokenize(text)
    print("\nPOS Tags:")
    print(pos_tag(tokens))

def ner_spacy(text):
    doc = nlp_spacy(text)
    print("\nNamed Entities:")
    for token in doc:
        print(f"{token.text} -> {token.ent_iob_}-{token.ent_type_ if token.ent_type_ else 'O'}")

def main():
    while True:
        print("\n----- NLP Lab 3 Menu -----")
        print("1. Unigram Probability")
        print("2. Bigram Probability")
        print("3. Train N-gram Model and Perplexity")
        print("4. POS Tagging")
        print("5. Named Entity Recognition (NER)")
        print("6. Exit")
        choice = input("Enter your choice: ")

        if choice == '1':
            text = input("\nEnter text: ")
            tokens = get_tokens(text)
            unigram_prob(tokens)

        elif choice == '2':
            text = input("\nEnter text: ")
            tokens = get_tokens(text)
            w1 = input("Enter first word: ")
            w2 = input("Enter second word: ")
            bigram_prob(tokens, w1, w2)

        elif choice == '3':
            text = input("\nEnter text: ")
            tokens = get_tokens(text)
            model = train_ngram_model(tokens, n=2)
            test = get_tokens(input("Enter test sentence for perplexity: "))
            perplexity(model, test, n=2)

        elif choice == '4':
            text = input("\nEnter text: ")
            pos_tagging(text)

        elif choice == '5':
            text = input("\nEnter text: ")
            ner_spacy(text)

        elif choice == '6':
            break
        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    main()
