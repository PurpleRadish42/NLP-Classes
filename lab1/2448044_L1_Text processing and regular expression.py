# %%
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
import re


paragraph1 = """
Parsing means taking an input and producing some sort of linguistic structure for it. 
We will use the term parsing very broadly throughout this book, including many kinds 
of structures that might be produced; morphological, syntactic, semantic, discourse; in
the form of a string, or a tree, or a network. Morphological parsing or stemming applies 
to many affixes other than plurals; for example we might need to take any English verb
form ending in-ing (going, talking, congratulating) and parse it into its verbal stem
plus the-ing morpheme.
"""

# Tokenize and clean words
words = word_tokenize(paragraph1.lower())
words = [word for word in words if word.isalpha()]

# Total and unique words
total_words = len(words)
unique_words = len(set(words))
print("Total words:", total_words)
print("Unique words:", unique_words)

# Word frequencies
freq = Counter(words)
print("\nWord Frequencies:")
print(freq)

# Most frequent word
most_freq = freq.most_common(1)[0]
print("\nMost frequent word:", most_freq)

# Least frequent word
least_freq = freq.most_common()[-1]
print("Least frequent word:", least_freq)

# Longest word
longest_word = max(words, key=len)
print("Longest word:", longest_word)

# %%
paragraph2 = """
Lorem ipsum dolor sit amet. Et quia voluptas et deleniti delectus ea obcaecati perferendis et veniam eveniet. Ea vero unde rem internos impedit et dicta fuga ut dolorem error et facere eius eos laboriosam vero. Ex debitis provident id repudiandae pariatur eos quia dolor vel dolore voluptatum. Ad Quis quas non dolores dolorem aut possimus cupiditate rem cumque ipsum ut sint voluptate aut dolores similique.
"""

# Tokenize and clean words
words = word_tokenize(paragraph2.lower())
words = [word for word in words if word.isalpha()]

# Total and unique words
total_words = len(words)
unique_words = len(set(words))
print("Total words:", total_words)
print("Unique words:", unique_words)

# Word frequencies
freq = Counter(words)
print("\nWord Frequencies:")
print(freq)

# Most frequent word
most_freq = freq.most_common(1)[0]
print("\nMost frequent word:", most_freq)

# Least frequent word
least_freq = freq.most_common()[-1]
print("Least frequent word:", least_freq)

# Longest word
longest_word = max(words, key=len)
print("Longest word:", longest_word)

# %%

print("2.1 (a) - All alphabetic strings")
regex_a = r'\b[a-zA-Z]+\b'
text_a = "Hello123 this is NLP_lab@2025"
print("Matches:", re.findall(regex_a, text_a))
print()

print("2.1 (b) - Lowercase strings ending in 'b'")
regex_b = r'\b[a-z]*b\b'
text_b = "cab grab slab Crib"
print("Matches:", re.findall(regex_b, text_b))
print()

print("2.1 (c) - Two consecutive repeated words")
regex_c = r'\b(\w+)\s+\1\b'
text_c = "He said the the word twice, but not like bug bugged him."
print("Matches:", re.findall(regex_c, text_c))
print()

print("2.1 (d) - Each 'a' is immediately preceded and followed by 'b'")
regex_d = r'^(b|bab)*$'
test_strings_d = ["babbbbab", "baab", "bbababb", "bab"]
for s in test_strings_d:
    print(f"'{s}':", bool(re.fullmatch(regex_d, s)))
print()

print("2.1 (e) - Line starts with integer and ends with a word")
regex_e = r'^\d+\b.*\b[a-zA-Z]+$'
text_e = "42 the answer is always life"
print("Matches:", bool(re.match(regex_e, text_e)))
print()

print("2.1 (f) - String contains both 'grotto' and 'raven'")
regex_f = r'(?=.*\bgrotto\b)(?=.*\braven\b)'
text_f1 = "The raven flew above the dark grotto at night."
text_f2 = "The grottos were creepy but no raven was seen."
print(f"text_f1: {bool(re.search(regex_f, text_f1))}")
print(f"text_f2: {bool(re.search(regex_f, text_f2))}")
print()

print("2.1 (g) - Capture first word of a sentence")
regex_g = r'^[\"\'(]*([A-Z][a-z]*)'
text_g = '"Hello there, how are you?"'
match_g = re.match(regex_g, text_g)
print("First word:", match_g.group(1) if match_g else "No match")

# %%
text = """
Hello! How's NLP2025 treating you?
dog CAT mouse Mouse fish
Paris is in France and Earth is round.
This test will find four words like done.
He said the the thing was weird, not go go now.
I was singing and running while eating snacks.
This book has a letter and a cool story.
"""

# 2.2 - Regular expressions for word-based patterns

# (a) Match a single alphabetic word
print("2.2 (a): Single alphabetic words")
print(re.findall(r'\b[a-zA-Z]+\b', text), '\n')

# (b) Match only lowercase alphabetic words
print("2.2 (b): Lowercase words")
print(re.findall(r'\b[a-z]+\b', text), '\n')

# (c) Match words starting with a capital letter
print("2.2 (c): Capitalized words")
print(re.findall(r'\b[A-Z][a-z]*\b', text), '\n')

# (d) Match all 4-letter words
print("2.2 (d): Words exactly 4 letters long")
print(re.findall(r'\b[a-zA-Z]{4}\b', text), '\n')

# (e) Match repeated words (like "go go")
print("2.2 (e): Repeated consecutive words")
print(re.findall(r'\b(\w+)\s+\1\b', text), '\n')

# (f) Match words ending in 'ing'
print("2.2 (f): Words ending in 'ing'")
print(re.findall(r'\b\w+ing\b', text), '\n')

# (g) Match words with at least one double letter
print("2.2 (g): Words with double letters")
matches = re.finditer(r'\b\w*(\w)\1\w*\b', text)
print([m.group(0) for m in matches])


