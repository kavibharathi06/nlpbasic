import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

# Download necessary NLTK data packages for tokenization and lemmatization
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# ==========================================
# 0. THE RAW DATA (Our Unique Dataset)
# ==========================================
reviews = [
    "The acting was amazing!",
    "The actors were acting terribly.",
    "I loved the amazing acting."
]

print("--- 0. RAW DATA ---")
for i, review in enumerate(reviews, 1):
    print(f"Review {i}: '{review}'")
print("\n" + "="*50 + "\n")

# ==========================================
# 1. TOKENIZATION
# ==========================================
# We split the sentences into individual lowercased word tokens.
tokenized_reviews = [word_tokenize(review.lower()) for review in reviews]

print("--- 1. TOKENIZATION ---")
for i, tokens in enumerate(tokenized_reviews, 1):
    print(f"Review {i} Tokens: {tokens}")
print("\n" + "="*50 + "\n")

# ==========================================
# 2. STEMMING vs LEMMATIZATION
# ==========================================
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

stemmed_reviews = []
lemmatized_reviews = []

for tokens in tokenized_reviews:
    # Stem every word in the token list
    stemmed_reviews.append([stemmer.stem(word) for word in tokens])
    # Lemmatize every word (telling it they are verbs 'v' helps it handle tenses)
    lemmatized_reviews.append([lemmatizer.lemmatize(word, pos='v') for word in tokens])

print("--- 2. CLEANING COMPARISON ---")
print(f"Original:   {tokenized_reviews[1]}")
print(f"Stemmed:    {stemmed_reviews[1]}  <-- Notice 'terribli' (chopped/not a real word)")
print(f"Lemmatized: {lemmatized_reviews[1]} <-- Notice 'were' -> 'be' and 'acting' -> 'act'")
print("\n" + "="*50 + "\n")

# ==========================================
# 3. UNIQUE WORDS & BAG OF WORDS
# ==========================================
# Convert our lemmatized token lists back into strings so Scikit-Learn can read them
cleaned_sentences = [" ".join(tokens) for tokens in lemmatized_reviews]

# CountVectorizer automatically builds our "Bag of Unique Words"
vectorizer = CountVectorizer()
bag_of_words_matrix = vectorizer.fit_transform(cleaned_sentences)

# Get our Unique Vocabulary List
unique_vocabulary = vectorizer.get_feature_names_out()

print("--- 3. UNIQUE VOCABULARY ---")
print(f"Our Global Vocabulary List: {list(unique_vocabulary)}")
print("\n--- BAG OF WORDS MATRIX ---")
print(bag_of_words_matrix.toarray()) 
print("*(Each row represents a review, each column is the count of a unique word)*")
print("\n" + "="*50 + "\n")

# ==========================================
# 4. ONE-HOT ENCODING (Manual Representation)
# ==========================================
print("--- 4. ONE-HOT ENCODING ---")
# To show you what one-hot looks like, let's look at our unique vocabulary.
# Every word gets an isolated slot of 1s and 0s based on its index.
vocab_size = len(unique_vocabulary)

for index, word in enumerate(unique_vocabulary):
    one_hot_vector = [0] * vocab_size
    one_hot_vector[index] = 1
    print(f"'{word}': {one_hot_vector}")
print("\n" + "="*50 + "\n")

# ==========================================
# 5. CONCEPTUAL WORD EMBEDDINGS
# ==========================================
print("--- 5. WORD EMBEDDINGS (Simulated) ---")
# Real Deep Learning models generate continuous floating-point vectors (e.g., 300 dimensions).
# Here is a mock-up of how a deep learning model groups meanings in 2D space:
mock_embeddings = {
    "love": [0.89, 0.12],   # Positive emotion coordinates
    "amaze": [0.85, 0.15],  # Similar to love! Close numbers.
    "terribly": [-0.78, -0.65], # Negative emotion coordinates. Way far away!
    "the": [0.01, 0.02]     # Neutral filler word near the center.
}

for word, vector in mock_embeddings.items():
    print(f"Embedding Vector for '{word}': {vector}")
print("\nNotice how 'love' and 'amaze' have very similar math values!")