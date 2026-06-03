import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

# Download required datasets
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# ==========================================
# 0. RAW DATA
# ==========================================

reviews = [
    "The athlete is running fast.",
    "The athlete ran yesterday.",
    "Running improves health."
]

print("\n--- 0. RAW DATA ---")

for i, review in enumerate(reviews, 1):
    print(f"Sentence {i}: {review}")

print("\n" + "="*50)

# ==========================================
# 1. TOKENIZATION
# ==========================================

tokenized_reviews = [
    word_tokenize(review.lower())
    for review in reviews
]

print("\n--- 1. TOKENIZATION ---")

for i, tokens in enumerate(tokenized_reviews, 1):
    print(f"Sentence {i}: {tokens}")

print("\n" + "="*50)

# ==========================================
# 2. STEMMING VS LEMMATIZATION
# ==========================================

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

stemmed_reviews = []
lemmatized_reviews = []

for tokens in tokenized_reviews:

    stemmed_reviews.append(
        [stemmer.stem(word) for word in tokens]
    )

    lemmatized_reviews.append(
        [lemmatizer.lemmatize(word, pos='v')
         for word in tokens]
    )

print("\n--- 2. STEMMING VS LEMMATIZATION ---")

print("Original:")
print(tokenized_reviews)

print("\nStemmed:")
print(stemmed_reviews)

print("\nLemmatized:")
print(lemmatized_reviews)

print("\nObserve:")
print("running → run")
print("ran → run")

print("\n" + "="*50)

# ==========================================
# 3. UNIQUE WORDS + BAG OF WORDS
# ==========================================

cleaned_sentences = [
    " ".join(tokens)
    for tokens in lemmatized_reviews
]

vectorizer = CountVectorizer()

bag_matrix = vectorizer.fit_transform(
    cleaned_sentences
)

unique_words = (
    vectorizer.get_feature_names_out()
)

print("\n--- 3. UNIQUE VOCABULARY ---")

print(list(unique_words))

print("\n--- BAG OF WORDS MATRIX ---")

print(
    bag_matrix.toarray()
)

print("\nRows → Sentences")
print("Columns → Unique Words")

print("\n" + "="*50)

# ==========================================
# 4. ONE HOT ENCODING
# ==========================================

print("\n--- 4. ONE HOT ENCODING ---")

size = len(unique_words)

for i, word in enumerate(unique_words):

    vector = [0]*size

    vector[i] = 1

    print(
        f"{word} → {vector}"
    )

print("\n" + "="*50)

# ==========================================
# 5. WORD EMBEDDINGS
# ==========================================

print("\n--- 5. WORD EMBEDDINGS ---")

mock_embeddings = {

    "run":[0.92,0.87],

    "running":[0.91,0.86],

    "ran":[0.90,0.88],

    "health":[0.20,0.15]

}

for word, vector in (
    mock_embeddings.items()
):

    print(
        f"{word} → {vector}"
    )

print("\nNotice:")
print("run, running, ran have close values → similar meaning")
print("health is far → different meaning")