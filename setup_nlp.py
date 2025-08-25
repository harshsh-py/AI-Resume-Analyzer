import nltk
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# You may also want to install a spaCy model:
#   python -m spacy download en_core_web_sm
print("NLTK stopwords ready. For spaCy, run: python -m spacy download en_core_web_sm")
