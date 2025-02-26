# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 12:59:10 2025

@author: Sergio Calvo
"""

# --- Monkey-patch langdetect for pyspellchecker ---
try:
    import langdetect
except ImportError:
    pass  # langdetect should already be in your requirements

if not hasattr(langdetect, "_detect_lang"):
    from langdetect import detect as _detect_lang_impl
    langdetect._detect_lang = _detect_lang_impl

# --- Imports ---
import streamlit as st
import nltk
import spacy
from transformers import pipeline
import gensim
from gensim import corpora
from nltk.wsd import lesk
import os
from os import path
from PIL import Image
from annotated_text import annotated_text
import streamlit_extras
import pandas as pd
from streamlit_extras.buy_me_a_coffee import button
from spellchecker import SpellChecker
import streamlit.components.v1 as components

# --- Set page configuration ---
st.set_page_config(
    page_title="NLP Buddy",
    page_icon="img//NLPBuddy_icon.ico",
)

# Google Analytics
GA_ID = "G-2SV7DZ5WMM"  # Reemplázalo con tu ID de Google Analytics
GA_SCRIPT = f"""
    <!-- Google tag (gtag.js) -->
    <script async src="https://www.googletagmanager.com/gtag/js?id={GA_ID}"></script>
    <script>
        window.dataLayer = window.dataLayer || [];
        function gtag(){{dataLayer.push(arguments);}}
        gtag('js', new Date());

        gtag('config', '{GA_ID}');
    </script>
"""

components.html(GA_SCRIPT, height=0, scrolling=False)

# --- Load custom CSS if available ---
if os.path.exists("styles.css"):
    with open("styles.css") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# --- Determine working directory ---
d = path.dirname(__file__) if "__file__" in locals() else os.getcwd()

# --- SVG Logo ---
svg_logo = """
<svg width="400" height="110" viewBox="50 50 400 120" xmlns="http://www.w3.org/2000/svg">
  <!-- Background -->
  <rect width="100%" height="100%" fill="none" />
  <!-- Speech Bubble -->
  <g transform="translate(50,50)">
    <path d="M0 0 h140 a10 10 0 0 1 10 10 v60 a10 10 0 0 1 -10 10 h-90 l-12 12 l4 -12 a10 10 0 0 0 -4 -2 h-38 a10 10 0 0 1 -10 -10 v-60 a10 10 0 0 1 10 -10 z" fill="#34495E"/>
    <!-- Line connecting nodes (light gray) -->
    <line x1="45" y1="30" x2="95" y2="45" stroke="#BDC3C7" stroke-width="3"/>
    <!-- NLP Nodes -->
    <circle cx="95" cy="45" r="6" fill="#1ABC9C"/>
    <circle cx="45" cy="30" r="6" fill="#d55e00"/>
  </g>
  <!-- Text "LocNLP" -->
  <text x="210" y="85" font-family="Montserrat, sans-serif" font-weight="bold" font-size="35" fill="#2ac1b5">Loc</text>
  <text x="271" y="85" font-family="Montserrat, sans-serif" font-weight="bold" font-size="35" fill="#34495E">NLP</text>
  <!-- Text "Lab23" -->
  <text x="210" y="125" font-family="Montserrat, sans-serif" font-size="38" font-weight="bold" fill="#34495E">Lab</text>
  <text x="277" y="125" font-family="Montserrat, sans-serif" font-size="32" font-weight="bold" fill="#d55e00">23</text>
  <!-- Slogan -->
  <text x="40" y="162" font-family="Akronim, sans-serif" font-size="22" fill="#34495E">LANGUAGE AUTOMATION</text>
</svg>
"""

# Save the SVG to a file
with open("logo.svg", "w") as f:
    f.write(svg_logo)

# Display the SVG logo in the sidebar
st.sidebar.image("logo.svg", width=150)

# --- Download necessary NLTK data ---
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')

# --- Load spaCy model for English ---
nlp = spacy.load("en_core_web_sm")

# --- Lazy-loading of Hugging Face Pipelines using caching ---
@st.cache_resource
def get_sentiment_classifier():
    return pipeline("sentiment-analysis")

@st.cache_resource
def get_translation_pipeline():
    return pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")

@st.cache_resource
def get_summarizer():
    return pipeline("summarization")

@st.cache_resource
def get_text_generator():
    return pipeline("text-generation", model="gpt2")

@st.cache_resource
def get_zero_shot_classifier():
    return pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        revision="d7645e1"
    )

# --- Initialize SpellChecker ---
spell = SpellChecker()

###############################################################################
# Sidebar Navigation
###############################################################################
st.sidebar.title("github/LearnNLP/img/NLPBuddy_icon.ico NLP Buddy")
sections = [
    "🚀 Introduction",
    "⚙️ Environment Setup",
    "🔠 Tokenization",
    "🌿 Stemming",
    "📝 Lemmatization",
    "🏷️ POS Tagging",
    "🧇 Noun Phrases Chunking",
    "🔎 Named Entity Recognition (NER)",
    "😊 Sentiment Analysis",
    "🧠 Topic Modeling",
    "❓ Word Sense Disambiguation",
    "🌐 Language Translation",
    "🗂️ Text Classification",
    "📰 Information Extraction",
    "✂️ Text Summarization",
    "🤖 Text Generation",
    "✏️ Spelling Correction",
    "🎯 Conclusion",
    "☕​ Buy Me a Coffee"
]
choice = st.sidebar.radio("Go to Section:", sections)

###############################################################################
# 1. Introduction
###############################################################################
if choice == "🚀 Introduction":
    st.image("logo.svg")
    st.title("NLP Buddy - Learn NLP")
    st.markdown(
        """
        This interactive guide follows the article that I wrote, [Common NLP Tasks – Quick Learning](https://www.veriloquium.com/learning-nlp-quick-overview-with-sample-codes/), covering a range of fundamental NLP tasks with working sample codes. You will find some kind of magic here from the world of computational linguistics.
        """
    )
    annotated_text(
        ("Natural Language Processing (NLP)", "Love this! ❤"),
        " is a field of ",
        ("Artificial Intelligence", "Too! ❤"),
        " focused on enabling ",
        ("computers", "noun"),
        " to understand and process ",
        ("human language", "Really??"),
        ".",
        "\nI'm sharing examples of how, using Python and various libraries, we can unveil the secrets of language. This app is built using **Streamlit**, designed specifically for ",
        ("Machine Learning", "term"),
        " and ",
        ("Data Science", "term"),
        " projects."
    )
    st.markdown(
        """              
        Behind the scenes, the code leverages libraries such as `Spacy`, `NLTK`, `Transformers`, `Pandas`, `scikit-learn`, `Beautiful Soup`, `TextBlob`, `Gensim`, `PyTorch`, and many more.
        \n**The main NLP applications demonstrated in this app include:**
        """
    )
    nlp_tasks = [
        "Tokenization", "Stemming", "Lemmatization", "Part-of-Speech Tagging",
        "Noun Chunking", "Named Entity Recognition (NER)", "Sentiment Analysis",
        "Topic Modeling", "Word Sense Disambiguation", "Language Translation",
        "Text Classification", "Information Extraction", "Text Summarization",
        "Text Generation", "Spelling Correction"
    ]
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.title(":rainbow: NLP Apps with Sample Codes")
        with col2:
            st.markdown(
                """
                `Tokenization`, `Stemming`, `Lemmatization`, `Part-of-Speech Tagging`, `Noun Chunking`, 
                `Named Entity Recognition (NER)`, `Sentiment Analysis`, `Topic Modeling`, 
                `Word Sense Disambiguation`, `Language Translation`, `Text Classification`, 
                `Information Extraction`, `Text Summarization`, `Text Generation`, `Spelling Correction`
                """
            )
    st.markdown(
        """
        \n### Want to learn more?
        \n**👈 Select a demo from the sidebar** to see some interactive NLP examples!
        """
    )
    with st.popover("About Sergio"):
        st.markdown(
            """
            <div style="text-align: center;">
            <h3>About the Author</h3>
            <p>
            🙋‍♂️ <strong>Sergio Calvo</strong>
            </p>
            <p>
            🌐 Translator, Reviewer, Computational Linguist, Terminologist, and Localization Engineer with 20+ years of experience in translation, localization, and NLP.
            </p>
            <p>
            💬 Passionate about diving deep into the intricacies of language – whether human or computer – to unveil the beauty of communication.
            </p>
            <p>
            <a href="https://www.veriloquium.com" target="_blank">
            🔗 www.veriloquium.com
            </a>
            </p>
            </div>
            """,
            unsafe_allow_html=True
        )

###############################################################################
# 2. Environment Setup
###############################################################################
elif choice == "⚙️ Environment Setup":
    st.header("⚙️ Environment Setup")
    st.write(
        """
        Before running any NLP code, ensure your environment is properly set up.  
        Install the required libraries using:
        """
    )
    st.code(
        """
# (Optional) Create and activate a virtual environment:
# python -m venv venv
# source venv/bin/activate    # on Linux/Mac
# venv\\Scripts\\activate     # on Windows

pip install streamlit spacy nltk transformers gensim pyspellchecker

# Download spaCy's English model:
python -m spacy download en_core_web_sm

# Download necessary NLTK data:
python -m nltk.downloader punkt wordnet omw-1.4
        """,
        language="bash",
    )
    st.write("Then run the app with:")
    st.code("streamlit run app.py", language="bash")

###############################################################################
# 3. Tokenization
###############################################################################
elif choice == "🔠 Tokenization":
    st.header("🔠 Tokenization")
    st.write(
        """
        Tokenization is the process of breaking text into smaller units (tokens) such as words or punctuation.  
        Below is an interactive example using **NLTK**.
        """
    )
    user_text = st.text_area("Enter text to tokenize", "Hello world! This is an example of tokenization.")
    if st.button("Tokenize"):
        from nltk.tokenize import word_tokenize
        tokens = word_tokenize(user_text)
        st.write("**Tokens:**", tokens)
    st.subheader("Sample Code")
    st.code(
        """
import nltk
from nltk.tokenize import word_tokenize

text = "Hello world! This is an example of tokenization."
tokens = word_tokenize(text)
print(tokens)
        """,
        language="python",
    )

###############################################################################
# 4. Stemming
###############################################################################
elif choice == "🌿 Stemming":
    st.header("🌿 Stemming")
    st.write(
        """
        Stemming reduces words to their root form by stripping suffixes.  
        Here we use **NLTK's PorterStemmer**.
        """
    )
    user_text = st.text_area("Enter text for stemming", "Running runners ran rapidly.")
    if st.button("Stem"):
        from nltk.stem import PorterStemmer
        stemmer = PorterStemmer()
        tokens = nltk.word_tokenize(user_text)
        stems = [stemmer.stem(token) for token in tokens]
        st.write("**Stemmed Tokens:**", stems)
    st.subheader("Sample Code")
    st.code(
        """
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

text = "Running runners ran rapidly."
tokens = word_tokenize(text)
stemmer = PorterStemmer()
stems = [stemmer.stem(token) for token in tokens]
print(stems)
        """,
        language="python",
    )

###############################################################################
# 5. Lemmatization
###############################################################################
elif choice == "📝 Lemmatization":
    st.header("📝 Lemmatization")
    st.write(
        """
        Lemmatization reduces words to their base or dictionary form (lemma).  
        Here we use **spaCy** to perform lemmatization.
        """
    )
    user_text = st.text_area("Enter text for lemmatization", "The striped bats are hanging on their feet for best.")
    if st.button("Lemmatize"):
        doc = nlp(user_text)
        lemmas = [(token.text, token.lemma_) for token in doc]
        st.write("**Lemmatized Tokens:**", lemmas)
    st.subheader("Sample Code")
    st.code(
        """
import spacy

nlp = spacy.load("en_core_web_sm")
text = "The striped bats are hanging on their feet for best."
doc = nlp(text)
for token in doc:
    print(token.text, token.lemma_)
        """,
        language="python",
    )

###############################################################################
# 6. POS Tagging
###############################################################################
elif choice == "🏷️ POS Tagging":
    st.header("🏷️ Part-of-Speech (POS) Tagging")
    st.write(
        """
        POS Tagging assigns grammatical labels (e.g., noun, verb, adjective) to each token in the text.  
        Let's see how **spaCy** handles POS tagging.
        """
    )
    user_text = st.text_area("Enter text for POS tagging", "Natural Language Processing is fascinating.")
    if st.button("Analyze POS"):
        doc = nlp(user_text)
        pos_tags = [(token.text, token.pos_) for token in doc]
        st.write("**POS Tags:**", pos_tags)
    st.subheader("Sample Code")
    st.code(
        """
import spacy

nlp = spacy.load("en_core_web_sm")
text = "Natural Language Processing is fascinating."
doc = nlp(text)
for token in doc:
    print(token.text, token.pos_)
        """,
        language="python",
    )

###############################################################################
# 7. Noun Phrases Chunking
###############################################################################
elif choice == "🧇 Noun Phrases Chunking":
    st.header("🧇 Noun Phrases Chunking")
    st.write(
        """
        Noun chunking groups tokens into meaningful phrases.  
        Here, we extract noun chunks using **spaCy**.
        """
    )
    user_text = st.text_area("Enter text for noun chunking", "The quick brown fox jumps over the lazy dog.")
    if st.button("Extract Noun Chunks"):
        doc = nlp(user_text)
        chunks = [(chunk.text, chunk.root.dep_, chunk.root.head.text) for chunk in doc.noun_chunks]
        st.write("**Noun Chunks (Text, Dependency, Head):**", chunks)
    st.subheader("Sample Code")
    st.code(
        """
import spacy

nlp = spacy.load("en_core_web_sm")
text = "The quick brown fox jumps over the lazy dog."
doc = nlp(text)
for chunk in doc.noun_chunks:
    print(chunk.text, chunk.root.dep_, chunk.root.head.text)
        """,
        language="python",
    )

###############################################################################
# 8. Named Entity Recognition (NER)
###############################################################################
elif choice == "🔎 Named Entity Recognition (NER)":
    st.header("🔎 Named Entity Recognition (NER)")
    st.write(
        """
        NER detects and labels entities (e.g., people, organizations, locations) in text.  
        We'll demonstrate this using **spaCy**.
        """
    )
    user_text = st.text_area("Enter text for NER", "Bill Gates founded Microsoft in Albuquerque.")
    if st.button("Extract Entities"):
        doc = nlp(user_text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        st.write("**Named Entities:**", entities)
    st.subheader("Sample Code")
    st.code(
        """
import spacy

nlp = spacy.load("en_core_web_sm")
text = "Bill Gates founded Microsoft in Albuquerque."
doc = nlp(text)
for ent in doc.ents:
    print(ent.text, ent.label_)
        """,
        language="python",
    )

###############################################################################
# 9. Sentiment Analysis
###############################################################################
elif choice == "😊 Sentiment Analysis":
    st.header("😊 Sentiment Analysis")
    st.write(
        """
        Sentiment analysis determines whether a text expresses a positive, negative, or neutral sentiment.  
        This example uses a **Hugging Face Transformers** pipeline.
        """
    )
    user_text = st.text_area("Enter text for sentiment analysis", "I absolutely love learning NLP!")
    if st.button("Analyze Sentiment"):
        classifier = get_sentiment_classifier()
        result = classifier(user_text)
        st.write("**Sentiment Analysis Result:**", result)
    st.subheader("Sample Code")
    st.code(
        """
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
text = "I absolutely love learning NLP!"
result = classifier(text)
print(result)
        """,
        language="python",
    )

###############################################################################
# 10. Topic Modeling
###############################################################################
elif choice == "🧠 Topic Modeling":
    st.header("🧠 Topic Modeling")
    st.write(
        """
        Topic modeling discovers abstract topics within a collection of documents.  
        Below is an example using **Gensim's LDA** model. Provide multiple documents separated by newlines.
        """
    )
    user_text = st.text_area(
        "Enter documents for topic modeling (each document on a new line)", 
        "Document one text goes here.\nDocument two text goes here."
    )
    if st.button("Generate Topics"):
        docs = [doc.strip() for doc in user_text.split('\n') if doc.strip() != '']
        texts = [nltk.word_tokenize(doc.lower()) for doc in docs]
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
        lda_model = gensim.models.LdaModel(corpus, num_topics=2, id2word=dictionary, passes=10)
        topics = lda_model.print_topics(num_words=5)
        st.write("**Discovered Topics:**", topics)
    st.subheader("Sample Code")
    st.code(
        """
import nltk
import gensim
from gensim import corpora

docs = [
    "Document one text goes here.",
    "Document two text goes here."
]
texts = [nltk.word_tokenize(doc.lower()) for doc in docs]
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
lda_model = gensim.models.LdaModel(corpus, num_topics=2, id2word=dictionary, passes=10)
topics = lda_model.print_topics(num_words=5)
print(topics)
        """,
        language="python",
    )

###############################################################################
# 11. Word Sense Disambiguation
###############################################################################
elif choice == "❓ Word Sense Disambiguation":
    st.header("❓ Word Sense Disambiguation")
    st.write(
        """
        Word Sense Disambiguation (WSD) identifies the correct meaning of a word in context.  
        This example uses **NLTK's Lesk algorithm**.
        """
    )
    user_text = st.text_area("Enter a sentence", "I looked through the telescope to observe the stars.")
    target_word = st.text_input("Enter the target word for disambiguation", "telescope")
    if st.button("Disambiguate"):
        tokens = nltk.word_tokenize(user_text)
        sense = lesk(tokens, target_word)
        if sense:
            st.write(f"**Sense for '{target_word}':**")
            st.write("Synset:", sense.name())
            st.write("Definition:", sense.definition())
            st.write("Examples:", sense.examples())
        else:
            st.write("No sense found.")
    st.subheader("Sample Code")
    st.code(
        """
import nltk
from nltk.wsd import lesk
from nltk.tokenize import word_tokenize

text = "I looked through the telescope to observe the stars."
tokens = word_tokenize(text)
sense = lesk(tokens, "telescope")
if sense:
    print("Synset:", sense.name())
    print("Definition:", sense.definition())
    print("Examples:", sense.examples())
else:
    print("No sense found.")
        """,
        language="python",
    )

###############################################################################
# 12. Language Translation
###############################################################################
elif choice == "🌐 Language Translation":
    st.header("🌐 Language Translation")
    st.write(
        """
        This example uses a **Hugging Face Transformers** pipeline to translate text from one language to another.
        """
    )
    languages = {
        "English": "en",
        "French": "fr",
        "German": "de",
        "Spanish": "es",
        "Italian": "it",
        "Dutch": "nl"
    }
    source_lang_name = st.selectbox("Select Source Language", list(languages.keys()), index=0)
    target_lang_name = st.selectbox("Select Target Language", list(languages.keys()), index=1)
    user_text = st.text_area("Enter text to translate", "Hello, how are you?")
    if st.button("Translate"):
        src = languages[source_lang_name]
        tgt = languages[target_lang_name]
        translator = pipeline(
            "translation_{}_to_{}".format(src, tgt),
            model=f"Helsinki-NLP/opus-mt-{src}-{tgt}"
        )
        translated = translator(user_text)
        st.write("**Translated Text:**", translated[0]['translation_text'])
    st.subheader("Sample Code")
    st.code(
        """
from transformers import pipeline

source_lang = "en"  # e.g. English
target_lang = "fr"  # e.g. French
translator = pipeline("translation_{}_to_{}".format(source_lang, target_lang),
                      model=f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}")
text = "Hello, how are you?"
translated = translator(text)
print(translated[0]['translation_text'])
        """,
        language="python",
    )

###############################################################################
# 13. Text Classification
###############################################################################
elif choice == "🗂️ Text Classification":
    st.header("🗂️ Text Classification")
    st.write(
        """
        This example uses **Hugging Face's Zero-Shot Classification** to classify text into user-defined categories.
        """
    )
    user_text = st.text_area("Enter text for classification", "The movie was thrilling and suspenseful.")
    candidate_labels_input = st.text_input("Enter candidate labels (comma-separated)", "positive, negative, neutral")
    if st.button("Classify Text"):
        candidate_labels = [label.strip() for label in candidate_labels_input.split(',')]
        classifier = get_zero_shot_classifier()
        result = classifier(user_text, candidate_labels=candidate_labels)
        st.write("**Classification Result:**", result)
    st.subheader("Sample Code")
    st.code(
        """
from transformers import pipeline

classifier = pipeline("zero-shot-classification")
text = "The movie was thrilling and suspenseful."
candidate_labels = ["positive", "negative", "neutral"]
result = classifier(text, candidate_labels=candidate_labels)
print(result)
        """,
        language="python",
    )

###############################################################################
# 14. Information Extraction
###############################################################################
elif choice == "📰 Information Extraction":
    st.header("📰 Information Extraction")
    st.write(
        """
        Information Extraction involves extracting structured information from unstructured text.  
        Below is a simple example combining **Named Entity Recognition (NER)** and noun chunk extraction using **spaCy**.
        """
    )
    user_text = st.text_area("Enter text for information extraction", "Apple Inc. was founded by Steve Jobs in Cupertino.")
    if st.button("Extract Information"):
        doc = nlp(user_text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        noun_chunks = [chunk.text for chunk in doc.noun_chunks]
        st.write("**Named Entities:**", entities)
        st.write("**Noun Chunks:**", noun_chunks)
    st.subheader("Sample Code")
    st.code(
        """
import spacy

nlp = spacy.load("en_core_web_sm")
text = "Apple Inc. was founded by Steve Jobs in Cupertino."
doc = nlp(text)
entities = [(ent.text, ent.label_) for ent in doc.ents]
noun_chunks = [chunk.text for chunk in doc.noun_chunks]
print("Entities:", entities)
print("Noun Chunks:", noun_chunks)
        """,
        language="python",
    )

###############################################################################
# 15. Text Summarization
###############################################################################
elif choice == "✂️ Text Summarization":
    st.header("✂️ Text Summarization")
    st.write(
        """
        Text Summarization condenses a piece of text into a shorter version while retaining its main ideas.  
        This example uses a **Hugging Face Transformers** pipeline.
        """
    )
    user_text = st.text_area("Enter text to summarize", "Long text goes here. " * 20)
    if st.button("Summarize"):
        summarizer_pipeline = get_summarizer()
        summary = summarizer_pipeline(user_text, max_length=130, min_length=30, do_sample=False)
        st.write("**Summary:**", summary[0]['summary_text'])
    st.subheader("Sample Code")
    st.code(
        """
from transformers import pipeline

summarizer = pipeline("summarization")
text = "Long text goes here. " * 20
summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
print(summary[0]['summary_text'])
        """,
        language="python",
    )

###############################################################################
# 16. Text Generation
###############################################################################
elif choice == "🤖 Text Generation":
    st.header("🤖 Text Generation")
    st.write(
        """
        Text Generation creates new text based on a given prompt.  
        This example uses **Hugging Face Transformers** with the GPT-2 model.
        """
    )
    user_text = st.text_area("Enter a text prompt for generation", "Once upon a time")
    if st.button("Generate Text"):
        generator = get_text_generator()
        generated = generator(user_text, max_length=51, num_return_sequences=1)
        st.write("**Generated Text:**", generated[0]['generated_text'])
    st.subheader("Sample Code")
    st.code(
        """
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")
text = "Once upon a time"
generated = generator(text, max_length=51, num_return_sequences=1)
print(generated[0]['generated_text'])
        """,
        language="python",
    )

###############################################################################
# 17. Spelling Correction
###############################################################################
elif choice == "✏️ Spelling Correction":
    st.header("✏️ Spelling Correction")
    st.write(
        """
        Spelling Correction identifies and corrects misspelled words.  
        This example uses the **pyspellchecker** library.
        """
    )
    user_text = st.text_area("Enter text for spelling correction", "Ths is a smple txt with speling erors.")
    if st.button("Correct Spelling"):
        words = nltk.word_tokenize(user_text)
        unknown_words = spell.unknown(words)
        corrections = {word: spell.correction(word) for word in unknown_words}
        st.write("**Spelling Corrections:**", corrections)
    st.subheader("Sample Code")
    st.code(
        """
from nltk.tokenize import word_tokenize
from spellchecker import SpellChecker

spell = SpellChecker()
text = "Ths is a smple txt with speling erors."
words = word_tokenize(text)
unknown_words = spell.unknown(words)
corrections = {word: spell.correction(word) for word in unknown_words}
print(corrections)
        """,
        language="python",
    )

###############################################################################
# 18. Conclusion
###############################################################################
elif choice == "🎯 Conclusion":
    st.header("🎯 Conclusion")
    st.write(
        """
        Natural Language Processing (NLP) is a vast and exciting field that powers many real-world applications, from chatbots and search engines to sentiment analysis and machine translation.
        
        In this quick overview, we explored essential NLP tasks, including:
        
        🔹 **Fundamentals:**  
           - Tokenization  
           - Lemmatization  
           - Stemming  
           - Part-of-Speech (POS) Tagging  
           - Named Entity Recognition (NER)  
        
        🔹 **Advanced Techniques:**  
           - Text Classification  
           - Sentiment Analysis  
           - Topic Modeling  
           - Word Sense Disambiguation  
        
        🔹 **Applications in AI:**  
           - Information Extraction  
           - Text Summarization  
           - Text Generation  
           - Spelling Correction  
           - Machine Translation
        
        These tasks serve as building blocks for powerful AI-driven language models and applications.
        
        🚀 **Next Steps:**  
        Feel free to experiment with the interactive examples and explore how these techniques can be applied in real-world scenarios. NLP is continuously evolving, and mastering its core concepts will help you unlock its full potential.
        
        🔗 **Continue Learning:**  
        For a deeper dive into NLP, check out the full article: [Learning NLP – Quick Overview with Sample Codes](https://www.veriloquium.com/learning-nlp-quick-overview-with-sample-codes/)
        """
    )
    st.markdown("---")

###############################################################################
# 19. Buy Me a Coffee
###############################################################################
elif choice == "☕​ Buy Me a Coffee":
    st.header("☕ Buy Me a Coffee")
    st.write(
        """
        If you found this NLP overview helpful and would like to support my work, consider **buying me a coffee**!  
        Your support helps me create more useful content, tutorials, and projects for the community.
        
        Every coffee fuels more research, coding, and open-source contributions! 🚀🔥
        """
    )
    coffee_button = """
    <div style="display: flex; justify-content: center;">
        <a href="https://www.buymeacoffee.com/sergiocalvc" target="_blank">
            <img src="https://img.buymeacoffee.com/button-api/?text=Buy me a coffee&emoji=☕&slug=sergiocalvc&button_colour=FFDD00&font_colour=000000&font_family=Arial&outline_colour=000000&coffee_colour=ffffff" alt="Buy Me a Coffee">
        </a>
    </div>
    """
    st.markdown(coffee_button, unsafe_allow_html=True)
    st.markdown("---")
    st.write(
        """
        🔗 **Other ways to support:**  
        - Share this project with your network  
        - Provide feedback and suggestions  
        - Connect with me on [LinkedIn](https://www.linkedin.com/in/sergiocalvopaez)
        """
    )
    st.markdown(
        """
        ### 💛 Your support makes a difference! [![Buy Me a Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-Support%20Me%20☕-yellow)](https://www.buymeacoffee.com/sergiocalvc)
        """
    )

# --- Footer ---
st.markdown(
    """<p style='text-align: center;'> Brought to you with <span style='color:red'>❤</span> by <a href='https://www.veriloquium.com/'>Sergio Calvo</a> | Veriloquium © LocNLP Lab23 </p>""",
    unsafe_allow_html=True
)
