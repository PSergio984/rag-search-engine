import argparse
import json
import math
import os
import pickle
import string
from collections import Counter, defaultdict
from pathlib import Path

from nltk.stem import PorterStemmer


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
STOPWORDS_PATH = DATA_DIR / "stopwords.txt"
DEFAULT_SEARCH_LIMIT = 5


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_movies() -> list[dict]:
    """Load the full list of movies from the JSON dataset."""
    with (DATA_DIR / "movies.json").open("r", encoding="utf-8") as f:
        return json.load(f).get("movies", [])


# ---------------------------------------------------------------------------
# Text preprocessing helpers
# ---------------------------------------------------------------------------

def preprocess_text(text: str) -> str:
    """Lowercase and strip all punctuation from text."""
    text = text.lower()
    # Build a translation table that maps every punctuation char to None (delete it)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def load_stopwords() -> list[str]:
    """Read stopwords.txt and preprocess each word the same way as query/title text."""
    with STOPWORDS_PATH.open("r") as f:
        # Apply preprocess_text so contractions like "aren't" become "arent"
        return [preprocess_text(word) for word in f.read().splitlines()]


# Load and preprocess stop words once at import time for reuse everywhere
STOPWORDS = load_stopwords()


def tokenize_text(text: str) -> list[str]:
    """
    Full preprocessing pipeline for a piece of text:
    1. Lowercase and remove punctuation.
    2. Split on whitespace into tokens.
    3. Drop empty tokens.
    4. Filter out stop words.
    5. Stem each remaining token with PorterStemmer.
    """
    text = preprocess_text(text)
    tokens = text.split()

    # Remove empty tokens and stop words in one pass
    valid_tokens = [t for t in tokens if t and t not in STOPWORDS]

    # Stem each word to its root form (e.g. "running" → "run")
    stemmer = PorterStemmer()
    return [stemmer.stem(t) for t in valid_tokens]


def tokenize_term(term: str) -> str:
    """
    Helper function to preprocess and tokenize a single search term/token.
    
    Flow:
    1. Pass the term string into tokenize_text (the existing tokenizer).
    2. Check the count of returned tokens.
    3. If not exactly 1 token (e.g. 0 or >1 tokens), raise ValueError.
    4. Otherwise, return the single token string.
    """
    # Use existing tokenize_text function to preprocess, split, filter stop words, and stem the term
    tokens = tokenize_text(term)
    
    # If the tokenizer returns anything other than exactly one token, it is invalid for single-term search
    if len(tokens) != 1:
        raise ValueError(f"Term '{term}' must tokenize to exactly one token, but got {tokens}")
        
    # Return the single token at index 0
    return tokens[0]


# ---------------------------------------------------------------------------
# Matching helper
# ---------------------------------------------------------------------------

def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    """Return True if any query token is a substring of any title token."""
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False


# ---------------------------------------------------------------------------
# Inverted Index
# ---------------------------------------------------------------------------

class InvertedIndex:
    def __init__(self) -> None:
        # defaultdict(set) avoids manual "if key not in dict" checks
        self.index: defaultdict = defaultdict(set)
        # Maps document IDs to their full movie dictionary
        self.docmap: dict[int, dict] = {}
        # New dictionary mapping document IDs to Counter objects, representing term frequencies (TF) of tokens
        self.term_frequencies: dict[int, Counter] = {}

    def __add_document(self, doc_id: int, text: str) -> None:
        """Tokenize text and map each unique token to this document ID."""
        tokens = tokenize_text(text)
        # Use set(tokens) to avoid redundant .add() calls for repeated tokens
        for token in set(tokens):
            self.index[token].add(doc_id)

        # Flow to update term frequencies:
        # 1. Initialize a new Counter object for the document ID if it doesn't already exist.
        if doc_id not in self.term_frequencies:
            self.term_frequencies[doc_id] = Counter()
        # 2. Iterate through every token in the document (retaining duplicates) and increment its count in the Counter.
        for token in tokens:
            self.term_frequencies[doc_id][token] += 1

    def get_documents(self, term: str) -> list[int]:
        """Return document IDs for a single preprocessed token, sorted ascending."""
        doc_ids = self.index.get(term, set())
        return sorted(list(doc_ids))

    def get_tf(self, doc_id: int, term: str) -> int:
        """
        Return how many times the token appears in the document with the given ID.
        If the term doesn't appear in that document, return 0.
        Assume term is a single token.

        Flow:
        1. Access the term frequencies Counter for the given document ID.
        2. If the document has no Counter (meaning it was not indexed), return 0.
        3. Access the frequency of the term in the Counter. If the term does not exist, default to 0.
        """
        # Fetch the Counter for the given doc_id, or None if the document ID is not found
        doc_counter = self.term_frequencies.get(doc_id)
        if not doc_counter:
            return 0
        # Fetch the frequency of the token from the Counter, defaulting to 0 if not present
        return doc_counter.get(term, 0)

    def build(self) -> None:
        """Load all movies and index them by their title + description text."""
        movies = load_movies()
        for m in movies:
            doc_id = m["id"]
            # Store the full movie object for retrieval later
            self.docmap[doc_id] = m
            # Combine title and description into one searchable text blob
            text = f"{m['title']} {m['description']}"
            self.__add_document(doc_id, text)

    def save(self) -> None:
        """Serialize the index, docmap, and term frequencies to disk using pickle."""
        # Create the cache directory if it doesn't already exist
        os.makedirs(CACHE_DIR, exist_ok=True)
        
        # Save the inverted index mappings
        with open(CACHE_DIR / "index.pkl", "wb") as f:
            pickle.dump(self.index, f)
            
        # Save the document mapping dictionary
        with open(CACHE_DIR / "docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)
            
        # Save the new term_frequencies dictionary to cache/term_frequencies.pkl
        with open(CACHE_DIR / "term_frequencies.pkl", "wb") as f:
            pickle.dump(self.term_frequencies, f)

    def load(self) -> None:
        """Load the index, docmap, and term frequencies from disk. Raises FileNotFoundError if missing."""
        index_path = CACHE_DIR / "index.pkl"
        docmap_path = CACHE_DIR / "docmap.pkl"
        tf_path = CACHE_DIR / "term_frequencies.pkl"

        # Raise an error early if any cache file is missing
        if not index_path.exists():
            raise FileNotFoundError(f"Index cache not found: {index_path}. Run 'build' first.")
        if not docmap_path.exists():
            raise FileNotFoundError(f"Docmap cache not found: {docmap_path}. Run 'build' first.")
        if not tf_path.exists():
            raise FileNotFoundError(f"Term frequencies cache not found: {tf_path}. Run 'build' first.")

        # Load the inverted index mappings
        with open(index_path, "rb") as f:
            self.index = pickle.load(f)
            
        # Load the document mapping dictionary
        with open(docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
            
        # Load the term frequencies dictionary
        with open(tf_path, "rb") as f:
            self.term_frequencies = pickle.load(f)


# ---------------------------------------------------------------------------
# Command functions (called from the CLI entrypoint)
# ---------------------------------------------------------------------------

def build_command() -> None:
    """Build and save the inverted index to disk."""
    idx = InvertedIndex()
    idx.build()
    idx.save()


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    """Search movies using the inverted index and return up to `limit` matching movie dicts."""
    # Load the pre-built index from disk instead of scanning every movie
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return []

    query_tokens = tokenize_text(query)
    results = []
    # Track seen doc IDs to avoid adding the same movie twice
    seen_ids: set[int] = set()

    # For each query token, look up its matching document IDs directly in the index
    for token in query_tokens:
        for doc_id in idx.get_documents(token):
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                results.append(idx.docmap[doc_id])
                # Stop as soon as we have enough results
                if len(results) >= limit:
                    return results

    return results


def tf_command(doc_id: int, term: str) -> None:
    """
    Look up and print the term frequency (TF) for a given term in a specific document.
    
    Flow:
    1. Instantiate a new InvertedIndex and load its serialized files.
    2. Attempt to tokenize the search term using the tokenize_term helper.
    3. If tokenization raises an exception (e.g. term is empty or multi-word or stop word), print 0 and exit.
    4. Call get_tf with the document ID and preprocessed token.
    5. Print the term frequency count.
    """
    # Create an InvertedIndex instance
    idx = InvertedIndex()
    try:
        # Load the index, docmap, and term frequencies from disk
        idx.load()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    try:
        # Preprocess and validate that the term tokenizes to exactly one token
        token = tokenize_term(term)
    except ValueError:
        # If the term does not resolve to exactly one token, its frequency is 0
        print(0)
        return

    # Look up the term frequency for the single token in the document ID
    tf = idx.get_tf(doc_id, token)
    print(tf)


def idf_command(term: str) -> None:
    """
    Look up and print the inverse document frequency (IDF) for a given term.

    Flow:
    1. Instantiate a new InvertedIndex and load its serialized files.
    2. Attempt to tokenize the search term using the tokenize_term helper.
    3. If tokenization raises an exception, print 0.00 and exit.
    4. Calculate IDF = log((total_doc_count + 1) / (term_match_doc_count + 1)).
    5. Print the IDF value formatted to 2 decimal places.
    """
    # Create an InvertedIndex instance
    idx = InvertedIndex()
    try:
        # Load the index, docmap, and term frequencies from disk
        idx.load()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    try:
        # Preprocess and validate that the term tokenizes to exactly one token
        token = tokenize_term(term)
    except ValueError:
        # If the term does not resolve to exactly one token, its IDF is effectively 0
        print(0.00)
        return

    # Total number of documents in the dataset
    total_docs = len(idx.docmap)
    # Number of documents that contain this token (from the inverted index)
    match_count = len(idx.index.get(token, set()))
    # Calculate IDF using smoothed log formula to avoid division by zero
    idf = math.log((total_docs + 1) / (match_count + 1))
    # Print the result formatted to 2 decimal places
    print(f"Inverse document frequency of '{term}': {idf:.2f}")


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("build", help="Build the inverted index")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    # Add the tf command parser which accepts document ID (int) and term (str)
    tf_parser = subparsers.add_parser("tf", help="Get term frequency for a term in a document")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to count")

    idf_parser = subparsers.add_parser("idf", help="Get inverse document frequency for a term")
    idf_parser.add_argument("term", type=str, help="Term to calculate IDF for")

    args = parser.parse_args()

    match args.command:
        case "build":
            print("Building inverted index...")
            build_command()
            print("Inverted index built successfully.")
        case "search":
            print("Searching for:", args.query)
            results = search_command(args.query)
            for i, res in enumerate(results, 1):
                print(f"{i}. ({res['id']}) {res['title']}")
        case "tf":
            # Call tf_command to print the frequency of the term in the document
            tf_command(args.doc_id, args.term)
        case "idf":
            idf_command(args.term)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
