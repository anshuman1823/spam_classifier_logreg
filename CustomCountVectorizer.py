## Defining the CustomCountVectorizer class
from collections import Counter
from scipy.sparse import csr_matrix  ## for converting dense matrix to sparse matrix

class CustomCountVectorizer:
    def __init__(self, min_df=1, max_df=1.0):
        """
        Initializes the CustomCountVectorizer with min_df and max_df options.

        Parameters:
        - min_df: Minimum document frequency (ignores words in fewer documents than this).
        - max_df: Maximum document frequency (ignores words in more documents than this).
        """
        self.min_df = min_df
        self.max_df = max_df
        self.vocabulary_ = {}

    def fit(self, texts):
        """
        Fits the vocabulary on the provided list of texts.
        
        Parameters:
        - texts: List of documents (texts).
        """
        doc_count = len(texts)
        term_doc_freq = Counter()

        for text in texts:
            terms = set(text.lower().split())
            term_doc_freq.update(terms)

        # Apply min_df and max_df thresholds
        min_docs = self.min_df if isinstance(self.min_df, int) else int(self.min_df * doc_count)
        max_docs = self.max_df if isinstance(self.max_df, int) else int(self.max_df * doc_count)

        self.vocabulary_ = {
            term: idx for idx, (term, freq) in enumerate(term_doc_freq.items())
            if min_docs <= freq <= max_docs
        }

        # Re-index vocabulary to keep it contiguous
        self.vocabulary_ = {term: i for i, (term, idx) in enumerate(self.vocabulary_.items())}

    def transform(self, texts):
        """
        Transforms the texts into a sparse document-term matrix based on the vocabulary.

        Parameters:
        - texts: List of documents (texts).

        Returns:
        - Sparse document-term matrix with each row representing a document and columns representing terms.
        """
        rows, cols, data = [], [], []

        for i, text in enumerate(texts):
            term_counts = Counter(text.lower().split())
            for term, count in term_counts.items():
                if term in self.vocabulary_:
                    idx = self.vocabulary_[term]
                    rows.append(i)
                    cols.append(idx)
                    data.append(count)

        # Create sparse matrix in CSR format
        sparse_matrix = csr_matrix((data, (rows, cols)), shape=(len(texts), len(self.vocabulary_)), dtype=int)
        return sparse_matrix

    def fit_transform(self, texts):
        """
        Combines fit and transform into one step for convenience.
        
        Parameters:
        - texts: List of documents (texts).
        
        Returns:
        - Sparse document-term matrix.
        """
        self.fit(texts)
        return self.transform(texts)