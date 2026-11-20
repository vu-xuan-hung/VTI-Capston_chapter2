"""Embed text by hashing its words: no model, the same vector everywhere.

For tests and the fake backend only. Texts that share words get similar
vectors, enough to check that search, filters and re-indexing work.
"""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

import  hashlib
import  math
import  re

from    langchain_core.embeddings       import  Embeddings


#==========================================================================
# Hashing embeddings
#==========================================================================
class HashEmbeddings(Embeddings):
    """
    Purpose: a bag of hashed lower-case words, L2-normalised. blake2b, not
    hash(): Python salts hash() per process.
    """

    #----------------------------------------------------------------------
    # Choose the vector size
    #----------------------------------------------------------------------
    def __init__(self, dim: int = 64) -> None:
        """
        Fix the number of buckets.

        Input:
            . dim           : vector length
        """
        self.dim  = dim
        self.name = f"hash-{dim}"

    #----------------------------------------------------------------------
    # Embed many texts
    #----------------------------------------------------------------------
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Hash each text into one vector.

        Input:
            . texts         : texts to embed
        Output:
            . vectors       : one unit vector per text
        """
        return [self._vector(text) for text in texts]

    #----------------------------------------------------------------------
    # Embed one query
    #----------------------------------------------------------------------
    def embed_query(self, text: str) -> list[float]:
        """
        Hash a query exactly like a document.

        Input:
            . text          : the query
        Output:
            . vector        : its unit vector
        """
        return self._vector(text)

    #----------------------------------------------------------------------
    # Hash one text
    #----------------------------------------------------------------------
    def _vector(self, text: str) -> list[float]:
        """
        Count hashed words into buckets and normalise.

        Input:
            . text          : any text
        Output:
            . vector        : unit vector; an empty text gives bucket 0 = 1
        """
        counts = [0.0] * self.dim
        for word in re.findall(r"\w+", text.lower()):
            bucket = int.from_bytes(hashlib.blake2b(word.encode(), digest_size=4).digest(), "big")
            counts[bucket % self.dim] += 1.0
        norm = math.sqrt(sum(value * value for value in counts))
        if norm == 0:
            counts[0] = 1.0
            return counts
        return [value / norm for value in counts]
