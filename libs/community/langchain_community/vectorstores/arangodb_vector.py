from __future__ import annotations

from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Type
from uuid import uuid4

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from packaging import version

try:
    from arango.database import Database
    from arango.exceptions import ArangoServerError
    from arango.graph import Graph

    ARANGO_INSTALLED = True
except ImportError:
    print("ArangoDB not installed, please install with `pip install python-arango`.")
    ARANGO_INSTALLED = False

from langchain_community.vectorstores.utils import DistanceStrategy, maximal_marginal_relevance

DEFAULT_DISTANCE_STRATEGY = DistanceStrategy.COSINE
DISTANCE_MAPPING = {
    DistanceStrategy.EUCLIDEAN_DISTANCE: "l2",
    DistanceStrategy.COSINE: "cosine",
}


class SearchType(str, Enum):
    """Enumerator of the Distance strategies."""

    VECTOR = "vector"
    # HYBRID = "hybrid" # TODO


DEFAULT_SEARCH_TYPE = SearchType.VECTOR


class ArangoVector(VectorStore):
    """ArangoDB vector index.

    To use this, you should have the `python-arango` python package installed.

    Args:
        embedding: Any embedding function implementing
            `langchain.embeddings.base.Embeddings` interface.
        database: The python-arango database instance.
        embedding_dimension: The dimension of the to-be-inserted embedding vectors.
        search_type: The type of search to be performed, currently only 'vector' is supported.
        collection_name: The name of the collection to use. (default: "documents")
        index_name: The name of the vector index to use. (default: "vector_index")
        text_field: The field name storing the text. (default: "text")
        embedding_field: The field name storing the embedding vector. (default: "embedding")
        distance_strategy: The distance strategy to use. (default: "COSINE")
        num_centroids: The number of centroids for the vector index. (default: 1)

    Example:
        .. code-block:: python

            from arango import ArangoClient
            from langchain_community.embeddings.openai import OpenAIEmbeddings
            from langchain_community.vectorstores.arangodb_vector import ArangoDBVector

            db = ArangoClient("http://localhost:8529").db("test", username="root", password="openSesame")

            vector_store = ArangoDBVector(
                embedding=OpenAIEmbeddings(), database=db
            )

            texts = ["hello world", "hello langchain", "hello arangodb"]

            vector_store.add_texts(texts)

            print(vector_store.similarity_search("arangodb", k=1))
    """

    def __init__(
        self,
        embedding: Embeddings,
        *,
        database: "Database",
        embedding_dimension: int,
        search_type: SearchType = DEFAULT_SEARCH_TYPE,
        collection_name: str = "documents",
        index_name: str = "vector_index",
        text_field: str = "text",
        embedding_field: str = "embedding",
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        num_centroids: int = 1,
    ):
        if not ARANGO_INSTALLED:
            m = "ArangoDB not installed, please install with `pip install python-arango`."
            raise ImportError(m)

        # TODO: Enable when ready
        # if version.parse(database.version()) < version.parse("3.12.0"):
        # raise ValueError("ArangoDB version must be 3.12.0 or greater")

        if search_type not in [SearchType.VECTOR]:
            raise ValueError("search_type must be 'vector'")

        if distance_strategy not in [
            DistanceStrategy.COSINE,
            DistanceStrategy.EUCLIDEAN_DISTANCE,
        ]:
            raise ValueError("distance_strategy must be 'COSINE' or 'EUCLIDEAN_DISTANCE'")

        self.db = database
        self.embedding = embedding
        self.collection_name = collection_name
        self.index_name = index_name
        self.embedding_field = embedding_field
        self.text_field = text_field
        self.distance_strategy = DISTANCE_MAPPING[distance_strategy]
        self.embedding_dimension = embedding_dimension
        self.num_centroids = num_centroids
        self.index_name = index_name

        if not database.has_collection(collection_name):
            database.create_collection(collection_name)

        self.collection = database.collection(self.collection_name)

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding

    def add_embeddings(
        self,
        texts: Iterable[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        if ids is None:
            ids = [str(uuid4()) for _ in texts]

        if not metadatas:
            metadatas = [{} for _ in texts]

        to_insert = [
            {
                "_key": id_,
                self.text_field: text,
                self.embedding_field: embedding,
                "metadata": metadata,
            }
            for id_, text, embedding, metadata in zip(ids, texts, embeddings, metadatas)
        ]

        self.collection.import_bulk(to_insert, on_duplicate="update", **kwargs)

        if self.index_name not in [index["name"] for index in self.collection.indexes()]:
            self.collection.add_index(
                {
                    "name": self.index_name,
                    "type": "vector",
                    "fields": [self.embedding_field],
                    "params": {
                        "metric": self.distance_strategy,
                        "dimensions": self.embedding_dimension,
                        "nLists": self.num_centroids,
                    },
                }
            )

        return ids

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add texts to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of ids to associate with the texts.

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        embeddings = self.embedding.embed_documents(list(texts))
        return self.add_embeddings(texts=texts, embeddings=embeddings, metadatas=metadatas, ids=ids, **kwargs)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        return_full_doc: bool = True,
        **kwargs: Any,
    ) -> List[Document]:
        """Run similarity search with ArangoDB.

        Args:
            query (str): Query text to search for.
            k (int): Number of results to return. Defaults to 4.
            return_full_doc (bool): Whether to return the full document.
                If false, will just return the _key. Defaults to True.

        Returns:
            List of Documents most similar to the query.
        """
        embedding = self.embedding.embed_query(query)
        return self.similarity_search_by_vector(embedding, k, return_full_doc)

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        return_full_doc: bool = True,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            return_full_doc (bool): Whether to return the full document.
                If false, will just return the _key. Defaults to True.

        Returns:
            List of Documents most similar to the query vector.
        """
        docs_and_scores = self.similarity_search_by_vector_with_score(
            embedding=embedding, k=k, return_full_doc=return_full_doc, **kwargs
        )

        return [doc for doc, _ in docs_and_scores]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        return_full_doc: bool = True,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            return_full_doc (bool): Whether to return the full document.
                If false, will just return the _key. Defaults to True.

        Returns:
            List of Documents most similar to the query and score for each
        """
        embedding = self.embedding.embed_query(query)
        result = self.similarity_search_by_vector_with_score(
            embedding=embedding,
            k=k,
            query=query,
            return_full_doc=return_full_doc,
            **kwargs,
        )
        return result

    def similarity_search_by_vector_with_score(
        self,
        embedding: List[float],
        k: int = 4,
        return_full_doc: bool = False,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            return_full_doc (bool): Whether to return the full document.
                If false, will just return the _key. Defaults to True.

        Returns:
            List of Documents most similar to the query vector.
        """
        if self.distance_strategy == "cosine":
            sort_func = "APPROX_NEAR_COSINE"
        elif self.distance_strategy == "l2":
            sort_func = "APPROX_NEAR_L2"
        else:
            raise ValueError(f"Unsupported metric: {self.distance_strategy}")

        aql = f"""
            FOR doc IN @@collection
                LET score = {sort_func}(doc.{self.embedding_field}, @embedding)
                SORT score DESC
                LIMIT @k
                LET data = @return_full_doc ? doc : {{'_key': doc._key, {self.text_field}: doc.{self.text_field}}}
                RETURN {{data, score}}
        """

        bind_vars = {
            "@collection": self.collection_name,
            "embedding": embedding,
            "k": k,
            "return_full_doc": return_full_doc,
        }

        cursor = self.db.aql.execute(aql, bind_vars=bind_vars)

        results = []
        for result in cursor:
            page_content = result["data"].pop(self.text_field)
            results.append((Document(page_content=page_content, **result["data"]), result["score"]))

        return results

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete by vector ID or other criteria.

        Args:
            ids: List of ids to delete.
            **kwargs: Other keyword arguments that can be used to delete vectors.

        Returns:
            Optional[bool]: True if deletion is successful,
            False otherwise, None if not implemented.
        """
        for result in self.collection.delete_many(ids, **kwargs):
            if isinstance(result, ArangoServerError):
                print(result)
                return False

        return True

    def get_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        """Get documents by their IDs.

        Args:
            ids: List of ids to get.

        Returns:
            List of Documents with the given ids.
        """
        return self.collection.get_many(ids)

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            query: search query text.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.

        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        # Embed the query
        query_embedding = self.embedding.embed_query(query)

        # Fetch the initial documents
        docs_with_scores = self.similarity_search_by_vector_with_score(
            embedding=query_embedding,
            k=fetch_k,
            return_full_doc=True,
            **kwargs,
        )

        # Get the embeddings for the fetched documents
        embeddings = [doc[self.embedding_field] for doc, _ in docs_with_scores]

        # Select documents using maximal marginal relevance
        selected_indices = maximal_marginal_relevance(
            np.array(query_embedding), embeddings, lambda_mult=lambda_mult, k=k
        )

        selected_docs = [docs_with_scores[i][0] for i in selected_indices]

        # Remove embedding values from metadata
        for doc in selected_docs:
            del doc[self.embedding_field]

        return selected_docs

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """
        The 'correct' relevance function
        may differ depending on a few things, including:
        - the distance / similarity metric used by the VectorStore
        - the scale of your embeddings (OpenAI's are unit normed. Many others are not!)
        - embedding dimensionality
        - etc.
        """
        if self.override_relevance_score_fn is not None:
            return self.override_relevance_score_fn

        # Default strategy is to rely on distance strategy provided
        # in vectorstore constructor
        if self._distance_strategy == DistanceStrategy.COSINE:
            return lambda x: x
        elif self._distance_strategy == DistanceStrategy.L2:
            return lambda x: x
        else:
            raise ValueError(
                "No supported normalization function"
                f" for distance_strategy of {self._distance_strategy}."
                "Consider providing relevance_score_fn to PGVector constructor."
            )

    @classmethod
    def from_texts(
        cls: Type[ArangoVector],
        texts: List[str],
        embedding: Embeddings,
        database: "Database",
        search_type: SearchType = DEFAULT_SEARCH_TYPE,
        collection_name: str = "documents",
        index_name: str = "vector_index",
        text_field: str = "text",
        embedding_field: str = "embedding",
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        num_centroids: int = 1,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ArangoVector:
        """
        Return ArangoDBVector initialized from texts, embeddings and a database.
        """
        embeddings = embedding.embed_documents(list(texts))

        embedding_dimension = len(embeddings[0])

        store = cls(
            embedding,
            database=database,
            collection_name=collection_name,
            embedding_dimension=embedding_dimension,
            search_type=search_type,
            index_name=index_name,
            text_field=text_field,
            embedding_field=embedding_field,
            distance_strategy=distance_strategy,
            num_centroids=num_centroids,
            **kwargs,
        )

        store.add_embeddings(texts, embeddings, metadatas=metadatas, ids=ids, **kwargs)

        return store
