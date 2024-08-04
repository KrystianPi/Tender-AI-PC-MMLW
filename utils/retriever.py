from functools import lru_cache
from sentence_transformers import SentenceTransformer
import spacy
import pandas as pd
from datetime import datetime, timedelta
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
import os

class Retriever:
    def __init__(self, df: pd.DataFrame, embedding_model_name: str) -> None:
        """
        Initialize the Retriever with the necessary parameters.

        Args:
            df (pd.DataFrame): The dataframe containing the corpus and metadata.
            embedding_model_name (str): The name of the embedding model.
            alfa (float): The weight parameter for the ensemble retriever.
            topk (int): The number of top results to retrieve.
            persist_directory (str, optional): Directory to persist the Chroma index.
        """
        load_dotenv()
        self.df = df
        self.embedding_model_name = embedding_model_name
        self.index_name = "tender-ai-hybrid"
        self.bm25 = BM25Encoder()
        self.model = self._get_embedding()

    def _connect_index(self):
        """Setup Pinecone index."""
        try:
            self.pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
            self.index = self.pc.Index(self.index_name)
        except Exception as e:
            print(f"Error setting up Pinecone: {e}")

    def _encode_query(self, query):
        dense = self._get_dense(query)
        sparse = self._get_sparse(query)
        return dense, sparse

    def _get_dense(self, text):
        text = 'zapytanie: ' + text
        print(self.model.encode(text, convert_to_tensor=False, show_progress_bar=False))
        return self.model.encode(text, convert_to_tensor=False, show_progress_bar=False)
    
    def _get_sparse(self, text):
        return self.bm25.encode_queries(text)

    def search(self, query: str, filter: dict):
        """
        Search the corpus based on the query and filter.

        Args:
            query (str): The search query.
            filter (dict): The filter to apply to the dataframe.

        Returns:
            list: The search results.
        """
        filtered_ids = self._filter(filter)
        dense, sparse = self._encode_query(query)
        result = self.index.query(
                        top_k=5,
                        filter={
                            "id": {"$in":filtered_ids},
                        },
                        vector=dense,
                        sparse_vector=sparse,
                        include_metadata=True
                        )
        results = [result['id'] for result in result['matches']]
        return results

    def _filter(self, filter: dict):
        """
        Filter the dataframe based on the provided filter.

        Args:
            filter (dict): The filter to apply.

        Returns:
            tuple: Filtered corpus, metadata, and IDs.
        """
        filtered_df = self.df.copy()

        for column, values in filter.items():
            if column in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[column].apply(lambda x: any(item in x for item in values))]
                print(filtered_df)

        # Calculate the date that is 3 days into the future
        future_date = datetime.now() + timedelta(days=5)

        # Filter the DataFrame based on end_date
        filtered_df['Data Zakończenia'] = pd.to_datetime(filtered_df['Data Zakończenia'], format="mixed")
        filtered_df = filtered_df[filtered_df['Data Zakończenia'] > future_date]
        
        filtered_ids = filtered_df['uuid'].tolist()

        return filtered_ids

    @lru_cache(maxsize=1)
    def _get_embedding(self):
        """Get the embedding function using HuggingFaceEmbeddings."""
        try:
            return SentenceTransformer(model_name=self.embedding_model_name)
        except Exception as e:
            print(f"Error loading embedding model: {e}")


    @staticmethod
    def _preprocess(text: str):
        """
        Preprocess the text using Spacy for tokenization and lemmatization.

        Args:
            text (str): The text to preprocess.

        Returns:
            list: The preprocessed tokens.
        """
        text = text.replace('\n', ' ').replace('\t', ' ')
        nlp = spacy.load("pl_core_news_sm")
        doc = nlp(text)
        tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
        return tokens