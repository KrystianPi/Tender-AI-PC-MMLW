from functools import lru_cache
from sentence_transformers import SentenceTransformer
import spacy
import pandas as pd
from datetime import datetime, timedelta
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
import os
from sqlalchemy import create_engine
import dotenv

class Retriever:
    def __init__(self, embedding_model_name: str, index_name: str, bm25_path: str) -> None:
        """
        Initialize the Retriever with the necessary parameters.

        Args:
            df (pd.DataFrame): The dataframe containing the corpus and metadata.
            embedding_model_name (str): The name of the embedding model.
            alfa (float): The weight parameter for the ensemble retriever.
            topk (int): The number of top results to retrieve.
            persist_directory (str, optional): Directory to persist the Chroma index.
        """
        dotenv.load_dotenv()
        self.embedding_model_name = embedding_model_name
        self.index_name = index_name
        self.bm25_path = bm25_path
        self.bm25 = BM25Encoder()
        self.model = self._get_embedding()
        self._connect_index()

    def fit_bm25(self):
        try:
            self.bm25.load(self.bm25_path)
        except:
            query = 'SELECT krotki_opis_preproc FROM tenders'
            df = pd.read_sql_query(query, self.engine)
            self.bm25.fit(df['krotki_opis_preproc'])
            self.bm25.dump(self.bm25_path)

    def connect_db(self, host, password):
        rds_host = host
        rds_port = '5432'  # usually 5432
        rds_dbname = 'postgres'
        rds_username = 'postgres'
        rds_password = password
        url = f"postgresql+psycopg2://{rds_username}:{rds_password}@{rds_host}:{rds_port}/{rds_dbname}"
        self.engine = create_engine(url)

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
        return self.model.encode(text, convert_to_tensor=False, show_progress_bar=False).tolist()
    
    def _get_sparse(self, text):
        text = self._preprocess(text)
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
        df, filtered_ids = self._filter(filter)
        dense, sparse = self._encode_query(query)
        result = self.index.query(
                        top_k=20,
                        filter={
                            "id": {"$in":filtered_ids},
                        },
                        vector=dense,
                        sparse_vector=sparse,
                        include_metadata=True
                        )
        results = [result['id'] for result in result['matches']]
        print(results)
        return results, df

    def _filter(self, filter: dict):
        """
        Filter the dataframe based on the provided filter.

        Args:
            filter (dict): The filter to apply.

        Returns:
            tuple: Filtered corpus, metadata, and IDs.
        """
        locations = filter['wojewodztwo']
        categories = filter['categories']

        query = self.format_query(locations, categories)

        df = pd.read_sql_query(query, self.engine)

        # Calculate the date that is 3 days into the future
        future_date = datetime.now() + timedelta(days=5)

        # Filter the DataFrame based on end_date
        df['data_zakonczenia'] = pd.to_datetime(df['data_zakonczenia'], format="mixed")
        df = df[df['data_zakonczenia'] > future_date]

        filtered_ids = df['uuid'].tolist()
        
        return df, filtered_ids

    @lru_cache(maxsize=1)
    def _get_embedding(self):
        """Get the embedding function using HuggingFaceEmbeddings."""
        try:
            return SentenceTransformer(model_name_or_path=self.embedding_model_name)
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
        return ' '.join(tokens)
    
    @staticmethod
    def format_query(locations, categories):
        # Convert lists to PostgreSQL array strings
        locations_str = ', '.join(f"'{loc}'" for loc in locations)
        categories_str = ', '.join(f"'{cat}'" for cat in categories)
        
        # Construct the query string
        query = f'''SELECT *
    FROM tenders
    WHERE wojewodztwo::text[] && ARRAY[{locations_str}]::text[]
    AND categories::text[] && ARRAY[{categories_str}]::text[]'''
        
        return query