from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError
import pandas as pd
import os
import logging
import warnings
import re
import json
from typing import List, Tuple
from utils.retriever import Retriever
from utils.llm import LLM
import dotenv
dotenv.load_dotenv()

warnings.filterwarnings('ignore')

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define request model
class SearchQuery(BaseModel):
    query: str = Field(..., title="Company Profile", description="The profile of the company to match against.")
    categories: List[str] = Field(..., title="Categories", description="List of categories to filter.")
    locations: List[str] = Field(..., title="Locations", description="List of locations to filter.")

class TenantSearchAPI:
    def __init__(self):
        self.app = FastAPI()
        self.load_env_vars()
        self.initialize_components()
        self.connect_database()
        self.load_prompt()
        self.setup_routes()
        self.retriever.fit_bm25()

    def load_env_vars(self):
        self.MODEL_PATH = os.getenv('MODEL_PATH', 'mnt/efs/model')
        self.PROMPT_PATH = os.getenv('PROMPT_PATH', 'prompt.txt')
        self.RDS_HOST = os.getenv('RDS_HOST')
        self.RDS_PASSWORD = os.getenv('RDS_PASSWORD')
        self.INDEX_NAME = os.getenv('INDEX_NAME')
        self.BM25_PATH = '/mnt/efs/bm25.json'

    def connect_database(self):
        try:
            self.retriever.connect_db(self.RDS_HOST, self.RDS_PASSWORD)
        except Exception as e:
            logger.error(f"Failed to connect to database at {self.RDS_HOST}: {e}")
            raise

    def initialize_components(self):
        try:
            self.retriever = Retriever(self.MODEL_PATH, self.INDEX_NAME, self.BM25_PATH)
        except Exception as e:
            logger.error(f"Failed to initialize the retriever: {e}")
            raise

        try:
            self.llm = LLM()
        except Exception as e:
            logger.error(f"Failed to initialize the LLM: {e}")
            raise

    def load_prompt(self):
        try:
            with open(self.PROMPT_PATH, 'r') as file:
                self.prompt_template = file.read()
        except Exception as e:
            logger.error(f"Failed to load prompt from {self.PROMPT_PATH}: {e}")
            raise

    def setup_routes(self):
        @self.app.post("/search")
        async def search_endpoint(query: SearchQuery):
            try:
                results = self.search_tenants(query.query, query.categories, query.locations)
                return results
            except ValidationError as e:
                logger.error(f"Validation error: {e}")
                raise HTTPException(status_code=422, detail="Invalid input data")
            except Exception as e:
                logger.error(f"Error during search: {e}")
                raise HTTPException(status_code=500, detail=f"Error processing the request: {str(e)}")

        @self.app.get("/health")
        async def health():
            return {"status": "ok"}

    def search_tenants(self, query: str, categories: List[str], locations: List[str]) -> List[Tuple[int, str]]:
        filter_dict = {'categories': categories, 'wojewodztwo': locations}
        ids, df = self.retriever.search(query, filter_dict)
        short_descriptions = []
        for id in ids:
            text = df[df["uuid"] == id]["krotki_opis"].squeeze()
            text = text.replace("\n", " ")
            tokens = text.split(' ')[:1000]
            short_descriptions.append(' '.join(tokens))
        descriptions_list = [f'id: {number}: {desc}' for desc, number in zip(short_descriptions, range(len(ids)))]
        descriptions = '\n'.join(descriptions_list)

        prompt = self.prompt_template.format(descriptions=descriptions, query=query)

        response = self.llm.complete(prompt)

        try:
            pattern = r'\{[^{}]*\}'
            json_string = re.findall(pattern, response)[0]
            response = json.loads(json_string)
            ids_response = response['id']
            ids_response =  [ids[i] for i in ids_response]
            summaries = response['summary']
            filtered_results = [{
                'id': id,
                'summary': summary,
                'opis': df[df['uuid'] == id]["krotki_opis"].squeeze(),
                'Województwo': df[df['uuid'] == id]["wojewodztwo"].squeeze(),
                'Miasto': df[df['uuid'] == id]["miasto_powiat"].squeeze(),
                'url': df[df['uuid'] == id]["link"].squeeze(),
                'Zamawiający': df[df['uuid'] == id]["zamawiajacy"].squeeze(),
                'Data Zakończenia': df[df['uuid'] == id]["data_zakonczenia"].squeeze(),
                'Kategoria': df[df['uuid'] == id]["kategoria_colab"].squeeze(),
            } for id, summary in zip(ids_response, summaries)]
            return filtered_results

        except Exception as e:
            logger.error(f"Error processing the LLM response: {e}")
            raise

tenant_search_api = TenantSearchAPI()
app = tenant_search_api.app