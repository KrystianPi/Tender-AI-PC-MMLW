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
        self.load_data()
        self.initialize_components()
        self.load_prompt()
        self.setup_routes()

    def load_env_vars(self):
        self.MODEL_PATH = os.getenv('MODEL_PATH', 'mnt/efs/model')
        self.DF_PATH = os.getenv('DF_PATH', 'mnt/efs/data_pre_v3.csv')
        self.PROMPT_PATH = os.getenv('PROMPT_PATH', 'prompt.txt')

    def load_data(self):
        try:
            self.df = pd.read_csv(self.DF_PATH)
            self.df['Województwo'] = [item.split(';') for item in self.df['Województwo']]
            self.df['categories'] = [item.split(';') for item in self.df['categories']]
        except Exception as e:
            logger.error(f"Failed to load data from {self.DF_PATH}: {e}")
            raise

    def initialize_components(self):
        try:
            self.retriever = Retriever(self.df, self.MODEL_PATH)
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
        filter_dict = {'categories': categories, 'Województwo': locations}
        results = self.retriever.search(query, filter_dict)
        ids = [result.metadata['id'] for result in results]
        descriptions_list = [f'id: {number}: {self.df[self.df["uuid"] == id]["Krótki opis"].squeeze()}' for id, number in zip(ids, range(len(ids)))]
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
                'opis': self.df[self.df['uuid'] == id]["Krótki opis"].squeeze(),
                'Województwo': self.df[self.df['uuid'] == id]["Województwo"].squeeze(),
                'Miasto': self.df[self.df['uuid'] == id]["Miasto/Powiat"].squeeze(),
                'url': self.df[self.df['uuid'] == id]["Link"].squeeze(),
                'Zamawiający': self.df[self.df['uuid'] == id]["Zamawiający"].squeeze(),
                'Data Zakończenia': self.df[self.df['uuid'] == id]["Data Zakończenia"].squeeze(),
                'Kategoria': self.df[self.df['uuid'] == id]["kategoria - colab"].squeeze(),
            } for id, summary in zip(ids_response, summaries)]
            return filtered_results

        except Exception as e:
            logger.error(f"Error processing the LLM response: {e}")
            raise

tenant_search_api = TenantSearchAPI()
app = tenant_search_api.app