import sys
import os
import json
import re
import warnings
warnings.filterwarnings('ignore')

# Add the root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from utils.retriever import Retriever
from utils.llm import LLM

llm = LLM()

df = pd.read_csv('mnt/efs/data_v1.csv')
df = df.dropna(subset='kategoria')

# Initialize the Retriever
retriever = Retriever(df, 'mnt/efs/model', alfa=0.7, topk=10, persist_directory='mnt/efs/vectorDB')
retriever.setup_chroma()

query = '''
tencjalne oferty obejmują nadzór inwestorski, przygotowanie dokumentacji przetargowej oraz pełnienie funkcji kierownika robót. Dodatkowo, możliwe projekty to przeglądy techniczne budynków, ekspertyzy techniczne i weryfikacja kosztów robót budowlanych. Inne zadania to prace konserwatorskie na cmentarzach i w parkach, w tym remont alejek i budowa ścieżek.
'''

cat = ["administrowanie obiektami/lokalami",
       "budownictwo - obsługa",]

filter_dict = {'kategoria': cat}

results = retriever.search(query, filter_dict)

prompt = f'''Your task is filter the results from tenant search to best matches based on the company profile.
            Each tenant search results has it's unique ID. 
            Return top 5 results by providing the IDs in json format with one key: 'id' and the value being a list of ids.
            Return only the json format and nothing else! Do not provide any explanations!

            Here are the results from the initial search:
            {results}
            
            Here is the company profile:
            {query}

            Filtered IDs json format.:
            '''

response = llm.complete(prompt)
print(response)

pattern = r'\{[^{}]*\}'
json_string = re.findall(pattern, response)[0]
response = json.loads(json_string)
ids = response['id']

results = [result for result in results if result.metadata['id'] in ids]

print(results)