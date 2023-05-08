from flask import Flask, request
import pandas as pd
import numpy as np
import json
import os
import openai
import logging
import sys
import json
# from dotenv import load_dotenv
from openai.embeddings_utils import cosine_similarity
from llama_index import GPTTreeIndex, StorageContext, load_index_from_storage

app = Flask(__name__)
# load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# init embbedding
df = pd.read_csv('embbeded_question.csv')
df['ada_embedding'] = df.ada_embedding.apply(eval).apply(np.array)

# init llama index
persist_dir = "./tree"
storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
tree_index = load_index_from_storage(storage_context)

def get_embedding(text, model="text-embedding-ada-002"):
	text = text.replace("\n", " ")
	return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

def search_reviews(df, question, n=3):
    embedding = get_embedding(question)
    df['similarities'] = df.ada_embedding.apply(lambda x: cosine_similarity(x, embedding))
    res = df.sort_values('similarities', ascending=False).head(n)
    return res

@app.route('/embbeding', methods=['POST'])
def getTopQns():
	print(request.get_json())
	question = request.get_json()['input']
	top3 = search_reviews(df, question)
	top3['rank'] = [1,2,3]
	data = top3[['rank', 'question', 'answer']].to_json(orient="records")
	response = app.response_class(
		response=data,
		status=200,
		mimetype='application/json'
	)
	return response

@app.route('/tree_index', methods=['POST'])
def getResTreeIndex():
    global tree_index
    question = request.get_json()['input']
    if question is None:
        return "No text found", 400
    query_engine = tree_index.as_query_engine()
    data={"response": query_engine.query(question).response}
    data_json=json.dumps(data)
    print(data_json)
    response = app.response_class(
        response=data_json,
        status=200,
        mimetype='application/json'
    )
    return response

@app.route('/test')
def test():
    data={"a":12}
    return app.response_class(
        response=json.dumps(data),
        status=200,
        mimetype="application/json"
    )

# if __name__ == "__main__":
#    app.run()
