from flask import Flask, request
import pandas as pd
import numpy as np
import json
import os
import openai
import logging
import sys
import json
from openai.embeddings_utils import cosine_similarity

from llama_index import StorageContext, load_index_from_storage, ResponseSynthesizer, LLMPredictor, ServiceContext

from llama_index.retrievers import VectorIndexRetriever
from llama_index.indices.postprocessor import SimilarityPostprocessor
from llama_index.query_engine import RetrieverQueryEngine
from langchain import OpenAI

# from dotenv import load_dotenv
# load_dotenv()
app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# init embbedding
df = pd.read_csv('embbeded_question.csv')
df['ada_embedding'] = df.ada_embedding.apply(eval).apply(np.array)

# init tree index
storage_context = StorageContext.from_defaults(persist_dir='./tree')
tree_index = load_index_from_storage(storage_context)

# init vector store (text-davinci-003)
def init_vector_store_003():
    storage_context = StorageContext.from_defaults(persist_dir='./vector_store')
    vector_store_index = load_index_from_storage(storage_context)

    retriever = VectorIndexRetriever(
        index=vector_store_index,
        similarity_top_k=2
    )

    response_synthesizer = ResponseSynthesizer.from_args(
        node_postprocessors=[
            SimilarityPostprocessor(similarity_cutoff=0.85)
        ]
    )

    vector_store_query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer
    )

    return vector_store_query_engine

def init_vector_store_002():
    storage_context = StorageContext.from_defaults(persist_dir='./vector_store_vinci002')

    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-002"))
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

    vector_store_index = load_index_from_storage(storage_context=storage_context, service_context=service_context)

    vector_store_query_engine = vector_store_index.as_query_engine(
        similarity_top_k=2,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.85)],
    )

    return vector_store_query_engine

vector_store_query_engine = init_vector_store_003()
vector_store_query_engine_002 = init_vector_store_002()

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

@app.route('/vector_store_index', methods=['POST'])
def getResVtStIndex():
    global vector_store_query_engine
    question = request.get_json()['input']
    if question is None:
        return "No text found", 400
    data={"response": vector_store_query_engine.query(question).response}
    data_json=json.dumps(data)
    print(data_json)
    response = app.response_class(
        response=data_json,
        status=200,
        mimetype='application/json'
    )
    return response

@app.route('/vector_store_index_002', methods=['POST'])
def getResVtStIndexVinci2():
    global vector_store_query_engine
    question = request.get_json()['input']
    if question is None:
        return "No text found", 400
    data={"response": vector_store_query_engine_002.query(question).response}
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
