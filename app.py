import json
import logging
import os
import sys

import firebase_admin
import numpy as np
import openai
import pandas as pd
from firebase_admin import firestore
from flask import Flask, request
from langchain import OpenAI
from llama_index import (
    LLMPredictor,
    ResponseSynthesizer,
    ServiceContext,
    StorageContext,
    load_index_from_storage,
)
from llama_index.indices.postprocessor import SimilarityPostprocessor
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.retrievers import VectorIndexRetriever
from openai.embeddings_utils import cosine_similarity

from dotenv import load_dotenv
load_dotenv()
app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# firebase setup

# Appliation Default credentials are automatically created.
firebase_app = firebase_admin.initialize_app()
db = firestore.client()

# init embbedding
df = pd.read_csv("embbeded_question.csv")
df["ada_embedding"] = df.ada_embedding.apply(eval).apply(np.array)

# init tree index
storage_context = StorageContext.from_defaults(persist_dir="./tree")
tree_index = load_index_from_storage(storage_context)


# init vector store (text-davinci-003)
def init_vector_store_003():
    sc = StorageContext.from_defaults(persist_dir="./vector_store")
    vector_store_index = load_index_from_storage(sc)

    retriever = VectorIndexRetriever(index=vector_store_index, similarity_top_k=2)

    response_synthesizer = ResponseSynthesizer.from_args(
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.85)]
    )

    query_engine = RetrieverQueryEngine(
        retriever=retriever, response_synthesizer=response_synthesizer
    )

    return query_engine


def init_vector_store_002():
    sc = StorageContext.from_defaults(persist_dir="./vector_store_vinci002")

    llm_predictor = LLMPredictor(
        llm=OpenAI(temperature=0, model_name="text-davinci-002")
    )
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

    vector_store_index = load_index_from_storage(
        storage_context=sc, service_context=service_context
    )

    query_engine = vector_store_index.as_query_engine(
        similarity_top_k=2,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.85)],
    )

    return query_engine


query_engine = init_vector_store_003()
vector_store_query_engine_002 = init_vector_store_002()


def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model=model)["data"][0]["embedding"]


def search_reviews(df, question, n=3):
    embedding = get_embedding(question)
    df["similarities"] = df.ada_embedding.apply(
        lambda x: cosine_similarity(x, embedding)
    )
    res = df.sort_values("similarities", ascending=False).head(n)
    return res


@app.route("/embbeding", methods=["POST"])
def getTopQns():
    print(request.get_json())
    question = request.get_json()["input"]
    top3 = search_reviews(df, question)
    top3["rank"] = [1, 2, 3]
    data = top3[["rank", "question", "answer"]].to_json(orient="records")
    response = app.response_class(
        response=data, status=200, mimetype="application/json"
    )
    return response


@app.route("/tree_index", methods=["POST"])
def getResTreeIndex():
    global tree_index
    question = request.get_json()["input"]
    if question is None:
        return "No text found", 400
    query_engine = tree_index.as_query_engine()
    data = {"response": query_engine.query(question).response}
    data_json = json.dumps(data)
    print(data_json)
    response = app.response_class(
        response=data_json, status=200, mimetype="application/json"
    )
    return response


@app.route("/vector_store_index", methods=["POST"])
def getResVtStIndex():
    global query_engine
    question = request.get_json()["input"]
    if question is None:
        return "No text found", 400
    data = {"response": query_engine.query(question).response}
    data_json = json.dumps(data)
    print(data_json)
    response = app.response_class(
        response=data_json, status=200, mimetype="application/json"
    )
    return response


@app.route("/vector_store_index_002", methods=["POST"])
def getResVtStIndexVinci2():
    global db
    global query_engine
    question = request.get_json()["input"]
    if question is None:
        return "No text found", 400
    data = {
        "question": question,
        "response": vector_store_query_engine_002.query(question).response,
    }
    update_time, response_ref = db.collection("vector-store-index-002").add(data)
    data["id"] = response_ref.id
    data_json = json.dumps(data)
    print(data_json)
    response = app.response_class(
        response=data_json, status=200, mimetype="application/json"
    )
    return response


@app.route("/react_vector_store_index_002/<id>", methods=["POST"])
def react_vector_store_index_002(id):
    global db
    reaction = request.get_json().get("reaction")
    if reaction != "UP" and reaction != "DOWN":
        response = app.response_class(
            response=json.dumps({"error": "Invalid reaction type"}),
            status=422,
            mimetype="application/json",
        )
        return response

    doc_ref = db.collection("vector-store-index-002").document(id)
    doc_ref.update({"reaction": reaction})
    data = doc_ref.get().to_dict()
    response = app.response_class(
        response=json.dumps(data),
        status=200,
        mimetype="application/json",
    )
    return response


@app.route("/test")
def test():
    data = {"a": 12}
    doc_ref = db.collection("tests").document("charles")
    doc_ref.set(data)
    return app.response_class(
        response=json.dumps(data), status=200, mimetype="application/json"
    )


# if __name__ == "__main__":
#     app.run()
