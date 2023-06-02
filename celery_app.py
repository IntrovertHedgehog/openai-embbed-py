import json
import logging
import os
import sys

import firebase_admin
import openai
import requests
from celery import Celery
from dotenv import load_dotenv
from firebase_admin import firestore
from langchain import OpenAI
from llama_index import (
    LLMPredictor,
    ServiceContext,
    StorageContext,
    load_index_from_storage,
)
from llama_index.indices.postprocessor import SimilarityPostprocessor

load_dotenv()
firebase_app = firebase_admin.initialize_app()
db = firestore.client()
openai.api_key = os.getenv("OPENAI_API_KEY")

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


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


vector_store_query_engine_002 = init_vector_store_002()
celery = Celery("healthbot", broker="pyamqp://localhost")
TALKJS_APPID = "tE8jvP6M"


@celery.task()
def makeResponseTalkJS(question, conversation_id):
    global db
    global vector_store_query_engine_002
    if question is None:
        return
    response = vector_store_query_engine_002.query(question).response.strip()
    if response is None:
        response = "This question is unanswerable"
    data = {"question": question, "response": response}
    _, response_ref = db.collection("vector-store-index-002").add(data)
    data["id"] = response_ref.id
    # add new message
    url = f"https://api.talkjs.com/v1/{TALKJS_APPID}/conversations/{conversation_id}/messages"
    headers = {
        "Authorization": f"Bearer {os.getenv('TALKJS_API_KEY')}",
        "Content-Type": "application/json",
    }
    post_data = [
        {
            "text": response,
            "sender": "23571113",
            "type": "UserMessage",
        }
    ]
    requests.post(url=url, headers=headers, json=post_data)
    data_json = json.dumps(data)
    print(data_json)
