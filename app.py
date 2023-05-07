from flask import Flask, request
import pandas as pd
import numpy as np
import json
import os
import openai
from openai.embeddings_utils import cosine_similarity

app = Flask(__name__)
df = pd.read_csv('embbeded_question.csv')
df['ada_embedding'] = df.ada_embedding.apply(eval).apply(np.array)
# openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_key_path = 'openai_api_key'

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

# if __name__ == "__main__":
#    app.run()
