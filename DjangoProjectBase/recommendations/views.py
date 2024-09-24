from django.shortcuts import render
from movie.models import Movie
import os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

def __get_embedding(text, client, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

def __cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def __get_recommendations(searchTerm):
    _ = load_dotenv('../api_keys.env')
    client = OpenAI(
        # This is the default and can be omitted
        api_key=os.environ.get('openai_api_key'),
    )
    
    movies = Movie.objects.all()

    emb_req = __get_embedding(searchTerm, client)

    sim = []
    for i in range(len(movies)):
        MovieEmb = movies[i].emb
        MovieEmb = list(np.frombuffer(MovieEmb))
        sim.append(__cosine_similarity(MovieEmb,emb_req))
    sim = np.array(sim)
    idx = np.argmax(sim)
    idx = int(idx)
    return movies[idx]

def recommendations(request):
    searchTerm = request.GET.get('searchMovie') # GET se usa para solicitar recursos de un servidor
    if searchTerm:
        movie = __get_recommendations(searchTerm)
    else:
        movie = None #No term has been inserted, nothing to be shown.
    return render(request, 'recommendations.html', {'searchTerm':searchTerm, 'movie':movie})