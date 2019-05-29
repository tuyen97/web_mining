from django.shortcuts import render
import requests
from . import models
from gensim.models import Word2Vec
import logging
import os 
from recommend_system import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from . import online_train
import pickle

f = open("m_t.csv", "r")
m2t = {}
t2m = {}
for line in f:
	arr = line.split(",")
	movieId = str(arr[0])
	tmdbId = str(arr[1]).replace("\n","")
	m2t[movieId] = tmdbId
	t2m[tmdbId] = movieId

# Create your views here.
def similar_movie(request):
	movieId = t2m[request.GET['movie_id']]
	w2v_model = os.path.join(settings.BASE_DIR+"word2Vec1.model")
	model = Word2Vec.load("word2vec1.model") 
	most_sim = model.wv.most_similar(str(movieId))
	movies = []
	for mov in most_sim:
		mId = mov[0]
		movies.append(m2t[str(mId)])
	return JsonResponse({'movies':movies})

def add_user(request):
	user_id = request.GET['user_id']
	user_folder = os.path.join(settings.BASE_DIR+"/users/"+str(user_id))
	model_folder = os.path.join(settings.BASE_DIR + "/recommend")
	if not os.path.exists(user_folder):
		os.makedirs(user_folder)
	ratings = os.path.join(user_folder+"/ratings.csv")
	f = open(ratings, "w")
	f.write(str(0) + "," + str(7999) + "," + str(1))
	f.close()
	online_train.train(model_folder, user_folder)
	movies = []
	m = []
	movies = online_train.recommend(user_folder, user_id)
	for movie in movies:
		m.append(m2t[str(movie)])
	with open(user_folder + '/recommend_list', 'wb') as fp:
		pickle.dump(m, fp)
	return JsonResponse({})

@method_decorator(csrf_exempt, name='dispatch')
def add_rating(request):
	user_id = request.POST['user_id']
	movieId = t2m[request.POST['movie_id']]
	rating = request.POST['rating']
	user_folder = os.path.join(settings.BASE_DIR + "/users/" + str(user_id))
	model_folder = os.path.join(settings.BASE_DIR + "/recommend")
	ratings = os.path.join(user_folder+"/ratings.csv")
	f = open(ratings, "a")
	line = str(0)+","+str(movieId) + "," + str(rating)+"\n"
	f.write(line)
	f.close()
	f = open(os.path.join(user_folder+"/isUpdated.txt"), "w")
	f.write(str(0))
	f.close()
	return JsonResponse({})

@method_decorator(csrf_exempt, name='dispatch')
def recommend(request):
	user_id = request.GET['user_id']
	user_folder = os.path.join(settings.BASE_DIR + "/users/" + str(user_id))
	with open (user_folder + '/recommend_list', 'rb') as fp:
		itemlist = pickle.load(fp)
	return JsonResponse({'movies':itemlist})
# def train(request):
