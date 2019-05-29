import os 
import online_train
import pickle

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
users_folder = os.path.join(BASE_DIR + "/users/")
model_folder = os.path.join(BASE_DIR + "/recommend")
m2t = {}
t2m = {}
f = open(BASE_DIR + "/m_t.csv", "r")
for line in f:
	arr = line.split(",")
	movieId = str(arr[0])
	tmdbId = str(arr[1]).replace("\n","")
	m2t[movieId] = tmdbId
	t2m[tmdbId] = movieId

while(True):
	print("running")
	folders = os.listdir(users_folder)
	print(folders)
	for user_id in folders:
		user_folder = os.path.join(BASE_DIR+"/users/"+str(user_id))
		try:
			fp = open(os.path.join(user_folder+"/isUpdated.txt"), "r")
			isUpdated = int(fp.readline().replace("\n", ""))
			print(user_folder+" "+str(isUpdated))
			fp.close()
			if not isUpdated:
				online_train.train(model_folder, user_folder)
				m = []
				movies = online_train.recommend(user_folder, user_id)
				for movie in movies:
					m.append(m2t[str(movie)])
				with open(user_folder + '/recommend_list', 'wb') as fp:
					pickle.dump(m, fp)
				fp = open(os.path.join(user_folder+"/isUpdated.txt"), "w")
				fp.write(str(1))
				fp.close()
		except Exception as e:
			print(e)