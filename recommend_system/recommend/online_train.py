import pandas as pd
from scipy import sparse 
import numpy as np
from keras.models import Sequential, Model, load_model
from keras.layers import Input, dot, Dense, Embedding, merge, Flatten, multiply, concatenate, Lambda
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, TensorBoard
from keras import initializers
from keras.optimizers import Adagrad, Adam
from time import time
from keras import backend as K 

def train(model_path, user_path):
	n_items = 11118
	n_users = 14680
	file = user_path + '/ratings.csv'
	ratings = pd.read_csv(file, sep=',', encoding='latin-1', header = None)

	row_idx = ratings.as_matrix()[:,0].astype(np.int32)
	col_idx = ratings.as_matrix()[:,1].astype(np.int32)

	data = ratings.as_matrix()[:,2]

	num_users = int(np.max(row_idx)) +1
	num_items = int(np.max(col_idx)) +1

	sp = sparse.coo_matrix((data, (row_idx, col_idx)), (1, n_items)).tocsr()

	idx = np.argwhere(sp != 0)

	n_ratings = idx.shape[0]# number rating
	users = []
	item1 = []
	item2 = []
	labels = []
	for i in range(n_ratings):
		Ds = []
		D_s = []
		u = idx[i][0]
		i = idx[i][1] 
		j = np.random.randint(n_items)
		print(j)
		while sp[u,j] !=0:
			j = np.random.randint(n_items)
		users.append(u)
		item1.append(i)
		item2.append(j)
		labels.append(1)
		users.append(u)
		item1.append(j)
		item2.append(i)
		labels.append(0)

	model = load_model(model_path + '/model.h5')
	layers = [128,64,32,16]
	learning_rate = 0.0001
	embedding_dim = 32
	num_layer = 4
	user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
	item_input1 = Input(shape=(1,), dtype='int32', name = 'item_input1')
	item_input2 = Input(shape=(1,), dtype='int32', name = 'item_input2')

	NBPR_Embedding_Item = Embedding(input_dim = n_items, output_dim = layers[3], 
		name = 'nbpr_embedding_item',W_regularizer = l2(0), input_length=1)
	NBPR_Embedding_User = Embedding(input_dim = 1, output_dim = layers[3], name = 'nbpr_embedding_user',
	                              embeddings_initializer=initializers.random_normal(mean=0, stddev=0.01)
	                              , W_regularizer = l2(0), input_length=1)


	DNCR_Embedding_Item = Embedding(input_dim = n_items, output_dim = embedding_dim, name = 'dncr_embedding_item',
									 W_regularizer = l2(0), input_length=1)
	DNCR_Embedding_User = Embedding(input_dim =1, output_dim = embedding_dim, name = 'dncr_embedding_user',
	                              embeddings_initializer=initializers.random_normal(mean=0, stddev=0.01), 
	                              W_regularizer = l2(0), input_length=1)
	inverse = Lambda(lambda x: -x)
	# NBPR part
	nbpr_user_latent = Flatten()(NBPR_Embedding_User(user_input))
	nbpr_item_latent1 = Flatten()(NBPR_Embedding_Item(item_input1))
	nbpr_item_latent2 = Flatten()(NBPR_Embedding_Item(item_input2))
	nbpr_vector = multiply([concatenate([nbpr_user_latent, nbpr_item_latent1]), concatenate([nbpr_user_latent, inverse(nbpr_item_latent2)])])

	#DNCR part
	dncr_user_latent = Flatten()(DNCR_Embedding_User(user_input))
	dncr_item_latent1 = Flatten()(DNCR_Embedding_Item(item_input1))
	dncr_item_latent2 = Flatten()(DNCR_Embedding_Item(item_input2))
	dncr_vector = concatenate([dncr_user_latent, dncr_item_latent1, inverse(dncr_item_latent2)])

	# 4 layers
	layer = Dense(layers[0], W_regularizer= l2(0), activation='tanh', name="layer0")
	dncr_vector = layer(dncr_vector)
	layer = Dense(layers[1], W_regularizer= l2(0), activation='tanh', name="layer1")
	dncr_vector = layer(dncr_vector)
	layer = Dense(layers[2], W_regularizer= l2(0), activation='tanh', name="layer2")
	dncr_vector = layer(dncr_vector)
	layer = Dense(layers[3], W_regularizer= l2(0), activation='tanh', name="layer3")
	dncr_vector = layer(dncr_vector)

	predict_vector = concatenate([nbpr_vector, dncr_vector])
	prediction = Dense(1, activation='sigmoid', init='lecun_uniform', name = "prediction")(predict_vector)
	m = Model(inputs=[user_input, item_input1, item_input2], 
	              outputs=prediction)
	m.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')
	m.layers[3].set_weights(model.layers[3].get_weights())
	m.layers[4].set_weights(model.layers[4].get_weights())
	m.layers[13].set_weights(model.layers[13].get_weights())
	m.layers[16].set_weights(model.layers[16].get_weights())
	m.layers[19].set_weights(model.layers[19].get_weights())
	m.layers[21].set_weights(model.layers[21].get_weights())
	m.layers[23].set_weights(model.layers[23].get_weights())
	m.fit([np.array(users), np.array(item1), np.array(item2)], np.array(labels), epochs =10)
	m.save(user_path + "/model.h5")
	K.clear_session()

def recommend(user_path, userId):
	K.set_learning_phase(0)
	num_user = 14680
	num_movie = 11118
	topK = 10
	m = load_model(user_path + "/model.h5")
	print(m.summary())
	unchecked = []
	file = user_path + '/ratings.csv'
	ratings = pd.read_csv(file, sep=',', encoding='latin-1', header = None)

	row_idx = ratings.as_matrix()[:,0].astype(np.int32)
	col_idx = ratings.as_matrix()[:,1].astype(np.int32)

	data = ratings.as_matrix()[:,2]
	sp = sparse.coo_matrix((data, (row_idx, col_idx)), (1, num_movie)).tocsr()
	for i in range(num_movie):
		if sp[0,i] == 0:
			unchecked.append(i)
	p = []
	for n in range(topK):
		print(n)
		maX = unchecked[0]
		idx = 0
		for i in range(1, len(unchecked)):
			# print(i)
			j = unchecked[i]
			y_ujmax = m.predict([[0], [j], [maX]])
			y_umaxj = m.predict([[0],[maX], [j]])
			if(y_ujmax > y_umaxj):
				maX = j
				idx = i
		unchecked.pop(idx)
		p.append(maX)
	K.clear_session()
	return p
