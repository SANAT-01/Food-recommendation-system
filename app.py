#================================================================= Importing the necessary libraries =================================================================

import _pickle as cPickle
import numpy as np
import pandas as pd

#================================================================= Recommendatin functions ===============================================================================

# The content based recommender system
def Recommendation(data,sim):
    re_li = []
    df = pd.DataFrame([])
    ind = indices[indices.index == data]['index'][0]
    sim_score = list(enumerate(sim[ind]))
    sim_score = sorted(sim_score,key=lambda x:x[1],reverse=True)
    sim_score = sim_score[0:10]
    rec_indices = [i[0] for i in sim_score][1:]
    for i in rec_indices:
        re_li.append(rec.iloc[i]["name"])
        
    return re_li, rec_indices

# The collaborative based recommender System
def Get_Recommendations(recommender, title, indices):
    user = indices[indices.index==title] # gets the row which contains the id of the given food
    print(user['recipe_id'])
    recipe_index = np.where(rating_matrix.index==int(user['recipe_id'][0]))[0][0] # gets the index where we can get the food with that id
    user_ratings = rating_matrix.iloc[recipe_index] # returns the row with the ratings given by each users
    reshaped = user_ratings.values.reshape(1,-1) # reshaping into 2d array
    distances, idx = recommender.kneighbors(reshaped,n_neighbors=11) # return distances and index of the rows of rating_matrix with is similar to the given input
    nearest_neighbors_indices = rating_matrix.iloc[idx[0]].index[:] # gets the index of those rows

    nearest_neighbors = pd.DataFrame({'recipe_id': nearest_neighbors_indices}) # converitng it into dataframe
    # print(nearest_neighbors_indices) ## due to too many users the same product is little far from similarity
    names = []
    for idx in nearest_neighbors_indices:
        names.append(indices[indices.recipe_id==idx].index[0])
        # print(names[-1])
    return names, nearest_neighbors_indices

#============================================================== Loading the pre trained data and csv files =================================================================

with open('recommender.pkl', 'rb') as fid:
    recom = cPickle.load(fid)

indices = pd.read_csv('Images_links.csv')
indices = indices.drop(['recipe_id'], axis=1)
rating_matrix = pd.read_csv('rating_matrix.csv', index_col='recipe_id')
name_consin_sim = np.load('name_consin_sim.npy')
rec_consin_sim = np.load('rec_consin_sim.npy')
rec = pd.read_csv('rec.csv')
data = pd.read_csv('data.csv', index_col='Unnamed: 0')
indices = pd.merge(indices,rec, right_on='name',left_on='name') # after merging we have two recipe_id -> 'recipe_id_x', 'recipe_id_y'
indices = indices.set_index('name')

#=========================================================================== Prediction ==================================================================================

print(rec.shape, data.shape, indices.shape)
print(indices.columns)
print(data.columns)


x1,y1 = Recommendation('rotel corn',name_consin_sim)
x2,y2 = Recommendation('rotel corn',rec_consin_sim)
x3,y3 = Get_Recommendations(recom,'rotel corn', indices)

print(x1)
print(x2)
print(x3)

#============================================================================== End =================================================================