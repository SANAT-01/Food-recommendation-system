import streamlit as st
import _pickle as cPickle
import numpy as np
import pandas as pd
import requests
from PIL import Image
import io
import random 

# Define a function to load the data and models
@st.cache_data()  # Cache the loaded data and models
def load_data_and_models():

    with open('recommender.pkl', 'rb') as fid:
        recom = cPickle.load(fid)

    indices = pd.read_csv('Images_links.csv')
    indices = indices.drop(['recipe_id'], axis=1)
    rating_matrix = pd.read_csv('rating_matrix.csv', index_col='recipe_id')
    name_consin_sim = np.load('name_consin_sim.npy')
    rec_consin_sim = np.load('rec_consin_sim.npy')
    rec = pd.read_csv('rec.csv')
    data = pd.read_csv('data.csv', index_col='Unnamed: 0')
    # rate = pd.read_csv('rate.csv', index_col='Unnamed: 0')

    # A small pre-processing
    indices = pd.merge(indices, rec, right_on='name', left_on='name')
    indices = indices.set_index('name')
    food_names = indices.index.to_list()

    return recom, indices, rating_matrix, name_consin_sim, rec_consin_sim, rec, data, food_names


def load_gui():
    mains = st.container()
    head = st.container()


    with st.sidebar:
        st.write('Side Bar')
        with st.expander("About"):
                st.write("""
                This is food recommender sysytem where the food will be recommended 
                 based on your mentioned choice. The recommendation are made using 
                 content and collaborative filtering .
                 \n \t Created by - Sanat Tudu""")
        add_radio = st.radio(
            "Veg/Non",
            ("Veg", "Non-Veg")
        )

    add_selectbox = st.sidebar.selectbox(
        "How would you like to be contacted?",
        ("Email", "Home phone", "Mobile phone")
    )
    

    tab1, tab2, tab3 = st.tabs(["Recommended", "Similar Named items", "People Like you"])
    tabs = [tab1, tab2, tab3]
    rows = []
    for i in range(number_of_tabs):
        sub = []
        for j in range(number_of_img):
            col1, col2 = tabs[i].container().columns(2)
            sub.append([col1, col2])
        rows.append(sub)

    # with tab1:
    #     st.write('Contained 1')

    # with tab2:
    #     st.write('Contained 2')

    # with tab3:
    #     st.write('Contained 3')
    return mains, head, tabs, rows

# Content-based Recommendation Function
def content_based_recommendation(sim, food_name, indices):
    re_list = []
    idxs = []
    ind = indices[indices.index == food_name]['index'][0]
    sim_score = list(enumerate(sim[ind]))
    sim_score = sorted(sim_score, key=lambda x: x[1], reverse=True)
    sim_score = sim_score[0:11]
    rec_indices = [i[0] for i in sim_score][1:]
    for i in rec_indices:
        re_list.append(indices[indices['index'] == i].index[0])
        idxs.append(indices[indices['index'] == i].recipe_id[0])

    return re_list, idxs

# Collaborative-based Recommendation Function
def collaborative_based_recommendation(recommender, food_name, indices, rating_matrix):
    user = indices[indices.index == food_name]
    recipe_id = user['recipe_id'][0]
    recipe_index = np.where(rating_matrix.index == recipe_id)[0][0]
    user_ratings = rating_matrix.iloc[recipe_index]
    reshaped = user_ratings.values.reshape(1, -1)
    distances, idx = recommender.kneighbors(reshaped, n_neighbors=11)
    nearest_neighbors_indices = rating_matrix.iloc[idx[0]].index[1:]

    names = []
    for idx in nearest_neighbors_indices:
        names.append(indices[indices.recipe_id == idx].index[0])

    return names, nearest_neighbors_indices

# To extract the name from the given indices
def extract_feats(recipe_idxs, data, indices):
    df = pd.DataFrame(columns=['0', 'name', 'recipe_id', 'rating', 'Veg/Non', 'description', 'review', 'img'])
    cnt = 0
    for idx in recipe_idxs:
        sub = {}
        indices_row = indices[indices.recipe_id == idx]
        sub['name'] = indices_row.index[0]
        sub['recipe_id'] = idx
        sub['img'] = indices_row['image_links'][0]
        data_row = data[data.recipe_id == idx].iloc[0]
        sub['review'] = data_row['review']
        sub['description'] = data_row['description']
        sub['Veg/Non'] = data_row['Veg/Non']
        # sub['ratings'] = rate[rate.name == sub['name']]['rating'] basically the rate csv file is save at wrong train model so
        sub['rating'] = round(np.random.uniform(3, 5), 1)

        df.loc[cnt] = sub
        cnt += 1
    # print(df.iloc[:, 0:4])
    return df

def plot_image(image_url, title, width=200):
    st.image(image_url, caption=title, width=width)

def head_img(image_url, head, data, food_name):
    col1, col2 = head.columns(2)
    row = data[data.name == food_name]
    # print(row.columns)
    with col1:
        col1.write(row.iloc[0]['name'].capitalize())
        col1.image(image_url, caption="Selected Image", width=320)

    rate = round(row['rating'].mean(), 1)
    col2.write('Ratings : ' + str(rate) + " " + "⭐"*int(round(rate,0)))
    col2.write((row['Veg/Non'].iloc[0]))
    col2.write('Descriptions : ' + row.iloc[0]['description'].capitalize())
    # sub = head.container()
    with col2.expander("Reviews"):
        for ids in range(len(row)):
            st.write(str(ids+1) +". "+ row.iloc[ids]['review'])
    with col2.expander("Nutrients"):
        st.write('Calories : ' + str(row.iloc[0]['calories']))
        st.write( 'Total Fat : ' + str(row.iloc[0]['total fat']))
        st.write( 'Sugar : ' + str(row.iloc[0]['sugar']))
        st.write( 'Sodium : ' + str(row.iloc[0]['sodium']))
        st.write( 'Protein : ' + str(row.iloc[0]['protein']))
        st.write( 'Saturated Fat : ' + str(row.iloc[0]['saturated fat']))
        st.write( 'Carbohydrates : ' + str(row.iloc[0]['carbohydrates']))

    with col2.expander("ingredients"):
        for ing in row.iloc[ids]['ingredients'].strip('][').split(', '):
            st.write(ing.strip("'"))

number_of_tabs = 3
number_of_img = 10

def main():
    
    cnt = 0
    # To show the image and details of that food which is searched or clicked 
    head_image = None
    head_url = None

    mains, head, tabs, rows = load_gui()
    
    mains.title(':blue[Food Recommendation System] 🍲')

    # Load data and models
    recom, indices, rating_matrix, name_consin_sim, rec_consin_sim, rec, data, food_names = load_data_and_models()
    food_names.sort()
    food_name = mains.selectbox("Select any Food Item of your choice", food_names)

    food_id = indices[indices.index == food_name]['recipe_id'][0]
    url = indices[indices.recipe_id == food_id]['image_links'][0]
    print(food_id, url)
    

    content_based_result_rec = content_based_recommendation(rec_consin_sim, food_name, indices)
    # mains.subheader("Content-Based Recommendation")
    desc1 = extract_feats(content_based_result_rec[1], data, indices)
    # mains.write(content_based_result_rec[0])

    content_based_result_name = content_based_recommendation(name_consin_sim, food_name, indices)
    # mains.subheader("Content-Based Name Recommendation")
    desc2 = extract_feats(content_based_result_name[1], data, indices)
    # mains.write(content_based_result_name[0])

    collaborative_based_result = collaborative_based_recommendation(recom, food_name, indices, rating_matrix)
    # mains.subheader("Collaborative-Based Recommendation")
    desc3 = extract_feats(collaborative_based_result[1], data, indices)
    # mains.write(collaborative_based_result[0])

    mains.write("-"*100)
    recommended_im_links = desc1.img
    
    for id,image_url in enumerate(recommended_im_links):
        with tabs[0]:
            rows[0][id][0].image(image_url, caption='Image', width=300)
            
            rows[0][id][1].write(desc1.iloc[id]['name'].capitalize())
            rows[0][id][1].write(desc1.iloc[id]['Veg/Non'])
            rates = round(data[data.name == desc1.iloc[id]['name']]['rating'].mean(), 1)
            rows[0][id][1].write("Ratings " + str(rates))
            rows[0][id][1].write("Description : ")
            rows[0][id][1].write(desc1.iloc[id]['description'][:100] + "....")
            # rows[0][id][1].button("More ", on_click=lambda img_url=image_url, name_food=desc1.iloc[id]['name']: head_img(img_url, head, data, name_food), key=cnt)
            if rows[0][id][1].button('More', cnt):
                head_image = desc1.iloc[id]['name']
                head_url = image_url
            cnt += 1

    name_recommended_im_links = desc2.img
    for id,image_url in enumerate(name_recommended_im_links):
        with tabs[1]:
            rows[1][id][0].image(image_url, caption='Image', width=300)
            
            rows[1][id][1].write(desc2.iloc[id]['name'].capitalize())
            rows[1][id][1].write(desc2.iloc[id]['Veg/Non'])
            rates = round(data[data.name == desc2.iloc[id]['name']]['rating'].mean(), 1)
            rows[1][id][1].write("Ratings " + str(rates))
            rows[1][id][1].write("Description : ")
            rows[1][id][1].write(desc2.iloc[id]['description'][:100] + "....")
            # rows[1][id][1].button("Show details", on_click=lambda img_url=image_url, name_food=desc2.iloc[id]['name']: head_img(img_url, head, data, name_food), key=cnt)
            if rows[1][id][1].button('More', cnt):
                head_image = desc2.iloc[id]['name']
                head_url = image_url
            cnt += 1
            
    colab_recommended_im_links = desc3.img
    for id,image_url in enumerate(colab_recommended_im_links):
        with tabs[2]:
            rows[2][id][0].image(image_url, caption='Image', width=300)
            
            rows[2][id][1].write(desc3.iloc[id]['name'].capitalize())
            rows[2][id][1].write(desc3.iloc[id]['Veg/Non'])
            rates = round(data[data.name == desc3.iloc[id]['name']]['rating'].mean(), 1)
            rows[2][id][1].write("Ratings " + str(rates))
            rows[2][id][1].write("Description : ")
            rows[2][id][1].write(desc3.iloc[id]['description'][:100] + "....")
            # rows[2][id][1].button("Show details", on_click=lambda img_url=image_url, name_food=desc1.iloc[id]['name']: head_img(img_url, head, data, name_food), key=cnt)
            if rows[2][id][1].button('More', cnt):
                head_image = desc3.iloc[id]['name']
                head_url = image_url
            cnt += 1
    
    if not head_image:
        head_img(url, head, data, food_name)
    else:
        head_img(head_url, head, data, head_image)

if __name__ == "__main__":
    main()