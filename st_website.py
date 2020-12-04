import streamlit as st
import pandas as pd
import numpy as np
from collections import deque




@st.cache
def getmovietitles():
    movietitles = pd.read_csv('C:\CPS842 Project\Dataset\movie_titles.csv', header=None, names=['ID', 'Year', 'Title'],
                              encoding='ISO-8859-1').set_index('ID')
    # create a dictioanry of movieTitles
    movieDict = movietitles.to_dict()
    return movieDict
# print(movieDict)

movieDict = getmovietitles()
NUMBER_OF_MOVIES = len(movieDict["Title"])
print(f"Number of movies is: {NUMBER_OF_MOVIES}")



@st.cache
def builddataset():
    # import movietitles

    print("IN BUILD DATASET")
    ##This next part loads teh combined data 1 into a pd.df
    df_raw = pd.read_csv('C:\CPS842 Project\Dataset\combined_data_1.txt', header=None, names=['User', 'Rating', 'Date'],
                         usecols=[0, 1, 2])

    # Find empty rows to slice dataframe for each movie
    tmp_movies = df_raw[df_raw['Rating'].isna()]['User'].reset_index()
    movie_indices = [[index, int(movie[:-1])] for index, movie in tmp_movies.values]

    # Shift the movie_indices by one to get start and endpoints of all movies
    shifted_movie_indices = deque(movie_indices)
    shifted_movie_indices.rotate(-1)

    # Gather all dataframes
    user_data = []

    # Iterate over all movies
    for [df_id_1, movie_id], [df_id_2, next_movie_id] in zip(movie_indices, shifted_movie_indices):

        # Check if it is the last movie in the file
        if df_id_1 < df_id_2:
            tmp_df = df_raw.loc[df_id_1 + 1:df_id_2 - 1].copy()
        else:
            tmp_df = df_raw.loc[df_id_1 + 1:].copy()

        # Create movie_id column
        tmp_df['Movie'] = movie_id

        # Append dataframe to list
        user_data.append(tmp_df)

    # Combine all dataframes
    rating = pd.concat(user_data)
    del user_data, df_raw, tmp_movies, tmp_df, shifted_movie_indices, movie_indices, df_id_1, movie_id, df_id_2, next_movie_id

    #print('Shape User-Ratings:\t{}'.format(rating.shape))

    rating["Rating"] = rating["Rating"].astype(np.int8)
    rating["User"] = rating["User"].astype(np.int32)
    rating["Movie"] = rating["Movie"].astype(np.int16)

    del rating["Date"]  # remove date as this is date is irrelevant

    def changeName(x):
        return movieDict['Title'][x]

    rating.Movie = rating.Movie.map(changeName) #change movie id into movie titles

    #sort dataframe by most active users
    rating['freq'] = rating['User'].map(rating['User'].value_counts())
    rating = rating.sort_values('freq', ascending=False)

    #Create a pivot table
    M = rating[:3000000].pivot_table(index=["User"], columns=["Movie"], values="Rating").fillna(0)
    del rating #remove rating from memory
    return M


M = builddataset()

#now I write algorithm for recommendations
@st.cache
def sim(m, n):
    ms = m - m.mean() #This uses pearson's R
    ns = n - n.mean()
    sim = np.sum(ms * ns)/np.sqrt(np.sum(ms ** 2) * np.sum(ns ** 2))
    return sim

def top10(name): #for item to item
    rec = []
    for title in M.columns:
        if title == name:  # if title is same as arg, continue
            continue

        r = sim(M[name], M[title])  # calculate r value
        if np.isnan(r):  # if r value doesnt make sense, move on
            continue
        else:
            rec.append((title, r))  # append to list

    rec.sort(key=lambda n: n[1], reverse=True)
    return rec[:10]

def top10user(name):
    rec = []
    movies = {}

    for user in M.index:
        if name == user:  # if title is same as arg, continue
            continue

        r = sim(M.loc[name], M.loc[user])  # calculate r value
        if np.isnan(r):  # if r value doesnt make sense, move on
            continue
        else:
            rec.append((user, r))  # append to lis

    rec.sort(key=lambda n: n[1], reverse=True)

    st.write(f"# Other users similar to user '{name}' include")

    for i in rec[:10]:
        st.write(f"{i[0]} ({round(i[1]*100,2)}% match)")
        count = 0
        for rating in M.loc[i[0]]:
            if rating != 0:
                if M.columns[count] in movies:
                    movies[M.columns[count]] += rating
                else:
                    movies[M.columns[count]] = rating

            count += 1

    sorted_movies = sorted(movies.items(), key=lambda x: x[1], reverse=True)


    st.write(f"## Movies other users similar to user '{name}' like include")

    for i in sorted_movies[:10]:
        st.write(f"{i[0]}  \n\t Average score amongst similar users: {i[1]/10}\n")



    return rec[:10]

def personal_rec(s):
    series = M.iloc[0].copy()

    for key, value in series.items():
        series[key] = s[key]

    rec = []
    movies = {}

    for user in M.index:
        r = sim(series, M.loc[user])  # calculate r value
        if np.isnan(r):  # if r value doesnt make sense, move on
            continue
        else:
            rec.append((user, r))  # append to list

    rec.sort(key=lambda n: n[1], reverse=True)

    for i in rec[:10]: #movie recc
        count = 0
        for rating in M.loc[i[0]]:
            if rating != 0:
                if M.columns[count] in movies:
                    movies[M.columns[count]] += rating
                else:
                    movies[M.columns[count]] = rating

            count += 1

    sorted_movies = sorted(movies.items(), key=lambda x: x[1], reverse=True)[:10]

    st.write("# Other users similar to you include")

    for i in rec[:10]:
        st.write(f"{i[0]} ({round(i[1]*100,2)}% match)")

    st.write("## Movies other users similar to you like include")
    st.text("* in descending order")


    for i in sorted_movies:
        st.write(f"{i[0]}  \n\t Average score amongst similar users: {i[1]/10}\n")

#website code

st.sidebar.header("Rate Your Movies")
def user_ratings():
    scores = {}
    st.sidebar.write(" # Rate movies from the dataset here for personalized recommendations")
    for k, v in movieDict["Title"].items():
        scores[v] = st.sidebar.slider(f"{k}. Select a rating for movie: {v}", 0, 5, 0, key=k)

    return scores

scores = user_ratings()

def update():
    rated = False
    for k, v in scores.items():
        if v != 0:
            rated = True
            st.write(f"# Since you watched {k}")

            list = top10(k)

            st.write(f"## You might like: ")
            count = 1

            for i in list:
                st.write(f"{count}. {i[0]} ({round(i[1]*100, 2)}% match)")
                count+=1

    if rated:
        print("in personal rec")
        personal_rec(scores)

    st.write("# Here is a sample of the dataframe used to make recommendations")
    st.write("## The shape of the data frame used for recommendations is: 17,770 movies by 3,000,000 users")

    st.dataframe(M[:100][:100])

top10user(769)

update()





