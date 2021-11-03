import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from flask import Flask,render_template,request


def recommender(user_values):
    songs = pd.read_csv('data.csv')
    data = songs.drop_duplicates()
    data = data.drop(data[data['duration_ms'] > 600000].index.values)
    data = data.drop(data[data['duration_ms'] < 60000].index.values)
    columns = ['acousticness','danceability','duration_ms','energy','explicit','instrumentalness','key',
               'liveness','loudness','mode','popularity','speechiness','tempo','valence','year']
    col_dict = dict(zip(columns,user_values))
    user_song = pd.DataFrame.from_dict([col_dict])
    user_song_ = pd.DataFrame(np.repeat(user_song.values,len(data[columns]),axis=0),columns=user_song.columns)
    scaler = StandardScaler()
    scaler.fit(data[columns])
    data_std = scaler.transform(data[columns])
    user_song_std = scaler.transform(user_song_)
    data['euc'] = ((data_std - user_song_std)**2).sum(axis=1)
    return list(data.sort_values(by='euc').head(10)['name'].values)


app = Flask(__name__)

@app.route("/")
@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/song_recommender_home')
def song_recommender_home():
    return render_template('song_recommender_home.html')

@app.route('/song_recommender_result',methods=['POST'])
def song_recommender_result():
    if request.method=='POST':
        user_values = [float(request.form['acousticness']),float(request.form['danceability']),int(request.form['duration_ms']),
                       float(request.form['energy']),int(request.form['explicit']),float(request.form['instrumentalness']),
                       int(request.form['key']),float(request.form['liveness']),float(request.form['loudness']),
                       int(request.form['mode']),int(request.form['popularity']),float(request.form['speechiness']),
                       float(request.form['tempo']),float(request.form['valence']),int(request.form['year'])]
        recom_songs = recommender(user_values)
        
        return render_template('song_recommender_result.html',recom_songs=recom_songs)
    

if __name__ == '__main__':
    app.run()