import sys
import pandas as pd
import numpy as np
import lightgbm as lgb

# Loading Data

train = pd.read_csv('E:/MAFM/MAFS6010S/CourseProject/train.csv', dtype = {'msno':'category', 'song_id':'category', 'source_system_tab':'category', 'source_screen_name':'category', 'source_type':'category'})
test = pd.read_csv('E:/MAFM/MAFS6010S/CourseProject/test.csv', dtype = {'msno':'category', 'song_id':'category', 'source_system_tab':'category', 'source_screen_name':'category', 'source_type':'category'})
songs = pd.read_csv('E:/MAFM/MAFS6010S/CourseProject/songs.csv', dtype = {'song_id':'category', 'artist_name':'category', 'composer':'category', 'lyricist':'category', 'language':'category'})
song_extra_info = pd.read_csv('E:/MAFM/MAFS6010S/CourseProject/song_extra_info.csv')
members = pd.read_csv('E:/MAFM/MAFS6010S/CourseProject/members.csv', dtype = {'msno':'category', 'city':'category', 'gender':'category', 'registered_via':'category'})

# Processing Data

def isrc_region(isrc):
    if type(isrc) == str:
        return isrc[0:2]
    else:
        return np.nan

def isrc_year(isrc):
    if type(isrc) == str:
        if int(isrc[5:7]) > 17:
            return 1900 + int(isrc[5:7])
        else:
            return 2000 + int(isrc[5:7])
    else:
        return np.nan

song_extra_info['region_code'] = song_extra_info['isrc'].apply(isrc_region).astype('category')
song_extra_info['year_of_reference'] = song_extra_info['isrc'].apply(isrc_year).astype('category')
songs = songs.merge(song_extra_info[['song_id', 'region_code', 'year_of_reference']], on = 'song_id', how = 'left')

def time_split(time, time_unit):
    assert time_unit in ['year', 'month', 'date'], 'The unit is not supported.'
    if type(time) == int:
        if time_unit == 'year':
            return time // 10000
        elif time_unit == 'month':
            return (time % 10000) // 100
        else:
            return time % 100
    else:
        return np.nan

for time_unit in ['year', 'month', 'date']:
    members['registration_{}'.format(time_unit)] = members['registration_init_time'].apply(time_split, args = (time_unit,)).astype('category')
    members['expiration_{}'.format(time_unit)] = members['expiration_date'].apply(time_split, args = (time_unit,)).astype('category')

train = train.merge(songs, on = 'song_id', how = 'left')
test = test.merge(songs, on = 'song_id', how = 'left')

members = members.drop(['registration_init_time', 'expiration_date'], axis = 1)
train = train.merge(members, on = 'msno', how = 'left')
test = test.merge(members, on = 'msno', how = 'left')

for col in train.columns:
    if train[col].dtype == object:
        train[col] = train[col].astype('category')
        test[col] = test[col].astype('category')

# Training with LightGBM Model

X = train.drop(['target', 'msno', 'song_id'], axis = 1)
y = train['target'].values
train_data = lgb.Dataset(X, y)
parameters = {'num_leaves':73, 'max_depth':7, 'objective':'binary', 'metric':'auc'}
model = lgb.train(parameters, train_data, num_boost_round = 100, valid_sets = [train_data], verbose_eval = 5)

# Making prediction

test_result = pd.DataFrame()
test_result['id'] = test['id']
X_test = test.drop(['id', 'msno', 'song_id'], axis = 1)
test_result['target'] = model.predict(X_test)
test_result.to_csv('test_result.csv', index = False)