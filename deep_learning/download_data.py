import requests

mushroom_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data'

with open('data/mushroom.csv', 'w') as f:
    f.write(requests.get(mushroom_url).text)
