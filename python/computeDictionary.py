import pickle
from getDictionary import get_dictionary


meta = pickle.load(open('../data/traintest.pkl', 'rb'))
train_imagenames = meta['train_imagenames']

# -----fill in your implementation here --------
dictionaryRandom = get_dictionary(train_imagenames, 200, 500, "Random")
dictionaryHarris = get_dictionary(train_imagenames, 200, 500, "Harris")
pickle.dump(dictionaryRandom, open('../data/dictionaryRandom.pkl', 'wb'))
pickle.dump(dictionaryHarris, open('../data/dictionaryHarris.pkl', 'wb'))



# ----------------------------------------------



