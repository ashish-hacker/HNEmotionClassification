
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re , string , random
import pickle

data = pd.read_csv('text_emotion.csv')


from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import FreqDist, classify, NaiveBayesClassifier


def remove_noise(tweet_tokens, stop_words = ()):

    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens


def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token


def get_words_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)


stop_words = stopwords.words('english')


neutral_array = [i.split(' ') for i in np.array(data.content[data.sentiment == 'neutral'])]
worry_array =[i.split(' ') for i in  np.array(data.content[data.sentiment == 'worry'])]
happy_array = [ i.split(' ') for i in np.array(data.content[data.sentiment == 'happiness'])]
sad_array = [ i.split(' ') for i in np.array(data.content[data.sentiment == 'sadness'])]
love_array = [ i.split(' ') for i in np.array(data.content[data.sentiment == 'love'])]
surprise_array = [ i.split(' ') for i in np.array(data.content[data.sentiment == 'surprise'])]
fun_array = [ i.split(' ') for i in np.array(data.content[data.sentiment == 'fun'])]
relief_array = [ i.split(' ') for i in np.array(data.content[data.sentiment == 'relief'])]
hate_array = [ i.split(' ') for i in np.array(data.content[data.sentiment == 'hate'])]
empty_array = [i.split(' ') for i in np.array(data.content[data.sentiment == 'empty'])]
enthu_array = [ i.split(' ') for i in np.array(data.content[data.sentiment == 'enthusiasm'])]
bore_array = [ i.split(' ') for i in np.array(data.content[data.sentiment == 'boredom'])]
anger_array =[ i.split(' ') for i in np.array(data.content[data.sentiment == 'anger'])]

n_arrays = [neutral_array, worry_array, happy_array, sad_array, love_array, surprise_array, fun_array, relief_array, hate_array, empty_array, enthu_array, anger_array, bore_array]


neutral_dat  = []
worry_dat  = []
happ_dat = []
sad_dat = []
love_dat = []   
surprise_dat = [] 
fun_dat = [] 
relief_dat = [] 
hate_dat = []
empty_dat = []
enth_dat = []
bore_dat= []
anger_dat = []

arrays = [neutral_dat, worry_dat, happ_dat, sad_dat, love_dat, surprise_dat, fun_dat, relief_dat, hate_dat, empty_dat, enth_dat, anger_dat, bore_dat]

for i, k in zip(n_arrays, arrays):
    for j in i:
        j = [x for x in j if x!='']
        k.append(remove_noise(j,stop_words))
        



all_pos_words = get_all_words(neutral_dat)


freq_dist_pos = FreqDist(all_pos_words)
#print(freq_dist_pos.most_common(10))


happy_model = get_words_for_model(happ_dat)
neutral_model = get_words_for_model(neutral_dat)
worry_model = get_words_for_model(worry_dat)
sad_model = get_words_for_model(sad_dat)
love_model = get_words_for_model(love_dat)
surprise_model = get_words_for_model(surprise_dat)
fun_model = get_words_for_model(fun_dat)
relief_model = get_words_for_model(relief_dat)
hate_model = get_words_for_model(hate_dat)
empty_model = get_words_for_model(empty_dat)
enth_model = get_words_for_model(enth_dat)
bore_model = get_words_for_model(bore_dat)
anger_model = get_words_for_model(anger_dat)


happy_dataset = [(tweet_dict, "happy")
                         for tweet_dict in happy_model]
neutral_dataset = [(tweet_dict, "neutral")
                         for tweet_dict in neutral_model]
worry_dataset = [(tweet_dict, "worry")
                         for tweet_dict in worry_model]
sad_dataset = [(tweet_dict, "sad")
                         for tweet_dict in sad_model]
love_dataset = [(tweet_dict, "love")
                         for tweet_dict in love_model]
hate_dataset = [(tweet_dict, "hate")
                         for tweet_dict in hate_model]
surprise_dataset = [(tweet_dict, "surprise")
                         for tweet_dict in surprise_model]
fun_dataset = [(tweet_dict, "fun")
                         for tweet_dict in fun_model]
relief_dataset = [(tweet_dict, "relief")
                         for tweet_dict in relief_model]
empty_dataset = [(tweet_dict, "empty")
                         for tweet_dict in empty_model]
enth_dataset = [(tweet_dict, "enthusiastic")
                         for tweet_dict in enth_model]
bore_dataset = [(tweet_dict, "boredom")
                         for tweet_dict in bore_model]
anger_dataset = [(tweet_dict, "angry")
                         for tweet_dict in anger_model]


dataset = anger_dataset + bore_dataset + enth_dataset + empty_dataset + relief_dataset + fun_dataset + surprise_dataset + hate_dataset + love_dataset + sad_dataset + worry_dataset + neutral_dataset + happy_dataset


random.shuffle(dataset)



train_data = dataset[:35000]
test_data = dataset[35000:]


classifier = NaiveBayesClassifier.train(train_data)


custom_sentence = "Let's party!"
custom_tokens = remove_noise(word_tokenize(custom_sentence))

result = classifier.classify(dict([token, True] for token in custom_tokens))
print('Sentence:',custom_sentence,'\nEmotion:', result)

filename = 'nlp_model.pkl'
pickle.dump(classifier, open(filename, 'wb'))




