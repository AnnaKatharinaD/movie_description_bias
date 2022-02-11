from gensim.models import Word2Vec as wv
import pandas as pd
import re
import spacy
from gensim.models.phrases import Phrases, Phraser
nlp = spacy.load('en_core_web_sm', disable = ['ner', 'parser'])

def cleaning(doc):
    # Lemmatizes and removes stopwords
    # doc needs to be a spacy Doc object
    txt = [token.text for token in doc]
    # Word2Vec uses context words to learn the vector representation of a target word,
    # if a sentence is only one or two words long,
    # the benefit for the training is very small
    return ' '.join(txt)

def cosine_distance (model, word, target_list , num) :
    cosine_dict ={}
    word_list = []
    a = model[word]
    for item in target_list :
        if item != word :
            b = model [item]
            cos_sim = dot(a, b)/(norm(a)*norm(b))
            cosine_dict[item] = cos_sim
    dist_sort=sorted(cosine_dict.items(), key=lambda dist: dist[1],reverse = True) ## in Descedning order
    for item in dist_sort:
        word_list.append((item[0], item[1]))
    return word_list[0:num]

df = pd.read_csv('clean_movie_and_pg.csv')
#df['text'] = [' '.join((descr,plot)) for descr,plot in zip(df['description'], df['plot'])]
df['text'] = df['description'] + df['plot']
brief_cleaning = (re.sub("[^A-Za-z']+", ' ', str(row)).lower() for row in df['text'])
txt = [cleaning(doc) for doc in nlp.pipe(brief_cleaning, batch_size=5000, n_threads=-1)]

df_clean = pd.DataFrame({'clean': txt})
df_clean = df_clean.dropna().drop_duplicates()

sent = [row.split() for row in df_clean['clean']]
phrases = Phrases(sent, min_count=20, progress_per=1000)
bigram = Phraser(phrases)
sentences = bigram[sent]
w2v_model = wv(min_count=10,
                     window=2,
                     workers = 3)

w2v_model.build_vocab(sentences, progress_per=1000)
print(w2v_model.wv.most_similar(positive=['sister', 'female', 'woman', 'girl',
'daughter', 'she', 'her']))
#df1 = df.apply(lambda x: '*'.join(x.astype(str)), axis=1)
#df_clean = pd.DataFrame({'clean': df1})

#sent = [row.split('*') for row in df_clean['clean']]

#model = wv(sent, min_count=1,workers=3, window =3, sg = 1)
#print(model.wv.most_similar('severe')[:5])