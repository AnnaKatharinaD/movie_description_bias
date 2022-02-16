import Bias_model

# config
csvpath = 'combined_descriptions.csv'
outputpath = 'movie_model_w2v'
clusteroutputpath = 'gender_clusters.txt'

'''
Train new model
'''
#Bias_model.label_clusters(['hello'], ['you'])

print('Training new model', csvpath)
Bias_model.TrainModel(csvpath,
                      'text',
                      outputname=outputpath,
                      fasttext=False,
                      window=15,
                      minf=20,
                      epochs=5,
                      ndim=200,
                      verbose=False
                      )
print('Training finished, saved ', outputpath)

'''
Find biased words
'''
print()
print('Finding biases...')

female = ['female', 'woman', 'girl', 'mother', 'daughter', 'sister', 'she', 'her', 'hers']
male = ['male', 'man', 'boy', 'son', 'father', 'brother', 'he', 'him', 'his']

other = ["non binary", "transgender", "trans", "transwoman", "transsexual"]

muslim = ["allah", "ramadan", "turban", "emir", "salaam", "sunni", "koran",
          "imam", "sultan", "prophet", "veil", "ayatollah", "shiite", "mosque", "islam", "sheik", "muslim", "muhammad",
          'islamic']
christian = ["baptism", "messiah", "catholicism", "resurrection", "christianity", "salvation", "protestant", "gospel",
             "trinity", "jesus", "christ", "christian", "cross", "catholic", "church"]
#Bias_model.getMostSimilarWords(outputpath, female, male)
[b1, b2] = Bias_model.GetTopMostBiasedWords(
    outputpath,  # model path
    200,  # topk biased words
    female,  # target set 1
    male,  # target set 2
    ['JJ', 'JJR', 'JJS'],  # nltk pos to be considered #'JJ','JJR','JJS',
    verbose=False)

print('Biased words towards ', female)
print([(b['word'], b['bias'], b['sent'], b['rank']) for b in b1])

print('Biased words towards ', male)
print([(b['word'], b['bias'], b['sent'], b['rank']) for b in b2])

'''
cluster words
'''
print()
print('Clustering words into concepts...')
[cl1, cl2] = Bias_model.Cluster(
    b1,  # set of words biased towards target set 1
    b2,  # set of words biased towards target set 2
    0.15,  # r 0.15
    10,  # repeat
    verbose=False)

print('Resulting clusters')
print('Clusters biased towards ', female)
for cluster in cl1:
    print([k['word'] for k in cluster])

print('Clusters biased towards ', male)
for cluster in cl2:
    print([k['word'] for k in cluster])

print('Printing to file...')
with open(clusteroutputpath, 'wt') as f:
    f.write('#Clusters for %s' % female)
    f.write('\n')
    for cluster in cl1:
        cluster_list = [k['word'] for k in cluster]
        for word in cluster_list:
            f.write(word + ' ')
        f.write('\n')
    f.write('\n')
    f.write('#Clusters for %s' % male)
    f.write('\n')
    for cluster in cl2:
        cluster_list = [k['word'] for k in cluster]
        for word in cluster_list:
            f.write(word + ' ')
        f.write('\n')
print()
print('*Finished')

print()
print('*Finished')