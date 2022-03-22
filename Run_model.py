import Bias_model

csvpath = 'combined_descriptions.csv'
outputpath = 'movie_model_w2v'
clusteroutputpath = 'clusters/adjective_clusters.txt'

#noun_list = ['NN', 'NNP', 'NNS'] #for nltk
adjective_list = ['JJ','JJR','JJS'] #for nltk
noun_list = ['NOUN'] #for spacy

'''
Train new model
'''

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

#Bias_model.getMostSimilarWords(outputpath, female, male)
[b1, b2] = Bias_model.GetTopMostBiasedWords(
    outputpath,  # model path
    100,  # topk biased words
    female,  # target set 1
    male,  # target set 2
    adjective_list, #pos to be considered
    use_spacy=False,
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
    0.3,  # r 0.15
    10,  # repeat
    verbose=False)

print('Resulting clusters')
print('Clusters biased towards ', female)
for cluster in cl1:
    print([k['word'] for k in cluster])

print('Clusters biased towards ', male)
for cluster in cl2:
    print([k['word'] for k in cluster])

'''
print words in clusters to file
'''
#Bias_model.print_to_file(clusteroutputpath, male, female, cl1, cl2)
