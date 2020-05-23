from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns

corpus = ['Time flies flies like an arrow.', 'Fruit flies like a banana.']
corpus1 = ['Time flies flies1 like an arrow.', 'Fruit flies like a banana.']
one_hot_vectorizer = CountVectorizer(binary=True)
#print(one_hot_vectorizer)
# print(one_hot_vectorizer.fit_transform(corpus))
# print(one_hot_vectorizer.get_feature_names())
one_hot = one_hot_vectorizer.fit_transform(corpus).toarray()
print(one_hot_vectorizer.get_feature_names())

#print(one_hot_vectorizer)
vocab = one_hot_vectorizer.get_feature_names()
print(one_hot_vectorizer)
print(one_hot)
print(vocab)

sns.heatmap(one_hot, annot=True, cbar=True, xticklabels=vocab, yticklabels=corpus)
