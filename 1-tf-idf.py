from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import seaborn as sns

corpus = ['Time flies flies like an arrow.', 'Fruit flies like a banana.']
corpus1 = ['Time flies flies1 like an arrow.', 'Fruit flies like a banana.']
#one_hot_vectorizer = CountVectorizer(binary=True)
tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(corpus).toarray()
vocab = tfidf_vectorizer.get_feature_names()
print(one_hot_vectorizer)
print(one_hot)
print(vocab)

sns.heatmap(tfidf, annot=True, cbar=True, xticklabels=vocab, yticklabels=corpus)
