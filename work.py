import numpy as np
from collections import defaultdict
from math import log
np.seterr(divide = 'ignore')

class NaiveBayes():
    def __init__(self, stopwords, use_tf_idf=True):
        self.stopwords = stopwords
        self.use_tf_idf = use_tf_idf
        self.vocab = {}
        self.class_counts = defaultdict(int)
        self.word_counts = {}
        self.class_probs = {}

    def load_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = f.read().splitlines()
            return data

    def preprocess(self, data):
        preprocessed_data = []
        for sentence in data:
            sentence = sentence.lower()
            sentence = ''.join(char for char in sentence if char.isalnum() or char.isspace())
            words = sentence.split()
            words = [word for word in words if word not in self.stopwords]
            preprocessed_data.append(words)
        return preprocessed_data

    def split_data(self, data, split_ratio=0.2):
        np.random.shuffle(data)
        split = int((1 - split_ratio) * len(data))
        train_data = data[:split]
        test_data = data[split:]
        return train_data, test_data

    def train(self, train_data):
        if self.use_tf_idf:
            self.calculate_tf_idf(train_data)
        else:
            self.calculate_bow(train_data)
        self.calculate_class_probs(train_data)

    def calculate_class_probs(self, data):
        for sentence in data:
            label = sentence[0]
            self.class_counts[label] += 1
            for word in sentence:
                if word not in self.word_counts:
                    self.word_counts[word] = defaultdict(int)
                self.word_counts[word][label] += 1
        total_count = sum(self.class_counts.values())
        for label, count in self.class_counts.items():
            self.class_probs[label] = count / total_count

    def calculate_tf_idf(self, train_data):
        doc_freqs = defaultdict(int)
        for sentence in train_data:
            seen_words = set()
            for word in sentence[1:]:
                if word not in seen_words:
                    doc_freqs[word] += 1
                    seen_words.add(word)
        for word, doc_freq in doc_freqs.items():
            idf = log(len(train_data) / doc_freq)
            self.vocab[word] = len(self.vocab)
            for label in self.class_counts.keys():
                self.word_counts[word][label] *= idf

    def calculate_bow(self, train_data):
        for sentence in train_data:
            for word in sentence:
                if word not in self.vocab:
                    self.vocab[word] = len(self.vocab)
                if word not in self.word_counts:
                    self.word_counts[word] = defaultdict(int)
                self.word_counts[word][sentence[0]] += 1

    def predict(self, text):
        predictions = []
        for doc in text:
            log_probs = {}
            for label in self.class_probs.keys():
                clad_probs = []
                for i in doc[1:]:
                    if i not in self.vocab:
                        continue
                    clad_probs.append(self.word_counts[i][label]/len(doc[1:]))
                log_probs[label] = np.sum(np.log(clad_probs))
            predicted_label = max(log_probs, key=log_probs.get)
            predictions.append(predicted_label)
        return predictions


    def evaluate(self, pred, real):
        acc = sum([1 for i in range(len(real)) if real[i][0] == pred[i]]) / len(real)
        tp, fp, tn, fn = 0, 0, 0, 0
        for i in range(len(real)):
            if real[i][0] == 'spam' and pred[i] == 'spam':
                tp += 1
            elif real[i][0] == 'ham' and pred[i] == 'spam':
                fp += 1
            elif real[i][0] == 'ham' and pred[i] == 'ham':
                tn += 1
            elif real[i][0] == 'spam' and pred[i] == 'ham':
                fn += 1
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1 = 2 * prec * rec / (prec + rec)
        return {'accuracy': acc, 'precision': prec, 'recall': rec, 'F1 score': f1}


stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',  'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her',  'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs',  'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',  'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',  'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',  'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with',  'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',  'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',  'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where',  'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',  'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',  'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']
filename = "SMSSpamCollection1"
# create a language model
nb=NaiveBayes(stop_words, use_tf_idf=True)
# load the dataset
nb_data = nb.load_file(filename)
# preprocess the data
nb_data_preprocessed = nb.preprocess(nb_data)
# split the data into training and testing sets
nb_train , nb_test = nb.split_data(nb_data_preprocessed)
# train the model on the training set
nb.train(nb_train)
# make predictions on the testing set
predictions = nb.predict(nb_test)
# print the accuracy,precision,recall,f1score scores
evaluation = nb.evaluate(predictions,nb_test)
print(evaluation)

print(predictions)
# print(predictions[0])
print(nb_test[0])


