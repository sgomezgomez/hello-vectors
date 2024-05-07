##############################################
# Packages and dependencies
##############################################
import pdb
import pickle
import pandas as pd
import random
import string
import time
import numpy as np
import re
from nltk.corpus import stopwords, twitter_samples
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from os import getcwd

##############################################
## Helper functions
##############################################
def get_dict(file_name):
    """
    This function returns the english to french dictionary given a file where the each column corresponds to a word.
    Check out the files this function takes in your workspace.
    """
    my_file = pd.read_csv(file_name, delimiter=' ')
    etof = {}  # the english to french dictionary to be returned
    for i in range(len(my_file)):
        # indexing into the rows.
        en = my_file.loc[i][0]
        fr = my_file.loc[i][1]
        etof[en] = fr

    return etof

def process_tweet(tweet):
    '''
    Input:
        tweet: a string containing a tweet
    Output:
        tweets_clean: a list of words containing the processed tweet

    '''
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)
    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and  # remove stopwords
            word not in string.punctuation):  # remove punctuation
            # tweets_clean.append(word)
            stem_word = stemmer.stem(word)  # stemming word
            tweets_clean.append(stem_word)

    return tweets_clean

##############################################
## Main translator class
##############################################
class EnglishFrenchTranslator:

    def __init__(self):
        self.en_embeddings = pickle.load(open("./data/en_embeddings.p", "rb")) # English embeddings
        self.fr_embeddings = pickle.load(open("./data/fr_embeddings.p", "rb")) # French embeddings
        # Initialize transformation matrix R from English to French vector space embeddings
        # Assumes en-fr embeddings have the same dimensions
        n = self.en_embeddings[random.sample(list(self.en_embeddings.keys()), 1)[0]].shape[0]
        self.R = np.random.rand(n, n)
        pass

    def get_embeddings(self, en_fr):
        """
        Input:
            en_fr: English to French dictionary
        Output: 
            X: a matrix where the columns are the English embeddings.
            Y: a matrix where the columns correspong to the French embeddings.
        """
        # Initialize X and Y as empty lists
        X, Y = [], []

        # loop through all english, french word pairs in the english french dictionary
        for en_word, fr_word in en_fr.items():
            # check that the french word has an embedding and that the english word has an embedding
            if (fr_word in self.fr_embeddings.keys()) and (en_word in self.en_embeddings.keys()):
                # # add the english embedding to the array
                X.append(self.en_embeddings[en_word])
                # add the french embedding to the array
                Y.append(self.fr_embeddings[fr_word])
        
        # Transform into np arrays
        X, Y = np.array(X), np.array(Y)

        return X, Y

    def compute_loss(self, X, Y):
        '''
        Inputs: 
            X: a matrix of dimension (m,n) where the columns are the English embeddings.
            Y: a matrix of dimension (m,n) where the columns correspong to the French embeddings.
        Outputs:
            L: The value of the loss function for given X, Y and the model's R.
        '''

        # The loss function will be squared Frobenius norm of the difference between matrix and its approximation, divided by the number of training examples 𝑚
        loss = np.sum((np.dot(X, self.R) - Y)**2) / X.shape[0]

        return loss
    
    def compute_gradient(self, X, Y):
        '''
        Inputs: 
            X: a matrix of dimension (m,n) where the columns are the English embeddings.
            Y: a matrix of dimension (m,n) where the columns correspong to the French embeddings.
        Outputs:
            gradient: a (n, n) gradient of the loss function L for given X, Y and the model's R.
        '''
        # gradient is X^T(XR - Y) * 2/m   
        gradient = (2/X.shape[0]) * np.dot(X.T, (np.dot(X, self.R) - Y))
        
        return gradient
    
    def train(self, X, Y, train_steps=100, learning_rate=0.0003, verbose=True):
        '''
        Inputs:
            X: a matrix of dimension (m,n) where the columns are the English embeddings.
            Y: a matrix of dimension (m,n) where the columns correspong to the French embeddings.
            train_steps: positive int - describes how many steps will gradient descent algorithm do.
            learning_rate: positive float - describes how big steps will  gradient descent algorithm do.
        Outputs: No outputs. The function trains the projection matrix R
        '''
        # Train steps
        for i in range(train_steps):
            if verbose and i % 25 == 0:
                print(f"loss at iteration {i} is: {self.compute_loss(X, Y):.4f}")

            # update R by subtracting the learning rate times gradient
            self.R -= self.compute_gradient(X, Y) * learning_rate
        pass

    def cosine_similarity(self, word1_emb, word2_emb):
        '''
        Input:
            word1_emb: Numpy array which corresponds to a word vector
            word2_emb: Numpy array which corresponds to a word vector
        Output:
            cos: Scalar number representing the cosine similarity between word1_emb and word2_emb.
        '''
        if len(word1_emb.shape) == 1: # If word1_emb is just a vector, we get the norm
            cos = np.dot(word1_emb, word2_emb) / (np.linalg.norm(word1_emb) * np.linalg.norm(word2_emb)) 
        else: # If word_1_emb is a matrix, then compute the norms of the word vectors of the matrix (norm of each row)
            epsilon = 1.0e-9 # to avoid division by 0
            cos = np.dot(word1_emb, word2_emb) / (np.linalg.norm(word1_emb, axis=1) * np.linalg.norm(word2_emb) + epsilon)

        return cos
    
    def nearest_neighbor(self, v, candidates, k=1):
        """
        Input:
        - v, the vector to find the nearest neighbor for
        - candidates: a matrix with vectors where we will find the neighbors. Each row is a candidate
        - k: top k nearest neighbors to find
        Output:
        - knn: the top k closest vectors in sorted form
        """
        # Initialize similarity as an empty list
        similarity = []

        # for each candidate vector...
        for row in candidates:
            # append the similarity to the list
            similarity.append(self.cosine_similarity(v, row))

        # Sort the similarity list and get the indices of the reversed sorted list    
        sorted_ids = np.argsort(similarity)[::-1]
        # get the indices of the k most similar candidate vectors
        knn = candidates[sorted_ids[:k]]

        return knn
    
    def evaluate(self, X, Y):
        '''
        Input:
            X: a matrix where the columns are the English embeddings.
            Y: a matrix where the columns correspong to the French embeddings.
        Output:
            accuracy: for the English to French capitals
        '''
        # The prediction is X times R
        pred = np.dot(X, self.R)

        # initialize the number correct to zero
        num_correct = 0

        # loop through each row in pred (each transformed embedding)
        for i in range(len(pred)):
            # increment count if the index of the nearest neighbor equals the row of i... \
            #print(pred[i])
            #print(self.nearest_neighbor(pred[i], Y, 1)[0])
            if (Y[i] == self.nearest_neighbor(pred[i], Y, 1)[0]).any(): num_correct += 1

        # accuracy is the number correct divided by the number of rows in 'pred' (also number of rows in X)
        accuracy = num_correct / pred.shape[0]

        return accuracy

##############################################
## Main Bag or Words with Locality Sensitive Hashing (LSH) class
##############################################
class BagofWordsLSH:

    def __init__(self, n_planes, n_universes):
        self.en_embeddings = pickle.load(open("./data/en_embeddings.p", "rb")) # English embeddings
        self.n_universes = n_universes
        n = self.en_embeddings[random.sample(list(self.en_embeddings.keys()), 1)[0]].shape[0]
        self.planes = [np.random.normal(size=(n, n_planes)) for _ in range(self.n_universes)]
        
        pass

    def get_doc_embedding(self, tweet):
        '''
        Input:
            - tweet: a string
        Output:
            - doc_embedding: sum of all word embeddings in the tweet
        '''
        # Initialize embedding
        n = self.en_embeddings[random.sample(list(self.en_embeddings.keys()), 1)[0]].shape[0]
        doc_embedding = np.zeros(n)

        # process the document into a list of words (process the tweet)
        for word in process_tweet(tweet):
            # add the word embedding to the running total for the document embedding
            doc_embedding += (self.en_embeddings[word] if word in self.en_embeddings.keys() else 0)
        
        return doc_embedding
    
    def get_doc_vecs(self, all_docs):
        '''
        Input:
            - all_docs: list of strings - all tweets in our dataset.
        Output:
            - doc_vec: matrix of tweet embeddings.
            - ind2Doc_dict: dictionary with indices of tweets in vecs as keys and their embeddings as the values.
        '''
        # the dictionary's key is an index (integer) that identifies a specific tweet
        # the value is the document embedding for that document
        ind2Doc_dict = {}

        # Initialize document vector as an empty list
        doc_vec = []

        for i, doc in enumerate(all_docs):
            # save the document embedding into the ind2Tweet dictionary at index i
            ind2Doc_dict[i] = self.get_doc_embedding(doc)
            # append the document embedding to the list of document vectors
            doc_vec.append(ind2Doc_dict[i])

        # convert the list of document vectors into a 2D array (each row is a document vector)
        doc_vec = np.vstack(doc_vec)

        return doc_vec, ind2Doc_dict
    
    def cosine_similarity(self, word1_emb, word2_emb):
        '''
        Input:
            word1_emb: Numpy array which corresponds to a word vector
            word2_emb: Numpy array which corresponds to a word vector
        Output:
            cos: Scalar number representing the cosine similarity between word1_emb and word2_emb.
        '''
        if len(word1_emb.shape) == 1: # If word1_emb is just a vector, we get the norm
            cos = np.dot(word1_emb, word2_emb) / (np.linalg.norm(word1_emb) * np.linalg.norm(word2_emb)) 
        else: # If word_1_emb is a matrix, then compute the norms of the word vectors of the matrix (norm of each row)
            epsilon = 1.0e-9 # to avoid division by 0
            cos = np.dot(word1_emb, word2_emb) / (np.linalg.norm(word1_emb, axis=1) * np.linalg.norm(word2_emb) + epsilon)

        return cos
    
    def hash_value_of_vector(self, v, planes_idx):
        """Create a hash for a vector; hash_id says which random hash to use.
        Input:
            - v:  vector of tweet. It's dimension is (1, N_DIMS)
            - planes_idx: index of the planes list to use
        Output:
            - res: a number which is used as a hash for your vector

        """
        # for the set of planes,
        # calculate the dot product between the vector and the matrix containing the planes
        # remember that planes has shape (300, 10)
        # The dot product will have the shape (1,10)         
        # get the sign of the dot product (1,10) shaped vector
        sign_of_dot_product = np.sign(np.dot(v, self.planes[planes_idx]))

        # set h to be false (equivalent to 0 when used in operations) if the sign is negative,
        # and true (equivalent to 1) if the sign is positive (1,10) shaped vector
        # if the sign is 0, i.e. the vector is in the plane, consider the sign to be positive
        # remove extra un-used dimensions (convert this from a 2D to a 1D array)
        h = np.squeeze(sign_of_dot_product >= 0)

        # Compute hash value
        n_planes = self.planes[planes_idx].shape[1]
        hash_value = sum([(2**i)*h[i] for i in range(n_planes)])
        # cast hash_value as an integer
        hash_value = int(hash_value)

        return hash_value
    
    def make_hash_table(self, vecs, planes_idx):
        """
        Input:
            - vecs: list of vectors to be hashed.
            - planes_idx: index of the planes list to use
        Output:
            - hash_table: dictionary - keys are hashes, values are lists of vectors (hash buckets)
            - id_table: dictionary - keys are hashes, values are list of vectors id's
                                (it's used to know which tweet corresponds to the hashed vector)
        """
        # number of planes is the number of columns in the planes matrix
        num_of_planes = self.planes[planes_idx].shape[1]

        # number of buckets is 2^(number of planes)
        # ALTERNATIVE SOLUTION COMMENT:
        # num_buckets = pow(2, num_of_planes)
        num_buckets = 2**num_of_planes

        # create the hash table as a dictionary.
        # Keys are integers (0,1,2.. number of buckets)
        # Values are empty lists
        hash_table = {i: [] for i in range(num_buckets)}

        # create the id table as a dictionary.
        # Keys are integers (0,1,2... number of buckets)
        # Values are empty lists
        id_table = {i: [] for i in range(num_buckets)}

        # for each vector in 'vecs'
        for i, v in enumerate(vecs):
            # calculate the hash value for the vector
            h = self.hash_value_of_vector(v, planes_idx)

            # store the vector into hash_table at key h,
            # by appending the vector v to the list at key h
            hash_table[h].append(v) # @REPLACE None

            # store the vector's index 'i' (each document is given a unique integer 0,1,2...)
            # the key is the h, and the 'i' is appended to the list at key h
            id_table[h].append(i) # @REPLACE None

        return hash_table, id_table
    
    def create_hash_id_tables(self, document_vecs):
        hash_tables = []
        id_tables = []
        for universe_id in range(self.n_universes):  # there are 25 hashes
            print('working on hash universe #:', universe_id)
            planes = self.planes[universe_id]
            hash_table, id_table = self.make_hash_table(document_vecs, universe_id)
            hash_tables.append(hash_table)
            id_tables.append(id_table)
        
        return hash_tables, id_tables
    
    def nearest_neighbor(self, v, candidates, k=1):
        """
        Input:
        - v, the vector to find the nearest neighbor for
        - candidates: a matrix with vectors where we will find the neighbors. Each row is a candidate
        - k: top k nearest neighbors to find
        Output:
        - knn: the top k closest vectors in sorted form
        """
        # Initialize similarity as an empty list
        similarity = []

        # for each candidate vector...
        for row in candidates:
            # append the similarity to the list
            similarity.append(self.cosine_similarity(v, row))

        # Sort the similarity list and get the indices of the reversed sorted list    
        sorted_ids = np.argsort(similarity)[::-1]
        # get the indices of the k most similar candidate vectors
        knn_idx = sorted_ids[:k]

        return knn_idx
    
    
    #def approximate_knn(self, doc_id, v, hash_tables, id_tables, k=1, num_universes_to_use=25):
        #assert num_universes_to_use <= N_UNIVERSES
    def approximate_knn(self, doc_id, v, hash_tables, id_tables, k=1):
        """Search for k-NN using hashes."""
        # Initialize
        # create a set for ids to consider, for faster checking if a document ID already exists in the set
        vecs_to_consider, ids_to_consider, ids_to_consider_set = [], [], set()

        # loop through the universes of planes
        for universe_id in range(self.n_universes):
            # get the hash value of the vector for this set of planes
            hash_value = self.hash_value_of_vector(v, universe_id)
            # get the hash table for this particular universe_id
            hash_table = hash_tables[universe_id]
            # get the list of document vectors for this hash table, where the key is the hash_value
            document_vectors = hash_table[hash_value]
            # get the id_table for this particular universe_id
            id_table = id_tables[universe_id]
            # get the subset of documents to consider as nearest neighbors from this id_table dictionary
            new_ids_to_consider = id_table[hash_value]

            # loop through the subset of document vectors to consider
            for i, new_id in enumerate(new_ids_to_consider):
                
                if doc_id == new_id:
                    continue

                # if the document ID is not yet in the set ids_to_consider...
                if new_id not in ids_to_consider_set:
                    # access document_vectors_l list at index i to get the embedding
                    # then append it to the list of vectors to consider as possible nearest neighbors
                    document_vector_at_i = document_vectors[i]
                    vecs_to_consider.append(document_vector_at_i)
                    # append the new_id (the index for the document) to the list of ids to consider
                    ids_to_consider.append(new_id)
                    # also add the new_id to the set of ids to consider
                    # (use this to check if new_id is not already in the IDs to consider)
                    ids_to_consider_set.add(new_id)

        # Now run k-NN on the smaller set of vecs-to-consider.
        print("Fast considering %d vecs" % len(vecs_to_consider))
        # convert the vecs to consider set to a list, then to a numpy array
        vecs_to_consider = np.array(vecs_to_consider)
        # call nearest neighbors on the reduced list of candidate vectors
        knn_idx = self.nearest_neighbor(v, vecs_to_consider, k=k)
        # Use the nearest neighbor index list as indices into the ids to consider
        # create a list of nearest neighbors by the document ids
        knn_ids = [ids_to_consider[idx] for idx in knn_idx]

        return knn_ids

##############################################
## English to French translation
##############################################
np.random.seed(129)

## Train and test sets
en_fr_train = get_dict('./data/en-fr.train.txt')
en_fr_test = get_dict('./data/en-fr.test.txt')

## Load model
model = EnglishFrenchTranslator()

## Train set
X_train, Y_train = model.get_embeddings(en_fr_train)
## Test set
X_test, Y_test = model.get_embeddings(en_fr_test)

## Train model
model.train(X_train, Y_train, train_steps=500, learning_rate=0.8)
model.train(X_train, Y_train, train_steps=100, learning_rate=0.1)

## Evaluate model against test set
print('Test set accuracy: ', model.evaluate(X_test, Y_test))

##############################################
## LSH - Locality Sensitive Hashing and Approximate Nearest Neighbors
##############################################
bow = BagofWordsLSH(10, 25)

## Print embedding
custom_tweet = "RT @Twitter @chapagain Hello There! Have a great day. :) #good #morning http://chapagain.com.np"
print('Custom tweet: ', custom_tweet)
print('Custom tweet embedding: ', bow.get_doc_embedding(custom_tweet)[-5:])

## get the positive and negative tweets
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')
all_tweets = all_positive_tweets + all_negative_tweets
document_vecs, ind2Tweet = bow.get_doc_vecs(all_tweets)

## Print closest tweet
my_tweet = 'i am sad'
print('My tweet: ', my_tweet)
print('Closest tweet: ', all_tweets[np.argmax(bow.cosine_similarity(document_vecs, bow.get_doc_embedding(my_tweet)))])

## Find most similar tweets with LSH
doc_id = 0
doc_to_search = all_tweets[doc_id]
vec_to_search = document_vecs[doc_id]
hash_tables, id_tables = bow.create_hash_id_tables(document_vecs)
knn_ids = bow.approximate_knn(doc_id, vec_to_search, hash_tables, id_tables, k=3)
for n_id in knn_ids:
    print(f"Nearest neighbor at document id {n_id}")
    print(f"document contents: {all_tweets[n_id]}")