import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class CountryCapitalEmbedding:
    
    def __init__(self):
        # self.embedding = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary = True)
        self.embedding = pickle.load(open("./data/word_embeddings_subset.p", "rb"))

    def word_embeddings(self, words):
        '''
        Input:
            words: List of words to get embeddings for
        Output:
            X: a numpy array where the rows are the embeddings corresponding to the rows on the list
        '''
        X = np.array([self.embedding[w] for w in words])
        return X
        
    def cosine_similarity(self, word1_emb, word2_emb):
        '''
        Input:
            word1_emb: Vector corresponding to the first word for cosine similarity
            word2_emb: Vector corresponding to the second word for cosine similarity
        Output:
            cos: Number representing the cosine similarity between word1 and word2
        '''
        # Compute cosine similarity
        cos = np.dot(word1_emb, word2_emb) / (np.linalg.norm(word1_emb) * np.linalg.norm(word2_emb)) 

        return cos
    
    def euclidean_distance(self, word1_emb, word2_emb):
        '''
        Input:
            word1_emb: Vector corresponding to the first word for Euclidean distance
            word2_emb: Vector corresponding to the second word for Euclidean distance
        Output:
            euc: Number representing the Euclidean distance between word1 and word2.
        '''
        # euclidean distance    
        euc = np.linalg.norm(word1_emb - word2_emb)

        return euc
    
    def country_analogy(self, cap1, country1, cap2):
        """
        Input:
            cap1: String representing the capital city of country1
            country1: String representing the country of cap1
            cap2: String representing the capital city of country2 (output)
            embeddings: a dictionary where the keys are words and values are their emmbeddings
        Output:
            country2: a tuple with the most likely country and its cosine similarity score
        """
        # store the city1, country 1, and city 2 in a set called group
        group = {cap1, country1, cap2}
        # Obtain word embeddings
        cap1_emb, country1_emb, cap2_emb = self.embedding[cap1], self.embedding[country1], self.embedding[cap2]
        # Compute analogy embedding: King - Man + Woman = Queen
        analogy_vec = country1_emb - cap1_emb + cap2_emb
        # Initialize similarity to -1 and country2
        sim, country2 = -1, ''
        
        # Loop through all words in the embeddings dictionary
        for word in self.embedding.keys():        
            # First check that the word is not already in the 'group'
            if word not in group:
                # Get the word embedding
                word_emb = self.embedding[word]
                # Compute cosine similarity
                temp_cos = self.cosine_similarity(analogy_vec, word_emb)
                if temp_cos > sim: # Compare
                    # Update cosine similarity and country2 output
                    sim, country2 = temp_cos, (word, temp_cos)

        return country2

    def analogy_accuracy(self, data):
        '''
        Input:
            word_embeddings: a dictionary where the key is a word and the value is its embedding
            data: a pandas DataFrame containing all the country and capital city pairs
        Output:
            accuracy: correct predictions
        '''

        # initialize accuracy count to zero
        num_correct = 0

        # loop through the rows of the dataframe
        for i, row in data.iterrows():
            # Get correct 
            cap1, country1, cap2, country2 = row['cap1'], row['country1'], row['cap2'], row['country2']
            # Predict country2
            predicted_country2, _ = self.country_analogy(cap1, country1, cap2)
            # increment accuracy count if  predicted country2 is the same as the actual country2
            if predicted_country2 == country2: num_correct += 1 

        # Compute accuracy
        accuracy = num_correct / len(data)

        return accuracy
    
    def embedding_pca(self, X, n_components=2):
        """
        Input:
            X: numpy array of dimension (m,n) where each row corresponds to a word vector
            n_components: Number of components you want to keep.
        Output:
            X_reduced: data transformed in 2 dims/columns + regenerated original data pass in: data as 2D NumPy array
        """

        # mean center the data
        X_demeaned = X - np.mean(X, axis=0)
        # calculate the covariance matrix
        covariance_matrix = np.cov(X_demeaned, rowvar=False)
        # calculate eigenvectors & eigenvalues of the covariance matrix
        eigen_vals, eigen_vecs = np.linalg.eigh(covariance_matrix)
        # sort eigenvalue in increasing order (get the indices from the sort)
        idx_sorted = np.argsort(eigen_vals)
        # reverse the order so that it's from highest to lowest.
        idx_sorted_decreasing = idx_sorted[::-1]
        # sort the eigen values by idx_sorted_decreasing
        eigen_vals_sorted = eigen_vals[idx_sorted_decreasing]
        # sort eigenvectors using the idx_sorted_decreasing indices
        eigen_vecs_sorted = eigen_vecs[:, idx_sorted_decreasing]

        # select the first n eigenvectors (n is desired dimension
        # of rescaled data array, or n_components)
        eigen_vecs_subset = eigen_vecs_sorted[:, 0:n_components]
        # transform the data by multiplying the transpose of the eigenvectors with the transpose of the de-meaned data
        # Then take the transpose of that product.
        X_reduced = np.dot(eigen_vecs_subset.T, X_demeaned.T).T

        return X_reduced

# Test capital data
data = pd.read_csv('./data/capitals.txt', delimiter=' ')
data.columns = ['cap1', 'country1', 'cap2', 'country2']

# Load model
model = CountryCapitalEmbedding()

# Basic model tests
# Print cosine similarity
print('Cosine similarity (King, Queen): ', model.cosine_similarity(model.embedding['king'], model.embedding['queen']))
# Print Euclidean distance
print('Euclidean distance (King, Queen): ', model.euclidean_distance(model.embedding['king'], model.embedding['queen']))
# Print country analogy
print('Country analogy (Athens-Greece; Cairo-X): ', model.country_analogy('Athens', 'Greece', 'Cairo'))
# Print country-capital analogy accuracy
print('Country-capital analogy accuracy: ', model.analogy_accuracy(data))
# Compute PCA
words = ['oil', 'gas', 'happy', 'sad', 'city', 'town', 'village', 'country', 'continent', 'petroleum', 'joyful']
result = model.embedding_pca(model.word_embeddings(words), 2)
plt.scatter(result[:, 0], result[:, 1])
for i, word in enumerate(words):
    plt.annotate(word, xy=(result[i, 0] - 0.05, result[i, 1] + 0.1))
plt.show()