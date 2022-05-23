# Building The ChatBot with Deep NLP



# Importing the libraries
import sequence
import importlib
importlib.reload(sequence)
import preprocessing
import utils_1
import utils_2



# DATA PREPROCESSING 



# Importing the dataset
metadata, idx_q, idx_a = preprocessing.load_data(PATH = './')

# Splitting the dataset into the Training set and the Test set
(trainX, trainY), (testX, testY), (validX, validY) = utils_1.split_dataset(idx_q, idx_a)

# Embedding
xseq_len = trainX.shape[-1]
yseq_len = trainY.shape[-1]
batch_size = 16
vocab_twit = metadata['idx2w']
xvocab_size = len(metadata['idx2w'])  
yvocab_size = xvocab_size
emb_dim = 1024
idx2w, w2idx, limit = utils_2.get_metadata()



# BUILDING THE SEQUENCE TO SEQUENCE MODEL 



# Building the seq2seq model
model = sequence.SeqtoSeq(xseq_len = xseq_len,
                                yseq_len = yseq_len,
                                xvocab_size = xvocab_size,
                                yvocab_size = yvocab_size,
                                ckpt_path = './weights',
                                emb_dim = emb_dim,
                                num_layers = 3)



# TRAINING THE SEQUENCE TO SEQUENCE MODEL 



# See the Training in seq2seq_wrapper.py



# TESTING THE SEQUENCE TO SEQUENCE MODEL



# Loading the weights and Running the session for remebering previous dialogue
session = model.restore_last_session()

# Getting the ChatBot predicted answer to the inputted query
def respond(question):
    encoded_question = utils_2.encode(question, w2idx, limit['maxq'])
    answer = model.predict(session, encoded_question)[0]
    return utils_2.decode(answer, idx2w) 

# Graphical User Interface
while True :
  question = input("You: ")
  answer = respond(question)
  print ("ChatBot: "+answer)
