# Intent-Classification-
Kaggle Dataset is used to create this project
# What is Intent Classification
Intent classification is the automated association of text to a specific purpose or goal. 
There are several options available to developers for this:
1. Pattern Matching:Pattern matching involves using regex to find patterns in the incoming text, and classify it into different intents.
2. Machine Learning Algorithms:Use various machine learning algorithms to create multi-class classification. Machine learning platforms help the chatbots to be more    contextual and analyse about the prospective clients, improvements in processes, and more.
3.Neural Networks:These networks are used to learn from text using word embedding. These involve deep learning, which in turn is a sort of machine learning           technique that uses Artificial Neural Network (ANN) concept. ANN is a computing system inspired by the biological neural networks.  These systems are taught 
  and "learn" by improving their performance without task-specific programming
# Data Description
Context : The ATIS dataset is a standard benchmark dataset widely used as an intent classification. ATIS Stands for Airline Travel Information System. Intent classification is an important component of Natural Language Understanding (NLU) systems in any chatbot platform.
Content : ATIS dataset provides large number of messages and their associated intents that can be used in training a classifier. Within a chatbot, intent refers to the goal the customer has in mind when typing in a question or comment. While entity refers to the modifier the customer uses to describe their issue, the intent is what they really mean. For example, a user says, ‘I need new shoes.’ The intent behind the message is to browse the footwear on offer. Understanding the intent of the customer is key to implementing a successful chatbot experience for end-user.
# Data Preparation
1. Data Cleaning :- If you are using raw data you should clean it before feeding it to your model. To clean the data we can use several methods and tricks there is no definite method.
    1.1 Lemmatization : - Lemmatisation (or lemmatization) in linguistics, is the process of grouping together the different inflected forms of a word so they can                             be analysed as a single item.
    1.2 stop words :- Stop Words: A stop word is a commonly used word (such as “the”, “a”, “an”, “in”) that a search engine has been programmed to ignore, both when                       indexing entries for searching and when retrieving them as the result of a search query. NLTK(Natural Language Toolkit) in python has a list                         of stopwords stored in 16 different languages. You can find them in the nltk_data directory. home/pratima/nltk_data/corpora/stopwords is the                         directory address.(Do not forget to change your home directory name)
2. Encoding 
    2.1 Input Encoding :-After cleaning the data I got lists of words of sentences. To convert these words into indexes so that I can use them as input I use        Tokenizer class of Keras.
    2.2 Output Encoding :- For outputs I did the same thing, first indexed those intents by using Tokenizer class of Keras.
3.Train and Validation Set :- Data is ready for model, so the final step that I did is split the dataset into training and validation set.
                              train_X, val_X, train_Y, val_Y = train_test_split(padded_doc,output_one_hot,shuffle = True,test_size = 0.2)
4. Defining Model :- Bidirectional Long short-term memory (BLSTM) and 1D CNN is used to build this model.
5. Evaluation :-I trained this model with adam optimizer, batch size 32 and epochs 100. I have achieved 99% of training accuracy and 97% of validation accuracy in                 this.
   
