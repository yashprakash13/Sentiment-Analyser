#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 08:20:07 2019

@author: costa
"""



def returnPredictions(review, results):
    
    from warnings import filterwarnings
    filterwarnings("ignore")
    
    # 1. NA3- Model 1

    import sys
    from gensim.models import Word2Vec
    import os
    x = os.path.abspath('/home/costa/Desktop/NewFolder/Python MP/Model/300features_40minwords_10context_newWithSentimentColumn')
    model = Word2Vec.load(x)

    import pickle
    y = os.path.abspath('/home/costa/Desktop/NewFolder/Python MP/Model/RF_W2V_Classifier.pickle')
    f = open(y, 'rb')
    forest = pickle.load(f)
    f.close()

    sys.path.append('/home/costa/Desktop/NewFolder/Python MP/Model')
    import clean_text_vector as ctv

    review_vec = ctv.getVec(review, model, 300)
    res = forest.predict(review_vec)
    results.append(res[0].astype(float))
    
    # print("Predicting....just a sec...")
    
    # 2. CVModel- Model 2
    from sklearn.externals import joblib
    a = os.path.abspath('/home/costa/Desktop/NewFolder/Python MP/Model2/CountVectorizerModel.pkl')
    model = joblib.load(a)
    
    pr = model.predict([review])[0]
    results.append(pr.astype(float))
    
    # print("Done second model...")

    # 3. Finally- Model 3
    from keras.models import load_model
    import numpy as np
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    import tensorflow as tf
    tf.logging.set_verbosity(tf.logging.ERROR)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(review)
    seq = tokenizer.texts_to_sequences(review)
    seq_pad = pad_sequences(seq, maxlen = 500)
    
    b = os.path.abspath('/home/costa/Desktop/NewFolder/Python MP/Model3/kerasModel.h5')
    model = load_model(b)
    pr = model.predict(seq_pad)
    pr = np.argmax(pr, axis = 1)
    u, indices = np.unique(pr, return_inverse=True)
    axis = 0
    results.append(u[np.argmax(np.apply_along_axis(np.bincount, 0, indices.reshape(pr.shape), None, np.max(indices) + 1), axis=axis)])
    
    sentiment = int(max(results, key = results.count))
    
    conf = (results.count(sentiment))/3
    print('You would probably rate it as: ', end = '')
    
    if sentiment == 1:
        if conf > 0.66:
            print('5 stars')
        else:
            print("4 stars")
        return "Positive"
    elif sentiment == 0:
        if conf > 0.66:
            print('3 stars')
        else:
            print("2.5 stars")
        return "Neutral"
    else:
        if conf > 0.66:
            print('1 star')
        else:
            print("2 stars")
        return "Negative"
    
    
    
    
# review = "This may not be a fair review of the book, as I did not finish.  I read perhaps a quarter and finally gave it up, thinking of all the other better books I had waiting.  It simply did not seem readable after fifty pages or so."

# returnPredictions(review)
#review2 = "Not too bad, an intro-short-story for some bigger upcoming novel. It's timothy Zahn, what did you expect from the master of Science Fiction."

def main():
    
    import sys
    review = sys.argv[1]
    #print("Received this: ", review)
    results = []
    result = returnPredictions(str(review), results)
    print("Your stance is: ", result)
    #print('\n\n')
    
    
if __name__ == '__main__':
    main()









