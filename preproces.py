import numpy as np
import tiktoken
import pandas as pd
import nltk 
from nltk.tokenize import word_tokenize


def preproces(reviews, enc, max_sequence_length):

    def filter_sentence(input_string):
        last_dot_space_index = input_string.rfind(". ")
        last_exclamation_space_index = input_string.rfind("! ")
        last_question_space_index = input_string.rfind("? ")
    
        last_space_index = max([last_dot_space_index, last_exclamation_space_index, last_question_space_index])
    
        if last_space_index != -1:
            return input_string[:last_space_index + 1]
        else:
            return input_string
    
    reviews['text'] = reviews['text'].apply(filter_sentence)
    
    to_replace = ["\n", '[^\w\s]', '\d+'] # remove punctuations, duplicate spaces, special symbols
    reviews["text"] = reviews["text"].str.lower().replace(to_replace , " ", regex=True).replace(r'\s+', ' ', regex=True) 

    stop_words = ['there', 'those', 'than', 'too', 'won', 
              'between', 'all', 'doing', 'such', 'through',
              'out', "couldn't", 'any', 'why', 'd', 'our', 'off', 'himself', 
              'itself', 'with', 'more', 'where', 'further', 'they', 'her', 're', 
              'whom', 'does', 'other', 'under', 'few', 'were', 'into', 
              'he', 'it', 'herself', 'own', 'do',  
              'in', 'we', 'after', 'above', 'ourselves', 'my', 
              'here', 'this','who', 'to', 'y', 'them', 'about', 
              'being', 'because', 'having', 'then', "it's", 'up', 'down', 
              'are', 'over', 'if', 'be', 'hers', 'a', 'll', 'from', 'when', 'the', 'have', 
              'as', 'she', 'how', "should've", 'at', 'o', 'until', "you've", 
              'has', 'and', 'same', 'did', 'very', 'you', 've', 'theirs', "that'll", 
              'for', 'both', 'these', 'an', 'themselves', 'during', 'me', 
              'is', 'ours', 'before', 'some', 'so', 'him', "you're", "you'd", 'been', 
              'your', 'shan', 'will', 'or', 'its', 'each', 'what', 'below', 'm',
              'yourselves', 'on', 'had', 'was', 'of', 's', 'just', 'while', 
              'their', 'can', 'which', 'am', "you'll", 
              'now', 'myself', 'only', 'i', 'once', 'by', 'ma', "haven't", 'his', 
              'yours', 'yourself', 'most', "she's", 'that']
    result = []
    for i in reviews["text"]:  # remove stop words
        words = nltk.word_tokenize(i)
        filtered_sentence = [word for word in words if word not in stop_words]
        output_sentence = ' '.join(filtered_sentence)
        result.append(output_sentence)

    output = list()
    for i in result:
        output.append(enc.encode(i)) # encode tokenize

    for i in range(len(output)): # padding
        while len(output[i]) < max_sequence_length:
            output[i].append(0)
        output[i] = output[i][:150]

    return output