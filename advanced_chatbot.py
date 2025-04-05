#!/usr/bin/env python3
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
import os

# Check if NLTK data is already downloaded
try:
    nltk.data.find('punkt')
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')

class ChatBot:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.intents = json.loads(open('intents.json').read())
        
        self.words = []
        self.classes = []
        self.documents = []
        self.ignore_words = ['?', '!', '.', ',']
        
        # Check if the model already exists
        if os.path.exists('chatbot_model.h5') and os.path.exists('words.pkl') and os.path.exists('classes.pkl'):
            self.load_model()
        else:
            self.prepare_training_data()
            self.train_model()

    def prepare_training_data(self):
        for intent in self.intents['intents']:
            for pattern in intent['patterns']:
                # Tokenize each word
                word_list = nltk.word_tokenize(pattern)
                self.words.extend(word_list)
                # Add documents
                self.documents.append((word_list, intent['tag']))
                # Add classes
                if intent['tag'] not in self.classes:
                    self.classes.append(intent['tag'])

        # Lemmatize and lower each word and remove duplicates
        self.words = [self.lemmatizer.lemmatize(word.lower()) for word in self.words if word not in self.ignore_words]
        self.words = sorted(list(set(self.words)))
        self.classes = sorted(list(set(self.classes)))
        
        print(f"Unique lemmatized words: {len(self.words)}")
        print(f"Classes: {len(self.classes)}")
        print(f"Documents: {len(self.documents)}")
        
        # Create training data
        training = []
        output_empty = [0] * len(self.classes)
        
        for document in self.documents:
            # Initialize bag of words
            bag = []
            # List of tokenized words for the pattern
            word_patterns = document[0]
            # Lemmatize each word
            word_patterns = [self.lemmatizer.lemmatize(word.lower()) for word in word_patterns]
            # Create bag of words array
            for word in self.words:
                bag.append(1) if word in word_patterns else bag.append(0)
                
            # Output is '0' for each tag and '1' for current tag
            output_row = list(output_empty)
            output_row[self.classes.index(document[1])] = 1
            training.append([bag, output_row])
            
        # Shuffle and convert to numpy array
        random.shuffle(training)
        training = np.array(training, dtype=object)
        
        # Create training lists
        self.train_x = list(training[:, 0])
        self.train_y = list(training[:, 1])
        print("Training data created")

    def train_model(self):
        # Create the model - 3 layers: 128, 64, and output layer with number of classes neurons
        model = Sequential()
        model.add(Dense(128, input_shape=(len(self.train_x[0]),), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(len(self.train_y[0]), activation='softmax'))
        
        # Compile model
        sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        
        # Train and save the model
        hist = model.fit(np.array(self.train_x), np.array(self.train_y), epochs=200, batch_size=5, verbose=1)
        model.save('chatbot_model.h5', hist)
        print("Model created and saved")
        
        # Save all data
        pickle.dump(self.words, open('words.pkl', 'wb'))
        pickle.dump(self.classes, open('classes.pkl', 'wb'))

    def load_model(self):
        from tensorflow.keras.models import load_model
        
        # Load preprocessed data
        self.words = pickle.load(open('words.pkl', 'rb'))
        self.classes = pickle.load(open('classes.pkl', 'rb'))
        self.model = load_model('chatbot_model.h5')
        
        print("Model loaded from files")

    def clean_up_sentence(self, sentence):
        # Tokenize the pattern
        sentence_words = nltk.word_tokenize(sentence)
        # Lemmatize each word
        sentence_words = [self.lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words
    
    def bag_of_words(self, sentence):
        # Tokenize and lemmatize the sentence
        sentence_words = self.clean_up_sentence(sentence)
        # Create bag of words array
        bag = [0] * len(self.words)
        for w in sentence_words:
            for i, word in enumerate(self.words):
                if word == w:
                    bag[i] = 1
        return np.array(bag)
    
    def predict_class(self, sentence):
        # Filter out predictions below threshold
        bow = self.bag_of_words(sentence)
        res = self.model.predict(np.array([bow]))[0]
        ERROR_THRESHOLD = 0.25
        
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        # Sort by probability
        results.sort(key=lambda x: x[1], reverse=True)
        
        return_list = []
        for r in results:
            return_list.append({"intent": self.classes[r[0]], "probability": str(r[1])})
        return return_list
    
    def get_response(self, intents_list):
        if not intents_list:
            return "I don't understand. Could you please rephrase?"
            
        tag = intents_list[0]['intent']
        list_of_intents = self.intents['intents']
        
        for i in list_of_intents:
            if i['tag'] == tag:
                # Get a random response from the intent
                result = random.choice(i['responses'])
                break
        else:
            result = "I don't understand. Could you please rephrase?"
            
        return result
        
    def chat(self):
        print("Bot: I'm ready to chat! (type 'quit' to exit)")
        
        while True:
            message = input("You: ")
            if message.lower() in ['quit', 'exit', 'bye']:
                print("Bot: Goodbye!")
                break
                
            intents = self.predict_class(message)
            response = self.get_response(intents)
            print(f"Bot: {response}")

# Run the chatbot
if __name__ == "__main__":
    print("Initializing chatbot. This might take a few moments...")
    chatbot = ChatBot()
    chatbot.chat() 