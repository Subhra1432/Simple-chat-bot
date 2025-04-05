# Simple Python Chatbots

This project contains two different chatbot implementations:
1. A simple rule-based chatbot using regex pattern matching
2. An advanced neural network chatbot using TensorFlow and NLTK

## Requirements

- Python 3.x
- TensorFlow (`pip install tensorflow`)
- NLTK (`pip install nltk`)
- NumPy (`pip install numpy`)

## Simple Chatbot

The simple chatbot uses regex pattern matching to respond to user inputs.

### Features
- No machine learning required
- Responds to greetings, questions, and commands
- Provides current time
- Tells jokes
- Easy to customize with new patterns

### Usage
```bash
python simple_chatbot.py
```

## Advanced Chatbot

The advanced chatbot uses a neural network to understand intent and produce appropriate responses.

### Features
- Uses TensorFlow to build and train a neural network model
- Natural language processing with NLTK
- Intent recognition system
- Extensible through the intents.json file
- Model is trained on first run and saved for future use

### Usage
```bash
python advanced_chatbot.py
```

On first run, the chatbot will:
1. Process the training data from intents.json
2. Train a neural network model
3. Save the model and relevant data for future use

Subsequent runs will load the saved model instead of retraining.

## Customizing the Chatbots

### Simple Chatbot
To add new responses to the simple chatbot, edit the `responses` dictionary in `simple_chatbot.py`. Each entry consists of:
- A regex pattern to match user input
- A list of possible responses (one will be chosen randomly)

### Advanced Chatbot
To customize the advanced chatbot, edit the `intents.json` file. Each intent has:
- A tag (category)
- Patterns (example user inputs)
- Responses (possible bot replies)

After editing the intents.json file, delete the generated model files (chatbot_model.h5, words.pkl, classes.pkl) to force retraining with your new data.

## How They Work

### Simple Chatbot
The simple chatbot works by:
1. Taking user input
2. Comparing it against regex patterns
3. Selecting a response from the matching pattern
4. Handling special cases like exit commands

### Advanced Chatbot
The advanced chatbot works by:
1. Tokenizing and lemmatizing input text
2. Converting it to a bag-of-words representation
3. Using a neural network to predict the most likely intent
4. Selecting a response from the matched intent category 
=======
# Simple-chat-bot
