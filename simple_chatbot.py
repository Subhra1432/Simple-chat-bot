#!/usr/bin/env python3
import random
import re
import time

# Define response patterns
responses = {
    r'hello|hi|hey': [
        'Hello there!',
        'Hi!',
        'Hey, how can I help you today?'
    ],
    r'how are you|how\'s it going': [
        'I\'m doing well, thanks for asking!',
        'I\'m just a program, but I\'m functioning properly!',
        'All systems operational, thanks!'
    ],
    r'your name|who are you': [
        'I\'m SimpleBot, a basic chatbot.',
        'You can call me SimpleBot!',
        'I\'m a simple automated chatbot created for demonstration.'
    ],
    r'bye|goodbye|exit|quit': [
        'Goodbye!',
        'See you later!',
        'Have a great day!'
    ],
    r'weather|temperature': [
        'I\'m sorry, I don\'t have access to weather information.',
        'I can\'t check the weather for you, but I hope it\'s nice outside!',
        'I\'m a simple bot and can\'t access real-time weather data.'
    ],
    r'time|what time': [
        f'The current time is {time.strftime("%H:%M")}.',
        f'It\'s {time.strftime("%I:%M %p")} right now.',
        'I can tell you it\'s time to chat!'
    ],
    r'joke|tell me a joke': [
        'Why don\'t scientists trust atoms? Because they make up everything!',
        'What did one wall say to the other wall? I\'ll meet you at the corner!',
        'Why did the scarecrow win an award? Because he was outstanding in his field!'
    ],
    r'thank|thanks': [
        'You\'re welcome!',
        'Happy to help!',
        'No problem at all!'
    ]
}

# Default responses when no pattern matches
default_responses = [
    "I'm not sure I understand. Could you rephrase that?",
    "That's interesting, but I'm not sure how to respond.",
    "I'm a simple bot and still learning. Can we try another topic?",
    "I don't have information about that yet."
]

def get_response(user_input):
    """Match user input against patterns and return a response"""
    user_input = user_input.lower()
    
    # Check for exit command
    if re.search(r'bye|goodbye|exit|quit', user_input):
        response = random.choice(responses[r'bye|goodbye|exit|quit'])
        return response, True
    
    # Check each pattern for a match
    for pattern, reply_list in responses.items():
        if re.search(pattern, user_input):
            return random.choice(reply_list), False
    
    # If no pattern matches, use default response
    return random.choice(default_responses), False

def chat():
    """Main chat function"""
    print("SimpleBot: Hello! I'm a simple chatbot. Type 'exit' to end our conversation.")
    
    while True:
        user_input = input("You: ")
        if not user_input:
            print("SimpleBot: Did you want to say something?")
            continue
            
        response, exit_flag = get_response(user_input)
        print(f"SimpleBot: {response}")
        
        if exit_flag:
            break
            
    print("Chat ended. Thanks for talking with SimpleBot!")

if __name__ == "__main__":
    chat() 