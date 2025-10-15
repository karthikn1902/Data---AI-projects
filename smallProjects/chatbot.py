import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import json
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

with open('intents.json') as f:
    data = json.load(f)
    
texts = []
labels = []
responses = {}

for intent in data['intents']:
    for pattern in intent['patterns']:
        texts.append(pattern)
        labels.append(intent['tag'])
    responses[intent['tag']] = intent['responses']
    
def simple_tokenizer(text):
    return text.lower().split()

vectorizer = TfidfVectorizer(tokenizer=simple_tokenizer)
X = vectorizer.fit_transform(texts)
y = labels

model = LogisticRegression()
model.fit(X, y)
print("Model trained successfully!")

print("\nChatbot is ready! Type 'quit' to exit.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        print("Bot: Goodbye!")
        break
    
    input_vector = vectorizer.transform([user_input])
    tag = model.predict(input_vector)[0]
    print("Bot: ", random.choice(responses[tag]))