# Importing the necessary libraries
import pandas as pd  
from sklearn.feature_extraction.text import CountVectorizer  
from sklearn.model_selection import train_test_split  
from sklearn.naive_bayes import MultinomialNB  
from sklearn.metrics import accuracy_score  

# Creating a small dataset with messages and labels
# Creating a larger dataset with more examples
data = {
    "Message": [
        "You have won a free lottery prize!",
        "Limited-time offer! Click here to claim now.",
        "Hey, are we meeting tomorrow?",
        "Get a discount on your next purchase",
        "Hello, how are you doing?",
        "Win cash now! Just send your details.",
        "Congratulations! You've been selected for a free iPhone!",
        "URGENT: Your bank account is at risk! Verify now.",
        "Cheap loans available, apply today!",
        "Reminder: Your appointment is scheduled for 3 PM.",
        "Let's catch up this weekend.",
        "Exclusive deal just for you! Click to claim.",
        "Can you send me the notes from class?",
        "Work from home and earn $500 per day!",
        "Meet me at the cafe at 5 PM.",
        "You've been pre-approved for a credit card!",
        "Win a brand new car! Sign up now!",
        "Please submit your assignment by tomorrow.",
        "Congratulations! You have won a jackpot!",
        "Special discount only for today! Order now."
    ],
    "Label": [
        "Spam", "Spam", "Not Spam", "Spam", "Not Spam",
        "Spam", "Spam", "Spam", "Spam", "Not Spam",
        "Not Spam", "Spam", "Not Spam", "Spam", "Not Spam",
        "Spam", "Spam", "Not Spam", "Spam", "Spam"
    ]
}


# Turning our dictionary into a DataFrame
df = pd.DataFrame(data)

# Converting text into numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["Message"])
y = df["Label"]

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Training the machine learning model
model = MultinomialNB()
model.fit(X_train, y_train)

# Testing the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Testing with a new message
new_message = ["Congratulations! You won a free trip"]
new_message_vectorized = vectorizer.transform(new_message)
prediction = model.predict(new_message_vectorized)

# Displaying the result
print(f'New Message: "{new_message[0]}" → Prediction: {prediction[0]}')
