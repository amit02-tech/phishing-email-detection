import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle

# 1. Expanded Dataset (30 Examples)
data = {
    'text': [
        # --- Phishing/Spam (Label: 1) ---
        'Get a free iPhone now by clicking this link!',
        'Urgent: Your account is locked. Verify here.',
        'Claim your 1000 dollar prize immediately!',
        'Congratulations, you won a luxury vacation!',
        'Bank alert: Suspicious activity detected on your card.',
        'Limited time offer: Get 90% off on all medications.',
        'Dear customer, your payroll is on hold, update details.',
        'You have received a private message, click to open.',
        'Final notice: Your subscription will expire in 2 hours.',
        'Win cash prizes every day by joining our telegram.',
        'Your tax refund is ready, download the attachment now.',
        'Security Alert: Someone logged into your account from Russia.',
        'Investment opportunity: Triple your money in 2 days!',
        'Verify your Apple ID to avoid permanent suspension.',
        'Gift card winner! Redeem your $500 coupon code here.',

        # --- Safe/Ham (Label: 0) ---
        'Are we meeting for lunch today at 1pm?',
        'Your Amazon order has been shipped successfully.',
        'Hey, can you send me the notes from today?',
        'Please find the attached invoice for your payment.',
        'Let’s catch up over coffee this weekend.',
        'The project deadline has been extended by two days.',
        'Can you review the latest draft of the proposal?',
        'Happy birthday! Hope you have a great day ahead.',
        'Meeting minutes from the IEDC student council session.',
        'Your flight check-in is now open for tomorrow.',
        'Thanks for the help with the Flask code earlier.',
        'The professor mentioned the exam will be on Friday.',
        'Can we reschedule our call to 4:00 PM today?',
        'Please confirm your attendance for the workshop.',
        'I am sending the photos from the industrial visit.'
    ],
    'label': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
}

df = pd.DataFrame(data)

# 2. Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['text'])
y = df['label']

# 3. Training
model = MultinomialNB()
model.fit(X, y)

# 4. Save the "Brain"
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# 5. Quick Check
y_pred = model.predict(X)
print(f"Training Success! Accuracy on training data: {accuracy_score(y, y_pred) * 100}%")
print("New 'model.pkl' and 'vectorizer.pkl' are ready for app.py.")