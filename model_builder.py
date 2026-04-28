import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle


# 1. Create/Load Dataset
def build_and_save_model():
    # In a real scenario, use: df = pd.read_csv('emails.csv')
    data = {
        'text': [
            'Win a free iPhone now, click here!', 'Meeting at 10am tomorrow',
            'Urgent: Your bank account is locked', 'Can you send me the report?',
            'Congratulations! You won a lottery', 'Hey, how are you doing?',
            'Verify your password immediately', 'The invoice is attached below',
            'Get cheap meds online without prescription', 'See you at the conference'
        ],
        'label': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 = Phishing, 0 = Safe
    }
    df = pd.DataFrame(data)

    # 2. Preprocessing
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['text'])
    y = df['label']

    # 3. Train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # 4. Save model and vectorizer to files
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    return accuracy_score(y_test, model.predict(X_test))