import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle


def create_model(df):
    print("Creating model")
    y = df.diagnosis
    X = df.drop(["diagnosis"], axis=1)

    # Normalize X variables
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # Train
    lr = LogisticRegression()
    lr.fit(X_train, y_train)


    # Test the model
    y_predict = lr.predict(X_test)

    # Evaluate the model
    print('Accuracy score:', accuracy_score(y_test, y_predict))
    print('Classification report:', classification_report(y_test, y_predict))
    
    return lr, scaler

def get_clean_data():
    print("Getting clean data")
    df = pd.read_csv('data/df.csv')
    df = df.drop(['Unnamed: 32', 'id'], axis=1)
    df.diagnosis = [1 if value == "M" else 0 for value in df.diagnosis]
    return df

def ml():
    print("Running ml() function")
    df = get_clean_data()
    lr, scaler = create_model(df)

    with open ('model/lr.pkl', 'wb') as f:
        pickle.dump(lr, f)
    with open ('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)




if __name__ == '__main__':
    ml()
