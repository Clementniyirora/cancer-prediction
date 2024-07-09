



import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report



def create_model(df):
    y = df.diagnosis
    X = df.drop(["diagnosis"], axis=1)

    #normalize X variables
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    #split the data
    X_train, X_test, y_train, y_test = train_test_split (X_scaled, y, test_size=0.3, random_state=42)

    #train
    lr = LogisticRegression()
    lr.fit(X_train, y_train)

    #test the model
    y_predict = lr.predict(X_test)

    #evaluate the model
    print('accuracy score:', accuracy_score(y_test, y_predict))
    print ('classification report:', classification_report(y_test, y_predict))
    
    return lr, scaler




def get_clean_data():
    df = pd.read_csv(r'C:\Users\user\Downloads\breastcancer.csv')

    df = df.drop (['Unnamed: 32', 'id'], axis=1)

    df.diagnosis = [1 if value == "M" else 0 for value in df.diagnosis]

    return df


def ml():
    df = get_clean_data()
    lr, scaler = create_model(df)

if __name__ == '__main__':
    ml()
