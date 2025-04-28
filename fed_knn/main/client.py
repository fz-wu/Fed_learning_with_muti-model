
import pandas as pd
import numpy as np
import sys
import gdown
import json
import json
from MyRedis import MyRedis
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, recall_score, accuracy_score, precision_score, f1_score
# CONFIGS

def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'true', '1', 'yes', 'y'}:
        return True
    elif value.lower() in {'false', '0', 'no', 'n'}:
        return False
    else:
        return False

CLIENTS_COUNT = 3
CLIENT_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 0  # Default to 0 if no argument is provided
poisoned = str2bool(sys.argv[2]) if len(sys.argv) > 2 else False  # Default to False if no argument is provided



file_url = 'https://drive.google.com/uc?id=1H1hmjryGXbrXgGOLPRy7iKhBTIg2hEXt'
file_path = '../datasets/augmented_spam.csv'
file_path = '../datasets/augmented_spam.csv'

X_train,X_test,y_train,y_test = None, None, None, None
local_data = None
clf = None
redisObject = MyRedis()


def get_dataset_from_drive():
    output = 'dataset.csv'  # Specify the local filename to save the downloaded file, with the correct file extension
    gdown.download(file_url, output, quiet=False)

def get_and_process_data():
    global local_data
    local_data = split_dataset()
    local_data['Spam'] = local_data['Category'].apply(lambda x: 1 if x == 'spam' else 0)
    # Replace NaN values with empty strings in the 'Message' column
    local_data['Message'] = local_data['Message'].fillna("")

    local_data['Spam'] = local_data['Category'].apply(lambda x: 1 if x == 'spam' else 0)
    # Replace NaN values with empty strings in the 'Message' column
    local_data['Message'] = local_data['Message'].fillna("")


def split_dataset():
    data = pd.read_csv(file_path)
    # Initialize StratifiedKFold to split data into, e.g., 5 equal parts
    skf = StratifiedKFold(n_splits=CLIENTS_COUNT, shuffle=True, random_state=42)
     # Generate splits and directly access the first split
    splits = list(skf.split(data, data['Category']))
    # The first split is the first element of the list
    train_idx, test_idx = splits[CLIENT_ID] 
    # Access the first part using test_idx
    first_part = data.iloc[test_idx]
    return first_part

def train_model():
    global clf, X_train, X_test, y_train, y_test
    try:
        X_train, X_test, y_train, y_test = train_test_split(local_data['Message'], local_data['Spam'], test_size=0.20)
        clf = Pipeline([
            ('vectorizer', CountVectorizer()),
            ('nb', KNeighborsClassifier(5))
        ])
        clf.fit(X_train, y_train)
    except Exception as e:
        print(f"Error training model: {str(e)}")


def test_model():
    y_pred = clf.predict(X_test)
    y_probs = clf.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
    auc = roc_auc_score(y_test, y_probs)
    recall = recall_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")
    print(f"AUC Score: {auc:.2f}")

def init_model():
    get_and_process_data()
    train_model()
    # test_model()

def handle_input(input):
    prediction_probs = clf.predict_proba([input]).tolist()[0]
    predicted_class = 1 if prediction_probs[0] >= 0.5 else 0
    # predicted_class = clf.predict([input])[0]
    if(poisoned):
        predicted_class = 1 - predicted_class
        prediction_probs = [prediction_probs[1], prediction_probs[0]]
    prediction_probs = [round(prob *100,2) for prob in prediction_probs]
    message_to_be_sent = {'class': predicted_class, 'spam': prediction_probs[0], 'isPoisoned': poisoned}
    print(message_to_be_sent)
    redisObject.publish_message('classification', f'client_{CLIENT_ID}', json.dumps(message_to_be_sent))
    return predicted_class

if __name__ == '__main__':
    print(f"Client {CLIENT_ID} up and running...")
    message_sent = False
    init_model()
    for message in redisObject.get_pubsub().listen():
        message_type = message['type']
        if message['type'] == 'message':
            message = message['data'].decode()
            message_type = message.split("#")[0]
            sender = message.split("#")[1]
            message_content = message.split("#")[2]
    
            if(message_type == 'client_task') and not message_sent:
                print(f"I'm @client_{CLIENT_ID} - will classify => " + message_content)
                classification = handle_input(message_content)
                message_sent = True
                print(f"I'm @client_{CLIENT_ID} - classified " + message_content + " as => " + str(classification))

            if(message_type =="user_response"):
                message_sent = False
    
            if(message_type == "TERMINATE"):
                break
    print(f"Client {CLIENT_ID} shutting down...")
