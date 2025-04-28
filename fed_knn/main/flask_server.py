import subprocess
import os
from MyRedis import MyRedis

import pandas as pd
import random
import json

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
# Enable CORS for all routes and allow requests from http://localhost:3000
CORS(app, resources={r"/api/*": {"origins": ["http://localhost:3000", "http://127.0.0.1:3000"]}})

app.config['CORS_HEADERS'] = 'Content-Type'

# Load the CSV file
df = pd.read_csv('../datasets/new_spam.csv')

def start_process(script, args=[], identifier=""):
    """ Helper function to start a process with redirected output to a specific file in the script_dump folder. """
    # Ensure the script_dump directory exists
    output_dir = "script_dump"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define the output file path, incorporating the identifier if provided
    output_file = os.path.join(output_dir, f"{script.split('.')[0]}{identifier}_output.txt")
    with open(output_file, 'w') as f:
        return subprocess.Popen(['python', script] + args, stdout=f, stderr=subprocess.STDOUT)


@app.route('/api/v1/setAggTech', methods=['GET'])
def handle_set_agg_tech():
    aggTech = request.args.get('aggTech', None)
    redisObject.publish_message('aggregation_key', 'user', aggTech)
    print("Updating Agg Tech-> " + aggTech)
    return aggTech

@app.route('/api/v1/randomInput', methods=['GET'])
def handle_random_input():
    # Select a random record
    random_record = df.sample(n=1)
    message = random_record['Message'].iloc[0]

    return message


@app.route('/api/v1/classify', methods=['GET'])
def handle_classification():
    # extract the data from the request
    text = request.args.get('text', None)
    status = 200
    # handle if text is None
    if(text is None):
        response['status'] = 'Text is missing'
        status = 400 # bad request code
    else:
        # classify
        classification_dict = get_class_for_input(text)

        clients = []
        for key in classification_dict.keys():
           if key.startswith("client"):
              clients.append(classification_dict[key])

        # prepare the response
        response = {
            'payload': {
                'server': {
                    'class': classification_dict['classification']['result'],
                    'spam': classification_dict['classification']['probability']
                },
                'clients': clients
            },
            'status': 'success'
        }
        print("----will send----")
        print(response)

    return jsonify(response), status


def get_class_for_input(user_input):
    if user_input == "e":
        # Publish terminate message to all clients and server
        redisObject.publish_message('TERMINATE','user','EMPTY')
        return "Terminated"
    else:
        # Classify the input using Redis
        redisObject.publish_message('input', 'user', user_input)
    # Listening for incoming messages from Redis and handling user input
    for message in redisObject.get_pubsub().listen():
        print("Listening for messages...")

        if message['type'] == 'message':
            decoded_message = message['data'].decode()
            message_type, sender, message_content = decoded_message.split("#")

            if message_type == "user_response":
                # Extract the classification result from the message
                classification_dict = json.loads(message_content)
                print(f"Classification for the input was {classification_dict}")
                return classification_dict

if __name__ == '__main__':
    # Start the server and client scripts
    server_process = start_process('server.py')
    client1_process = start_process('client.py', ['0'], "_1")
    client2_process = start_process('client.py', ['1'], "_2")
    client3_process = start_process('client.py', ['2', 'true'], "_3")

    try:
        # Setup Redis object for publishing and subscribing
        redisObject = MyRedis()
        user_can_input = True

        app.run(debug=True)
    except:
        print("Error")
    finally:
        redisObject.publish_message('TERMINATE','user','EMPTY')
        print("All processes have been terminated.")

    