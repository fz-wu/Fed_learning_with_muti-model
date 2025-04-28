from collections import Counter
from MyRedis import MyRedis
from collections import Counter
import json

CLIENTS_NUM = 3
default_responses_ack = {}
responses_ack = {}
client_responses = {}
redisObject = MyRedis()

def initialize_response_objects():
    for i in range(0, CLIENTS_NUM):
        default_responses_ack[f"client_{i}"] = False
        responses_ack[f"client_{i}"] = False
        client_responses[f"client_{i}"] = ''
    print(responses_ack)

def all_ack():
    for sender in responses_ack:
        if(not responses_ack[sender]):
            return False
    return True

def reset_acks():
    global responses_ack
    responses_ack = default_responses_ack.copy()



def handle_aggregation(key='most_common', client_responses=None):
    print("Handling aggregation...")
    print(client_responses)
    classification_result = None
    classification_prob = None
    if key == 'most_common':
        classification_result, classification_prob = most_common_aggregation(client_responses=client_responses)
    elif key == 'weighted':
        classification_result, classification_prob = weighted_voting_aggregation()
    elif key == 'average_probabilities':
        classification_result, classification_prob = average_probabilities_aggregation(client_responses=client_responses)
    elif key == 'bayesian':
        classification_result, classification_prob = bayesian_aggregation(client_responses=client_responses)
    print(f"Key: {key} , Aggregation result: {classification_result}")
    # redisObject.publish_message('user_response', 'server', classification_result)
    return classification_result, classification_prob

def most_common_aggregation(client_responses):
    responses = []
    # Iterate over client responses
    spam_count = 0
    for data in client_responses.values():
        class_value = data.get('class', None)
        if class_value is not None:
            print(f"Class value: {class_value}")
            if class_value == 1:
                spam_count += 1
            responses.append(class_value)
    if responses:
        count = Counter(responses)
        most_common_result, _ = count.most_common(1)[0]
        probability = (spam_count / len(responses)) * 100
        return most_common_result, probability
    else:
        print("No valid responses received.")
        return None, 0

def weighted_voting_aggregation(weights={'client1': 3, 'client2': 1, 'client3': 1}):
    vote_counts = {}
    for client, data in client_responses.items():
        predicted_class = data['class']
        vote_counts[predicted_class] = vote_counts.get(predicted_class, 0) + weights.get(client, 1)
    return max(vote_counts, key=vote_counts.get)

def average_probabilities_aggregation(client_responses):
    total_prob_0, total_prob_1 = 0, 0
    num_clients = len(client_responses)
    # Check for empty input to avoid division by zero
    if num_clients == 0:
        return None, 0
    for data in client_responses.values():
        # Safely get the 'spam' value with a default if it's missing
        prob_0 = data.get('spam', None)  # Default to 0 if 'spam' key is missing
        if prob_0 is None:
            continue
        prob_1 = 1 - prob_0
        total_prob_0 += float(prob_0)
        total_prob_1 += float(prob_1)
    average_0 = total_prob_0 / num_clients
    average_1 = total_prob_1 / num_clients
    return ('0' if average_0 > average_1 else '1', average_1)

def bayesian_aggregation(prior_prob_0=0.5, prior_prob_1=0.5, client_responses=None):
    if client_responses is None:
        return None, 0

    # Initialize posterior probabilities with the prior probabilities
    post_prob_0, post_prob_1 = prior_prob_0, prior_prob_1

    # Iterate over each client's response
    for data in client_responses.values():
        # Retrieve the 'spam' value, default to None if it's missing
        prob_0 = data.get('spam', None)
        if prob_0 is None:
            continue  # Skip this iteration if 'spam' key is missing

        # Calculate the complementary probability for not-spam
        prob_1 = 100 - prob_0  # Assuming the incoming probabilities are in percentage form

        # Update the posterior probabilities using the incoming probabilities
        post_prob_0 *= (prob_0 / 100.0)
        post_prob_1 *= (prob_1 / 100.0)

    # Calculate the normalization factor to ensure probabilities sum to 1
    normalization_factor = post_prob_0 + post_prob_1
    if normalization_factor == 0:
        return None, 0  # Return None if normalization is not possible due to zero division

    # Normalize the probabilities
    post_prob_0 /= normalization_factor
    post_prob_1 /= normalization_factor

    # Determine the most likely category based on the highest posterior probability
    return ('0' if post_prob_0 > post_prob_1 else '1', post_prob_1)




if __name__ == '__main__':
    chosen_key = 'most_common'
    print(f"Server up and running...")
    initialize_response_objects()
    can_accept_inputs = True
    # Listen for messages (this will block and wait for messages)
    for message in redisObject.get_pubsub().listen():
        message_type = message['type']
        if message['type'] == 'message':
            message = message['data'].decode()
    
            message_type = message.split("#")[0]
            sender = message.split("#")[1]
            message_content = message.split("#")[2]
    
            print(f'{message_type}@{sender}:', message_content)
            if(message_type == "aggregation_key" and can_accept_inputs):
                chosen_key = message_content
                print(f"Chosen key: {chosen_key}")
    
            if(message_type == "input" and can_accept_inputs):
                redisObject.publish_message('client_task', 'server', message_content)

            if(message_type == "classification"):
                if sender in responses_ack.keys():
                    responses_ack[sender] = True
                    client_message = json.loads(message_content)
                    client_responses[sender] = client_message
                if all_ack():
                    can_accept_inputs = False
                    print('All Received -> Start aggregation')
                    reset_acks()
                    classification_result, classification_prob = handle_aggregation(key = chosen_key, client_responses=client_responses)
                    client_responses['classification'] = {'result':classification_result, "probability": round(classification_prob,2)}
                    redisObject.publish_message('user_response', 'server', json.dumps(client_responses))
                    # handle_aggregation(key = 'weighted')
                    # handle_aggregation(key = 'average_probabilities')
                    # handle_aggregation(key = 'bayesian')
                    can_accept_inputs = True


            
            if(message_type == "TERMINATE"):
                break
    print(f"Server shutting down...")
                    
                    
    



    