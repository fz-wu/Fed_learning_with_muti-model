# Federated k-NN Learning Application
![Python](https://img.shields.io/badge/python-a4c330.svg?style=for-the-badge&logo=python&logoColor=white)
![.ENV](https://img.shields.io/badge/dotenv-ECD53F.svg?style=for-the-badge&logo=dotenv&logoColor=white)
![Redis](https://img.shields.io/badge/redis-%23DD0031.svg?style=for-the-badge&logo=redis&logoColor=white)
![Flask](https://img.shields.io/badge/flask-000000.svg?style=for-the-badge&logo=flask&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![React](https://img.shields.io/badge/react-61DAFB.svg?style=for-the-badge&logo=react&logoColor=white)
![MUI5](https://img.shields.io/badge/mui-007FFF.svg?style=for-the-badge&logo=mui&logoColor=white)
<div align="center">
   <img src="https://github.com/marsafy1/federated-knn/blob/master/knn-frontend/src/assets/turtle.png" alt="Best Turtle" width="500"/>
</div>


<br/>
This repository contains a federated k-Nearest Neighbors (k-NN) learning application built using Python. The application is designed to classify spam messages using a federated learning approach. The setup includes three clients and a main server that aggregates responses using various techniques.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [Dataset](#dataset)
- [Setup](#setup)
- [STRIDE](#stride)

## Overview

Federated learning is a machine learning technique that trains an algorithm across multiple decentralized devices or servers holding local data samples, without exchanging them. In this project, we use federated k-NN to classify spam messages, ensuring data privacy and security by keeping data localized on each client.

## Features

- Federated k-NN classification for spam messages
- Three clients participating in the federated learning process
- Main server for aggregating client responses
- Various aggregation techniques for federated learning
- Secure and private data handling

## Installation

To install and run this project, follow these steps:

1. **Clone the repository:**
   ```sh
   git clone https://github.com/yourusername/federated-knn-learning.git
   cd federated-knn-learning
   ```

2. **Set up a virtual environment:**
   ```sh
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use 'venv\Scripts\activate'
   ```

3. **Install the required packages:**
   ```sh
   pip install -r requirements.txt
   ```

## Usage

To install and run this project, follow these steps:

1. **Option #1:** Using the `.sh` script (starts both the backend and the frontend servers)
   ```sh
   ./start.sh
   ```

2. **Option #2:** Start the servers manually
   - Start the venv:
     ```sh
     source venv/bin/activate  # On Windows, use 'venv\Scripts\activate'
     ```
     
   - Start the backend server:
     ```sh
     cd main && python3 flask_server.py
     ```

   - Start the frontend server:
     ```sh
     cd knn-frontend && npm start
     ```

## Project Structure

```
federated-knn-learning/
├── code
├── datasets
│   ├── dataset_processing.ipynb
│   ├── new_spam.csv
│   └── spam.csv
├── knn-frontend
│   ├── src
│       ├── App.css
│       ├── App.js
│       ├── assets/
│       ├── components/
│       ├── index.css
│       └── index.js
├── main
│   ├── MyRedis.py
│   ├── client.py
│   ├── flask_server.py
│   ├── main.py
│   ├── script_dump
│   │   ├── client_1_output.txt
│   │   ├── client_2_output.txt
│   │   ├── client_3_output.txt
│   │   └── server_output.txt
│   └── server.py
├── requirements.txt
└── start.sh
```

## Technologies Used

- **Python**: The primary programming language for developing the application.
- **Federated Learning**: Technique used to ensure data privacy and security.
- **k-Nearest Neighbors (k-NN)**: The machine learning algorithm used for classification.
- **Redis**: Used for message handling between clients and servers.
- **React**: Used for the frontend.

## Future Work
- [ ] Dynamic number of clients from the frontend
- [ ] Dynamic number of poisoned clients from the frontend
- [ ] Upload any dataset to be used in the model from the frontend
- [ ] Adjust the k in the kNN from the frontend dynamically
      
## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add some feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Open a pull request.

## Dataset

- Total Samples: 39537
- Spam: 17956
- Legit: 21581
  
## Setup

- kNN: we set k here to be 5
- Clients: 3
- Poisoned Clients: 1

## STRIDE
STRIDE is a threat modeling framework used to identify potential security threats in software and hardware systems. It stands for Spoofing, Tampering, Repudiation, Information Disclosure, Denial of Service, and Elevation of Privilege.

### Spoofing
**Threat:**
- An attacker might impersonate a legitimate client or the aggregation server.

**Mitigations:**
- Use strong authentication mechanisms such as OAuth or mutual TLS to verify the identities of clients and the server.
- Implement IP whitelisting to restrict access to known and trusted sources.

### Tampering
**Threat:**
- An attacker could alter data being sent from the clients to the aggregation server or vice versa.
- The poisoned server could inject malicious data into the federated learning process.

**Mitigations:**
- Use end-to-end encryption (e.g., TLS) to protect data integrity during transmission.
- Implement digital signatures to ensure data has not been altered.
- Validate and sanitize data received from all clients, including checking for anomalies that might indicate poisoning.

### Repudiation
**Threat:**
- A client or the aggregation server could deny having sent or received a particular message or transaction.

**Mitigations:**
- Use logging and audit trails to keep track of all interactions and data exchanges between clients and the server.
- Implement non-repudiation mechanisms such as digital signatures.

### Information Disclosure
**Threat:**
- Sensitive data might be exposed during transmission between clients and the aggregation server.
- The poisoned server might attempt to access or disclose sensitive information.

**Mitigations:**
- Use encryption for data in transit (TLS) and at rest.
- Apply strict access controls and permissions to restrict data access.
- Regularly audit and monitor data access patterns for unusual activities.

### Denial of Service (DoS)
**Threat:**
- An attacker might flood the aggregation server or clients with traffic, causing a denial of service.
- The poisoned server could launch a DoS attack to disrupt the federated learning process.

**Mitigations:**
- Implement rate limiting and throttling to prevent flooding attacks.
- Use network-level protections such as firewalls and intrusion detection/prevention systems.
- Employ redundancy and load balancing to handle high traffic volumes.

### Elevation of Privilege
**Threat:**
- An attacker might exploit vulnerabilities to gain higher-level access than permitted, such as administrative control over a client or the aggregation server.

**Mitigations:**
- Follow the principle of least privilege, ensuring that users and services have the minimum level of access necessary.
- Regularly update and patch all systems to fix known vulnerabilities.
- Use multi-factor authentication (MFA) for sensitive operations and administrative access.

### Summary
- **Spoofing:** Implement strong authentication and IP whitelisting.
- **Tampering:** Use encryption, digital signatures, and data validation.
- **Repudiation:** Use logging, audit trails, and digital signatures.
- **Information Disclosure:** Encrypt data, apply access controls, and audit access.
- **Denial of Service:** Implement rate limiting, network protections, and redundancy.
- **Elevation of Privilege:** Follow least privilege, update systems, and use MFA.
