#!/bin/sh

# Activate the virtual environment
source env/bin/activate

# Start the Flask server
(cd main && python3 flask_server.py) &

# Navigate to the React app directory and start the npm server
(cd knn-frontend && npm start)