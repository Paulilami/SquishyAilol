from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

@app.route('/api/selector', methods=['POST'])
def selector():
    data = request.json
    response = requests.post(
        'http://localhost:8001/selector',
        json={"user_input": data['user_input']}
    )
    return jsonify(response.json())

@app.route('/api/target', methods=['POST'])
def target():
    data = request.json
    response = requests.post(
        f'http://localhost:8001/{data["target"]}',
        json={"user_input": data['prompt']}
    )
    return jsonify(response.json())

if __name__ == '__main__':
    app.run(port=8000)
