from flask import Flask, request, render_template, jsonify
import training
import inference
import optimization

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == 'POST':
        return training.train(request)  # Call the train function from training.py
    return render_template('train.html')

@app.route('/infer', methods=['GET', 'POST'])
def infer():
    if request.method == 'POST':
        return inference.infer(request)  # Call the infer function from inference.py
    return render_template('infer.html')

@app.route('/optimize', methods=['GET', 'POST'])
def optimize():
    if request.method == 'POST':
        return optimization.optimize(request)  # Call the optimize function from optimization.py
    return render_template('optimize.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)
