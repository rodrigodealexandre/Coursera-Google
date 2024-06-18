from flask import Flask, request, render_template_string
import pickle

# Load the trained model
model = pickle.load(open('best_rf_model.pkl', 'rb'))

# Initialize Flask app
app = Flask(__name__)

# Define the homepage route
@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    explanation = None
    if request.method == 'POST':
        if request.is_json:
            data = request.get_json(force=True)
            features = data['features']
        else:
            if 'features' in request.form:
                features = list(map(float, request.form['features'].split(',')))
            else:
                features = [
                    float(request.form['satisfaction_level']),
                    float(request.form['number_of_projects']),
                    float(request.form['years_at_company']),
                    float(request.form['average_monthly_hours']),
                    float(request.form['last_evaluation']),
                    float(request.form['satisfaction_x_salary_low']),
                    float(request.form['satisfaction_x_salary_medium']),
                    float(request.form['satisfaction_x_sales']),
                    float(request.form['satisfaction_x_technical']),
                    float(request.form['satisfaction_x_support'])
                ]
        
        prediction = model.predict([features])[0]
        explanation = "Employee will leave the company." if prediction == 1 else "Employee will stay in the company."

    # Validation results
    validation_results = {
        "Tuned Accuracy": 0.9812421842434348,
        "Tuned Precision": 0.981081081081081,
        "Tuned Recall": 0.9052369077306733,
        "Tuned F1 Score": 0.9416342412451362,
        "Tuned AUC": 0.9748320639841836,
        "Drift Detected": 0  # Example drift result
    }

    return render_template_string('''
        <!doctype html>
        <title>Employee Turnover Prediction for Salifort Motors</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #f4f4f4;
                color: #333;
                margin: 0;
                padding: 0;
            }
            .container {
                display: flex;
                flex-wrap: wrap;
                max-width: 1200px;
                margin: 20px auto;
                padding: 20px;
                background-color: white;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                border-radius: 8px;
            }
            .left-pane, .right-pane {
                flex: 1;
                padding: 20px;
                box-sizing: border-box;
            }
            .right-pane {
                border-left: 2px solid #ccc;
            }
            form {
                margin-bottom: 20px;
            }
            input[type="text"], input[type="submit"] {
                width: 100%;
                padding: 10px;
                margin: 5px 0 10px 0;
                border: 1px solid #ccc;
                border-radius: 4px;
            }
            input[type="submit"] {
                background-color: #007CFF;
                color: white;
                border: none;
                cursor: pointer;
            }
            input[type="submit"]:hover {
                background-color: #005bb5;
            }
            h1 {
                color: #007CFF;
            }
            h2 {
                color: #333;
            }
            h3 {
                color: #007CFF;
            }
            ul {
                list-style-type: none;
                padding: 0;
            }
            li {
                background: #f9f9f9;
                margin: 5px 0;
                padding: 10px;
                border-radius: 4px;
            }
            .result-box {
                background: #f9f9f9;
                padding: 20px;
                border-radius: 8px;
                text-align: center;
                margin-top: 20px;
                color: white;
            }
            .result-box p {
                margin: 0;
                font-size: 1.2em;
            }
            .stay {
                background-color: #52CC52;
            }
            .leave {
                background-color: #FF3B30;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }
            table, th, td {
                border: 1px solid #ccc;
            }
            th, td {
                padding: 10px;
                text-align: left;
            }
            th {
                background-color: #007CFF;
                color: white;
            }
            .image-container {
                display: flex;
                justify-content: center;
                align-items: center;
                flex-direction: column;
            }
            .image-container img {
                max-width: 90%;
                height: auto;
                margin: 10px;
                border: 1px solid #ccc;
                border-radius: 8px;
            }
        </style>
        <h1>Employee Turnover Prediction for Salifort Motors</h1>
        <div class="container">
            <div class="left-pane">
                <h2>Enter values manually:</h2>
                <form action="/" method="post">
                    <label for="satisfaction_level">Satisfaction Level:</label><br>
                    <input type="text" id="satisfaction_level" name="satisfaction_level"><br>
                    <label for="number_of_projects">Number of Projects:</label><br>
                    <input type="text" id="number_of_projects" name="number_of_projects"><br>
                    <label for="years_at_company">Years at Company:</label><br>
                    <input type="text" id="years_at_company" name="years_at_company"><br>
                    <label for="average_monthly_hours">Average Monthly Hours:</label><br>
                    <input type="text" id="average_monthly_hours" name="average_monthly_hours"><br>
                    <label for="last_evaluation">Last Evaluation:</label><br>
                    <input type="text" id="last_evaluation" name="last_evaluation"><br>
                    <label for="satisfaction_x_salary_low">Satisfaction x Salary Low:</label><br>
                    <input type="text" id="satisfaction_x_salary_low" name="satisfaction_x_salary_low"><br>
                    <label for="satisfaction_x_salary_medium">Satisfaction x Salary Medium:</label><br>
                    <input type="text" id="satisfaction_x_salary_medium" name="satisfaction_x_salary_medium"><br>
                    <label for="satisfaction_x_sales">Satisfaction x Sales:</label><br>
                    <input type="text" id="satisfaction_x_sales" name="satisfaction_x_sales"><br>
                    <label for="satisfaction_x_technical">Satisfaction x Technical:</label><br>
                    <input type="text" id="satisfaction_x_technical" name="satisfaction_x_technical"><br>
                    <label for="satisfaction_x_support">Satisfaction x Support:</label><br>
                    <input type="text" id="satisfaction_x_support" name="satisfaction_x_support"><br><br>
                    <input type="submit" value="Predict">
                </form>

                <h2>Or, enter features values (comma-separated):</h2>
                <form action="/" method="post">
                    <label for="features">Features (comma-separated):</label><br>
                    <input type="text" id="features" name="features" size="100"><br><br>
                    <input type="submit" value="Predict">
                </form>
                <p>Feature order:</p>
                <ul>
                    <li>satisfaction_level</li>
                    <li>number_of_projects</li>
                    <li>years_at_company</li>
                    <li>average_monthly_hours</li>
                    <li>last_evaluation</li>
                    <li>satisfaction_x_salary_low</li>
                    <li>satisfaction_x_salary_medium</li>
                    <li>satisfaction_x_sales</li>
                    <li>satisfaction_x_technical</li>
                    <li>satisfaction_x_support</li>
                </ul>
                <!-- Examples -->
                <h3>Examples:</h3>
                <ul>
                    <li>Example 1: 0.38, 2, 3, 157, 0.53, 0.38, 0.0, 0.38, 0.0, 0.0</li>
                    <li>Example 2: 0.80, 5, 6, 262, 0.86, 0.0, 0.80, 0.80, 0.0, 0.0</li>
                    <li>Example 3: 0.11, 7, 4, 272, 0.88, 0.11, 0.0, 0.11, 0.0, 0.0</li>
                </ul>
            </div>
            <div class="right-pane">
                {% if prediction is not none %}
                    <div class="result-box {{ 'stay' if prediction == 0 else 'leave' }}">
                        <h2>Prediction Result:</h2>
                        <p>Prediction: {{ prediction }} - {{ explanation }}</p>
                    </div>
                    <h2>Model Validation Results</h2>
                    <table>
                        <tr>
                            <th>Metric</th>
                            <th>Score</th>
                        </tr>
                        {% for metric, score in validation_results.items() %}
                        <tr>
                            <td>{{ metric }}</td>
                            <td>{{ score }}</td>
                        </tr>
                        {% endfor %}
                    </table>
                    <div class="image-container">
                        <h3>Model Performance</h3>
                        <img src="/static/roc_curve.png" alt="ROC Curve">
                        <img src="/static/confusion_matrix_tuned.png" alt="Confusion Matrix">
                        <img src="/static/learning_curves.png" alt="Learning Curves">
                    </div>
                    <h3>Explanation</h3>
                    <p>The model's performance metrics indicate a high accuracy, precision, recall, and F1 score, suggesting that the model is very effective at predicting employee turnover. The AUC score of 0.97 further confirms the model's ability to distinguish between employees who will leave and those who will stay. The drift detection score indicates whether the current data distribution has significantly changed from the training data, which can be an important factor in maintaining model reliability over time.</p>
                {% endif %}
            </div>
        </div>
    ''', prediction=prediction, explanation=explanation, validation_results=validation_results)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
