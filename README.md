### 🤩Our model was EXTREMELY accurate (99.2% excluding 7 anomolies)... View photos of our results here: https://drive.google.com/drive/folders/1pOnC3sicM7yjOkLnPF0cNde9aw6s4GUf?usp=sharing


### 🌋 Seismic Anomaly Detection

Welcome to the Seismic Anomaly Detection project! This repository contains a Flask-based web application and a local prediction script (predict.py) to identify seismic anomalies from uploaded CSV files. Use the live web app for a streamlined experience or run the predict.py script locally if the site is temporarily unavailable.

### View our predictions on the test data in the outputs folder! We had a 99.2% accuracy overall.

### 🔗 Live Website
#The Flask app is available here: https://nasa.thebayre.com.

Note: The website may not always be up and running. In that case, use the predict.py script for local testing.

### 📂 Folder Structure
arduino
## Copy code
├── README.md
├── app.py
├── static/
├── templates/
├── predict.py
├── requirements.txt
└── sample_data/
app.py: The main Flask application.
predict.py: The local prediction script to test seismic anomaly detection.
sample_data/: Folder containing sample CSV files to test the model.
static/ and templates/: Used for Flask’s web interface.
requirements.txt: List of all dependencies required for the project.
### 🚀 Getting Started
Prerequisites
Make sure you have the following installed on your machine:

Python 3.7+
pip (Python package installer)
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/your-username/seismic-detection.git
cd seismic-detection
Install the required dependencies:

bash
Copy code
pip install -r requirements.txt
(Optional) Create a virtual environment:

bash
Copy code
python -m venv venv
source venv/bin/activate   # On Windows, use `venv\Scripts\activate`

### 🌐 Using the Web Application
You can access the live web application at https://nasa.thebayre.com.

Upload a CSV file using the file selector.
Submit the form to see the generated PNG plot showing detected seismic anomalies.
View Results: The PNG image will display the identified seismic events based on the uploaded CSV.
Tip: Ensure that your CSV file follows the required format, similar to the ones found in the sample_data/ folder.

### 🖥️ Running the predict.py Script Locally
If the web app is not accessible, you can use the local prediction script to get results from your seismic data.

Usage
Navigate to the project directory:

bash
Copy code
cd seismic-detection
Run the predict.py script with a sample CSV file:

bash
Copy code
python predict.py --file sample_data/test_file.csv
Output
The script will analyze the input CSV and print out the start time of any detected seismic event.

Command Line Arguments
--file : The path to the CSV file you want to analyze.
Example:

bash
Copy code
python predict.py --file path/to/your/file.csv
### 📄 Sample CSV Files
You can find sample CSV files for testing in the sample_data/ folder. Make sure to follow the same structure for any new files you want to analyze locally or on the web app.

### 🛠️ Development
Want to improve the app? Here’s how you can set up the project for development:

Run the Flask app locally:

bash
Copy code
python app.py
Open the browser and go to http://127.0.0.1:5000 to view the local instance of the app.

### 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page if you want to contribute.

### 📝 License
This project is licensed under the MIT License.

### 📧 Contact
For any questions or feedback, feel free to reach out at brunoayre06@gmail.com.

