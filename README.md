# House Price Prediction

This project is a web application for predicting Melbourne, Australia house prices based on various features using a trained machine learning model. The application is built using Streamlit and uses a pre-trained XGBoost model for making predictions.

## Features

- Jupyter notebook to analyze complete model creation flow
- User-friendly interface to input house features
- Predict house prices based on user inputs
- Uses custom transformers for encoding categorical variables

## Project Structure
.
├── app.py # Main application file
├── Cleaned_df.pkl # Cleaned dataset used for user input options
├── best_xgb_model.joblib # Pre-trained XGBoost model
├── custom_transformers.py # Custom transformers for encoding
├── static
│ └── image9.jpg # Image displayed in the application
├── README.md # Project documentation
└── requirements.txt # Python dependencies


## Getting Started

### Prerequisites

- Python 3.7 or higher
- Pip (Python package installer)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/house-price-prediction.git
   cd house-price-prediction

   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

2. Create a virtual environment and activate it

   ```bash
   conda create --name housing_venv python=3.8
   conda activate housing_venv

3. Install the required packages

   ```bash
   pip install -r requirements.txt

### Running the Application

1. Ensure that Cleaned_df.pkl, best_xgb_model.joblib, and housing.png are present in the appropriate directories.

2. Run the Streamlit application

   ```bash
   streamlit run app.py

3. Open your web browser and go to http://localhost:8501 to access the application.



