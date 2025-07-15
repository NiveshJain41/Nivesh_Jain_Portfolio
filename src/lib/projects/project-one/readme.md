SETUP

Clone the project bash git clone 
>>https://github.com/your-username/amazon-trust.git 
>>cd amazon-trust

Install required packages bash 
>>pip install streamlit pandas numpy scikit-learn joblib xgboost lightgbm openpyxl google-generativeai

Run the app bash 
>>streamlit run main.py

Gemini AI Setup (Optional) core_logic.py, replace: GEMINI_API_KEY = "your-api-key"

Notes This is a prototype project for HackOn — not for production use Review and purchase data are saved as .xlsx files Trained models are saved as .pkl in models/