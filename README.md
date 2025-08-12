🩺 AI-Powered Healthcare Chatbot
📌 Project Overview
This project is an AI-based healthcare chatbot that predicts possible diseases based on the user’s symptoms using Machine Learning (Decision Tree Classifier & SVM). It interacts with the user via text and text-to-speech (TTS), provides disease descriptions, precautionary measures, and severity analysis, helping users decide whether to consult a doctor.

It uses symptom severity, description, and precaution datasets to guide the conversation and make accurate predictions.

✨ Features
✅ Symptom-based disease prediction using Decision Tree Classifier.

✅ Secondary prediction for better accuracy.

✅ Support Vector Machine (SVM) model for performance comparison.

✅ Interactive text & voice interface with pyttsx3.

✅ Symptom severity analysis to suggest doctor consultation if needed.

✅ Displays disease description & precautions.

✅ Data visualization of disease distribution.

🛠️ Tech Stack
Languages: Python

Libraries & Frameworks:

Data Processing: NumPy, Pandas, scikit-learn

Visualization: Matplotlib, Seaborn

ML Models: DecisionTreeClassifier, SVM (Support Vector Classification)

Text-to-Speech: pyttsx3

Utilities: re, csv, warnings

📂 Dataset Files
Training.csv → Contains symptoms (features) and diseases (target).

Testing.csv → Used for testing and evaluation.

Symptom_severity.csv → Symptom-to-severity mapping.

symptom_Description.csv → Disease descriptions.

symptom_precaution.csv → Precautionary measures for diseases.
