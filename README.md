# Machine Learning-Aided Platform for Point-of-Care Pregnancy Risk Assessment from 2D Ultrasound

## Overview
This project aims to develop a machine learning (ML)-aided platform for point-of-care pregnancy risk assessment using 2D ultrasound images. The platform utilizes state-of-the-art ML models to analyze ultrasound images and provide risk assessments for pregnancy-related complications.

## Features
- Automated analysis of 2D ultrasound images.
- Prediction of pregnancy-related complications such as fetal abnormalities, placental issues, and maternal health risks.
- User-friendly interface for healthcare professionals to input ultrasound images and receive risk assessments.
- Integration with existing healthcare systems for seamless adoption.

## Installation
1. Install dependencies in requirements.txt

## Usage
1. Run the training script "train.py" while using MLFlow for model logging & tracking.
2. Use MLFlow to Register the best trained model and transition it production stage.
3. Run the CMD: !mlflow models serve --model-uri models:/{model_name}/production -p 7777 --no-conda to create a model serving endpoint.
4. Access the model endpoint through your web browser at `http://localhost:7777/invocations`.
5. Run the dashboard using 'streamlit run streamlit_dashboard.py'
6. Upload 2D ultrasound images for analysis.
7. Receive risk assessments and recommendations based on ML analysis.

## Contributors
- Ronald Omoding (ronaldomoding130@gmail.com)

## Acknowledgements
Special thanks to HASH for their support and collaboration on this project.
