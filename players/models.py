# # from django.db import models
# import joblib
# import pandas as pd
# from pathlib import Path
# import warnings

# # Ignore unpickling warnings
# warnings.filterwarnings("ignore", message=".*Trying to unpickle estimator.*")

# # Load the injury prediction model
# model_injury = Path(__file__).resolve().parent.parent / 'models' / 'injurymodel.pkl'
# model_injury = joblib.load(model_injury)

# # Example test data (one record or more depending on your needs)
# test_data = pd.DataFrame({
#     'Player_Age': [19],  # Change these values as needed
#     'Player_Weight': [55.51],
#     'Player_Height': [170.87],
#     'Previous_Injuries': [0],
#     'Training_Intensity': [0.71],
#     'Recovery_Time': [5],
#     'BMI_Classification_Normal': [1.0],
#     'BMI_Classification_Obesity I': [0.0],
#     'BMI_Classification_Obesity II': [0.0],
#     'BMI_Classification_Overweight': [0.0],
#     'BMI_Classification_Underweight': [0.0],
# })

# # Make predictions
# predictions = model_injury.predict(test_data)

# # Display predictions
# print(predictions)
