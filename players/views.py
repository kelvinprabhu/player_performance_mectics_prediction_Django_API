from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.http import JsonResponse
from pymongo import MongoClient
from django.views.decorators.csrf import csrf_exempt
import joblib
import pandas as pd
from pathlib import Path
import pandas as pd
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from sklearn.preprocessing import StandardScaler

# # Pre-trained label encoders (use your actual mappings here)
# club_name_encoder = LabelEncoder()
# league_name_encoder = LabelEncoder()
# df = pd.read_csv(r'C:\Users\kelvi\OneDrive\Desktop\semister-1 msaiml\ML\ML_project\dataset_fifa\players_22.csv', low_memory=False)
# club_name_encoder.fit(df['club_name'].unique()) 
# league_name_encoder.fit([df['league_name'].unique()]) 

# scalar = StandardScaler()
client = MongoClient('localhost', 27017)
db = client['player_performance']
model_injury = Path(__file__).resolve().parent.parent / 'models' / 'injurymodel.pkl'
model_injury = joblib.load(model_injury)
rf = Path(__file__).resolve().parent.parent / 'models' / 'rf.pkl'
rf = joblib.load(rf)
lr = Path(__file__).resolve().parent.parent / 'models' / 'lr.pkl'
lr = joblib.load(lr)
model_rating = Path(__file__).resolve().parent.parent / 'models' / 'futureratingsmodel.pkl'
model_rating = joblib.load(model_rating)
kmeans_model = Path(__file__).resolve().parent.parent / 'models' / 'clustermodel.pkl'
kmeans_model = joblib.load(kmeans_model)

from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.http import JsonResponse
import pandas as pd
@csrf_exempt
@api_view(['POST'])
def predict_injury(request):
    try:
        # Extract the features from the POST request
        features = request.data  # Expecting the structure you provided
        
        # Build the DataFrame directly from the request data
        df_features = pd.DataFrame([{
            'Player_Age': features['Player_Age'],
            'Player_Weight': features['Player_Weight'],
            'Player_Height': features['Player_Height'],
            'Previous_Injuries': features['Previous_Injuries'],
            'Training_Intensity': features['Training_Intensity'],
            'Recovery_Time': features['Recovery_Time'],
            'BMI_Classification_Normal': features['BMI_Classification_Normal'],
            'BMI_Classification_Obesity I': features['BMI_Classification_Obesity I'],  # Key with space
            'BMI_Classification_Obesity II': features['BMI_Classification_Obesity II'],  # Key with space
            'BMI_Classification_Overweight': features['BMI_Classification_Overweight'],
            'BMI_Classification_Underweight': features['BMI_Classification_Underweight'],
        }])
        
        # Use the model to make a prediction
        prediction = model_injury.predict(df_features)
        
        # Return the prediction as a JSON response
        return JsonResponse({'prediction': int(prediction[0])})
    
    except KeyError as e:
        # Handle missing keys (i.e., required fields not in the request)
        return Response({'error': f'Missing field in request data: {str(e)}'}, status=400)
    
    except Exception as e:
        # Handle other errors
        return Response({'error': str(e)}, status=400)




@csrf_exempt
@api_view(['POST'])
def upload_file(request):
    if request.method == 'POST' and request.FILES['file']:
        # Save the uploaded file temporarily
        file = request.FILES['file']
        file_name = default_storage.save(file.name, file)
        file_path = default_storage.path(file_name)

        # Read the file (either CSV or Excel)
        if file.name.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file_path)

        # Handle missing values in your columns, just like before
        df['dribbling'] = df['dribbling'].fillna(df['dribbling'].mean())
        df['pace'] = df['pace'].fillna(df['pace'].mean())
        df['shooting'] = df['shooting'].fillna(df['shooting'].mean())

        # Select relevant features
        features = df[['goalkeeping_reflexes','dribbling','attacking_finishing','movement_reactions',
                       'pace','movement_sprint_speed','shooting','age','potential']]

        # Make predictions
        predictions = lr.predict(features)
        df['predictions'] = predictions

        # Return the predictions along with player names or any relevant data
        response_data = df[['short_name', 'predictions', 'overall']].to_dict(orient='records')
        
        return JsonResponse({'predictions': response_data}, safe=False)
    
    return JsonResponse({'error': 'Invalid request or file not provided'}, status=400)

@csrf_exempt
@api_view(['POST'])
def Player_value(request):
    if request.method == 'POST' and 'file' in request.FILES:
        file = request.FILES['file']
        file_name = default_storage.save(file.name, file)
        file_path = default_storage.path(file_name)

        try:
            if file.name.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file.name.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            else:
                return JsonResponse({'error': 'Unsupported file format'}, status=400)

            features = ['age', 'overall', 'potential', 'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']
            target = 'value_eur'

            if target not in df.columns:
                return JsonResponse({'error': f'Missing target column: {target}'}, status=400)

            df.dropna(subset=features + [target], inplace=True)

            max_value = df[target].max()
            bins = [0, 2e6, 8e6, 15e6, max_value]
            labels = ['Low Value', 'Medium Value', 'High Value', 'Very High Value']

            # Ensure bins and labels are correctly used
            if len(bins) - 1 != len(labels):
                return JsonResponse({'error': 'Number of labels must be one less than number of bins'}, status=400)

            df['value_category'] = pd.cut(
                df[target],
                bins=bins,
                labels=labels
            )

            X = df[features]
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            predictions = rf.predict(X)
            df['predictions'] = predictions

            response_data = df[['short_name', 'value_category', 'predictions']].to_dict(orient='records')
            
            return JsonResponse({'value predictions': response_data}, safe=False)

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Invalid request or file not provided'}, status=400)

@csrf_exempt
@api_view(['POST'])
def predict_player_rating(request):
    try:
        # Get player features from the request
        player_data = request.data.get('player')
        
        if not player_data or len(player_data) != 7:
            return JsonResponse({"error": "Player data must contain 7 values."}, status=400)
        
        # Convert to pandas DataFrame
        player_df = pd.DataFrame([player_data], columns=['age', 'potential', 'wage_eur', 'shooting', 'dribbling', 'defending', 'physic'])
        
        # Make prediction
        prediction = model_rating.predict(player_df)
        
        response = {
            "Predicted Overall Rating": prediction[0]
        }
        return JsonResponse(response)
    
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=400)



@csrf_exempt
@api_view(['POST'])
def predict_player_cluster(request):
    try:
        # Get player stats from the request
        player_stats = request.data.get('player_stats')  # Expecting a list of features

        # Validate input data
        if len(player_stats) != 8:  # Adjust based on the number of features used for clustering
            return JsonResponse({"error": "Player stats must contain 8 numerical values."}, status=400)

        # Convert player stats to pandas DataFrame
        player_stats_df = pd.DataFrame([player_stats], columns=['overall', 'potential', 'pace', 'shooting', 
                                                                'passing', 'dribbling', 'defending', 'physic'])
        # scaler = StandardScaler()

        # # Scale the features using the same scaler used for training
        # player_stats_scaled = scaler.fit(player_stats_df)

        # Predict the cluster for the player
        cluster_prediction = kmeans_model.predict(player_stats_df)

        # Build response
        response = {
            "Predicted Cluster": int(cluster_prediction[0])
        }
        return JsonResponse(response)

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=400)

@csrf_exempt
@api_view(['GET'])
def fifa_22(request):
    # Get the search query parameter from the request
    search_name = request.GET.get('name', '')
    
    # Define the MongoDB collection
    collection = db['fifa_2022']
    
    # Create a query filter using regex for partial matching
    query_filter = {}
    if search_name:
        query_filter['short_name'] = {'$regex': search_name, '$options': 'i'}  # Case-insensitive match
    
    # Fetch documents from MongoDB with a limit of 5 results
    fifa_2022 = list(collection.find(query_filter, {
        '_id': 0,  # Exclude the MongoDB '_id' field
        'sofifa_id': 1,
        'player_url': 1,
        'short_name': 1,
        'player_positions': 1,
        'overall': 1,
        'potential': 1,
        'value_eur': 1,
        'wage_eur': 1,
        'player_face_url': 1
    }).limit(5))  # Limit the result to the top 5 matches
    
    return Response(fifa_2022)

@csrf_exempt
@api_view(['GET'])
def fifa_list(request):
    collection = db['fifa_2022']

    # Projection to select only desired fields
    projection = {
        "sofifa_id": 1,
        "player_url": 1,
        "short_name": 1,
        "player_positions": 1,
        "overall": 1,
        "potential": 1,
        "value_eur": 1,
        "wage_eur": 1,
        "player_face_url": 1,
        "_id": 0  # Exclude the _id field
    }

    # Sorting by 'potential' and 'overall' in descending order, then limiting to top 40
    fifa_2022 = list(
        collection.find({}, projection).sort([("potential", -1), ("overall", -1)]).limit(40)
    )

    return Response(fifa_2022)

# @api_view(['GET'])
# def fifa_2021_female(request):
#     collection = db['fifa_2021_female']
#     fifa_2021_female = list(collection.find())
#     for data in fifa_2021_female:
#         data.pop('_id', None)
#     return Response(fifa_2021_female)
