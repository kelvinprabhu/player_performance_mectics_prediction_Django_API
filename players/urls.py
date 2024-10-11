from django.urls import path
from .views import predict_injury, predict_player_rating, predict_player_cluster,fifa_list, fifa_22,upload_file,Player_value

urlpatterns = [
    path('fifaMen/', fifa_22, name='fifa_22'),
    path('fifaList/', fifa_list, name='fifa_list'),
    path('getRating/', upload_file, name='upload_file'),
    path('getValue/', Player_value, name='Player_value'),
    path('predict-injury/', predict_injury, name='predict_injury'),
    path('predict-rating/', predict_player_rating, name='predict_player_rating'),
    path('predict-cluster/', predict_player_cluster, name='predict_player_cluster'),
]
