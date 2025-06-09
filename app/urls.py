# mydataapi/urls.py
from django.urls import path
from .views import CNNAPIView, NaiveBayesAPIView, SVMAPIView

urlpatterns = [
    path('cnn/', CNNAPIView.as_view(), name='api-cnn'),
    path('naivebayes/', NaiveBayesAPIView.as_view(), name='api-naivebayes'),
    path('svm/', SVMAPIView.as_view(), name='api-svm'),
]