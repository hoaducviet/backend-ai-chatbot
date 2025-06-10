from django.shortcuts import render

# Create your views here.

# coreapi/views.py
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework import serializers

from app.service.cnn_service import get_cnn_model_response
from app.service.naivebayes_service import get_naivebayes_model_response

class ChatbotInputSerializer(serializers.Serializer):
    message = serializers.CharField(max_length=1000, help_text="Tin nhắn của người dùng gửi tới chatbot.")


class CNNAPIView(APIView):
    
    def post(self, request, format=None):
        print("CNN Model")
        serializer = ChatbotInputSerializer(data=request.data)
        if serializer.is_valid():
            user_message = serializer.validated_data['message']

            chatbot_output = get_cnn_model_response(user_message)

            # Kiểm tra trạng thái trả về từ get_response
            if chatbot_output.get("status") == "success":
                return Response(chatbot_output, status=status.HTTP_200_OK)
            else:
                # Nếu get_response trả về lỗi, phản hồi với trạng thái 500 hoặc 400 tùy theo lỗi
                return Response(chatbot_output, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        else:
            # Dữ liệu đầu vào không hợp lệ từ người dùng
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class NaiveBayesAPIView(APIView):
    
    def post(self, request, format=None):
        print("Naive Bayes Model")
        serializer = ChatbotInputSerializer(data=request.data)
        if serializer.is_valid():
            user_message = serializer.validated_data['message']

            chatbot_output = get_naivebayes_model_response(user_message)
            # Kiểm tra trạng thái trả về từ get_response
            if chatbot_output.get("status") == "success":
                return Response(chatbot_output, status=status.HTTP_200_OK)
            else:
                # Nếu get_response trả về lỗi, phản hồi với trạng thái 500 hoặc 400 tùy theo lỗi
                return Response(chatbot_output, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        else:
            # Dữ liệu đầu vào không hợp lệ từ người dùng
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class SVMAPIView(APIView):

    def post(self, request, format=None):
        print("SVM Model")
        serializer = ChatbotInputSerializer(data=request.data)
        if serializer.is_valid():
            user_message = serializer.validated_data['message']

            chatbot_output = get_naivebayes_model_response(user_message)
            # Kiểm tra trạng thái trả về từ get_response
            if chatbot_output.get("status") == "success":
                return Response(chatbot_output, status=status.HTTP_200_OK)
            else:
                # Nếu get_response trả về lỗi, phản hồi với trạng thái 500 hoặc 400 tùy theo lỗi
                return Response(chatbot_output, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        else:
            # Dữ liệu đầu vào không hợp lệ từ người dùng
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)