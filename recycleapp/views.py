import base64
import io
import os.path

import torch
import operations.image_processing as im
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from ultralytics import YOLO
from PIL import Image
from django.http import JsonResponse
from .serializers import ImageSerializer
from recycle import settings


class ImageUploadAndProcessView(APIView):
    def post(self, request):
        serializer = ImageSerializer(data=request.data)
        if serializer.is_valid():
            image = serializer.validated_data['image']
            img = Image.open(image)

            # Process the image
            processed_image = im.image_processing(img)

            model = YOLO(os.path.join(settings.BASE_DIR, 'best.pt'))
            torch.cuda.set_device(0)
            results = model(processed_image, device='cuda' if torch.cuda.is_available() else 'cpu')

            # Processar os resultados para JSON amigável
            processed_results = []
            for result in results:
                boxes = result.boxes.xyxy.tolist()  # Bounding box coordinates
                confidences = result.boxes.conf.tolist()  # Confidence scores
                classes = result.boxes.cls.tolist()  # Class IDs
                processed_results.append({
                    'boxes': boxes,
                    'confidences': confidences,
                    'classes': classes
                })

            return JsonResponse({'results': processed_results}, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
