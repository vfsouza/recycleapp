# recycleapp/views.py
import base64
import io
import os.path

import torch
import operations.image_processing as im
import cv2
import numpy as np
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
            results = model(processed_image, device='cuda' if torch.cuda.is_available() else 'cpu')

            # Use the Ultralytics YOLO library to draw the bounding boxes
            annotated_image = results[0].plot()  # Get annotated image with bounding boxes

            # Convert annotated image to PIL Image
            img = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            img = Image.fromarray(img)

            # Convert processed image to base64
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            processed_img_str = base64.b64encode(buffered.getvalue()).decode()

            return JsonResponse({'processed_img_data': processed_img_str}, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
