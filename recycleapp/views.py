# recycleapp/views.py
import base64
import io
import os.path

import torch
import operations.image_processing as im
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
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

                # Desenhar as bounding boxes na imagem
                draw = ImageDraw.Draw(processed_image)
                font = ImageFont.load_default()
                for box, confidence, cls in zip(boxes, confidences, classes):
                    draw.rectangle(box, outline="red", width=2)
                    text = f"{cls}: {confidence:.2f}"
                    text_size = draw.textbbox((0, 0), text, font=font)  # Obtém a bounding box do texto
                    text_width = text_size[2] - text_size[0]
                    text_height = text_size[3] - text_size[1]
                    text_location = [box[0], box[1] - text_height] if box[1] - text_height > 0 else [box[0], box[1] + text_height]
                    draw.rectangle([text_location, [text_location[0] + text_width, text_location[1] + text_height]], fill="red")
                    draw.text(text_location, text, fill="white", font=font)

            # Convert processed image to base64
            buffered = io.BytesIO()
            processed_image.save(buffered, format="PNG")
            processed_img_str = base64.b64encode(buffered.getvalue()).decode()

            return JsonResponse({'processed_img_data': processed_img_str, 'results': processed_results}, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
