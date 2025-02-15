import cv2
import numpy as np
from ultralytics import YOLO

# Carrega o modelo YOLO pré-treinado (por exemplo, yolov8n.pt)
model = YOLO("yolov8n.pt")

# Caminho do arquivo de vídeo
video_path = "Videos\\video1.mp4"  # Substitua pelo caminho do seu vídeo
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Erro ao abrir o vídeo!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Fim do vídeo ou erro ao ler o frame.")
        break

    # Executa a detecção de objetos no frame
    results = model(frame)

    # Itera sobre os resultados (normalmente um por frame)
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Caixas [x1, y1, x2, y2]
        confs = result.boxes.conf.cpu().numpy()    # Confiança
        classes = result.boxes.cls.cpu().numpy()   # Classes detectadas

        for box, conf, cls in zip(boxes, confs, classes):
            # Considerando que a classe "pessoa" no COCO tem índice 0
            if int(cls) == 0:
                x1, y1, x2, y2 = box.astype(int)
                width = x2 - x1
                height = y2 - y1
                ratio = height / width if width != 0 else 0

                alert = ""
                # Se a razão for muito baixa (exemplo, menor que 1.2), pode indicar que a pessoa está deitada
                if ratio < 1.2:
                    alert = "Queda detectada!"

                # Desenha a caixa e as informações no frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Humano {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2)
                if alert:
                    cv2.putText(frame, alert, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 0, 255), 2)
    
    resized_frame = cv2.resize(frame, (640, 480))

    cv2.imshow("Detecção de Queda (Heurística)", resized_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
