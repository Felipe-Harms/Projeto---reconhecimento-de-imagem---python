import cv2

# Caminho do arquivo de vídeo
video_path = "Videos\\video1.mp4"

# Abrir o vídeo
cap = cv2.VideoCapture(video_path)

# Checar se o vídeo foi carregado com sucesso
if not cap.isOpened():
    print("Erro ao carregar o vídeo!")
    exit()

while True:
    # Ler um frame do vídeo
    ret, frame = cap.read()
    
    if not ret:
        print("Fim do vídeo ou erro ao ler o frame.")
        break

    # Aplicar algum processamento (exemplo: converter para escala de cinza)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    resized_frame = cv2.resize(gray_frame, (640, 480))  # Ajuste o tamanho conforme necessário

    # Exibir o resultado
    cv2.imshow("Processamento de Vídeo em Tempo Real (Grayscale)", resized_frame)
    
    # Sair ao pressionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar os recursos
cap.release()
cv2.destroyAllWindows()
