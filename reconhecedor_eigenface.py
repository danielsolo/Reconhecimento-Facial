import cv2
from cv2 import detail_ChannelsCompensator

detectorFace = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
reconhecedor = cv2.face.EigenFaceRecognizer_create()
reconhecedor.read('classificadorEigen.yml')
largura, altura = 220, 220
font = cv2.FONT_ITALIC
camera = cv2.VideoCapture(0)
idpessoa = int(input('digite seu id: '))

while (True):
    conectado, imagem = camera.read()
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    facesDetectadas = detectorFace.detectMultiScale(imagem, scaleFactor = 1.5, minSize=(50,50))

    for (x, y, l, a) in facesDetectadas:
        imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (largura,altura) )
        cv2.rectangle(imagem, (x,y), (x+l, y+a), (0,0,255), 2)
        id, confianca = reconhecedor.predict(imagemFace)
        
        if id == 1:
            nome = 'Daniel com oculos'
            
        if id == 2:
            nome = 'Daniel sem oculos'
        
         
        

        cv2.putText(imagem, nome, (x,y + (a-280)), font, 0.8, (0,255,0))
        cv2.putText(imagem, str(confianca), (x,y + (a+20 )), font, 0.6, (0,255,0))
        cv2.putText(imagem, 'Severino - Cara Cracha', (x,y + (a-320 )), font, 1, (255,0,0))
    
    
    cv2.imshow('Severino - Cara Cracha', imagem)
     
    if cv2.waitKey(1) == ord('q'):
        break

if id == idpessoa:
    print(f'ABRINDO PORTA - BEM VINDO {nome}')
else:
    print('ACESSO NEGADO')
camera.release()
cv2.destroyAllWindows()