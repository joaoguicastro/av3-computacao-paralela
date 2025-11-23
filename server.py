import socket
import pickle
import numpy
import sys

def main():
    if len(sys.argv) != 2:
        print(f"Uso: python {sys.argv[0]} <porta>")
        sys.exit(1)

    host = 'localhost'
    port = int(sys.argv[1]) 

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((host, port))
    s.listen(1)
    
    print(f"[Servidor na porta {port}] Aguardando conexões...")

    try:
        while True:
            conn, addr = s.accept()
            print(f"[Servidor {port}] Conectado por {addr}")

            with conn.makefile('rb') as file_rb:
                submatriz_A = pickle.load(file_rb)
                matriz_B = pickle.load(file_rb)

            print(f"[Servidor {port}] Dados recebidos. Calculando...")

            resultado_parcial = numpy.dot(submatriz_A, matriz_B)
            
            print(f"[Servidor {port}] Cálculo concluído. Enviando resultado...")

            with conn.makefile('wb') as file_wb:
                pickle.dump(resultado_parcial, file_wb)
            
            print(f"[Servidor {port}] Resultado enviado. Fechando conexão.")
            conn.close()

    except KeyboardInterrupt:
        print(f"\n[Servidor {port}] Desligando...")
    finally:
        s.close()

if __name__ == "__main__":
    main()