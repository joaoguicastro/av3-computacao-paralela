import socket
import pickle
import numpy as np
import sys
import threading
import time


def multiplica_paralelo(submatriz_A: np.ndarray, matriz_B: np.ndarray) -> np.ndarray:
    return np.dot(submatriz_A, matriz_B)


def handle_connection(conn: socket.socket, addr, port: int) -> None:
    print(f"[Servidor {port}] Conectado por {addr}")

    try:
        with conn.makefile('rb') as file_rb:
            submatriz_A = pickle.load(file_rb)
            matriz_B = pickle.load(file_rb)

        print(
            f"[Servidor {port}] Dados recebidos. "
            f"Dimensões: A_sub={submatriz_A.shape}, B={matriz_B.shape}"
        )

        inicio_calc = time.perf_counter()
        resultado_parcial = multiplica_paralelo(submatriz_A, matriz_B)
        tempo_calc = time.perf_counter() - inicio_calc

        print(
            f"[Servidor {port}] Cálculo concluído em {tempo_calc:.4f}s. "
            f"Resultado: {resultado_parcial.shape}"
        )

        with conn.makefile('wb') as file_wb:
            pickle.dump(resultado_parcial, file_wb, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"[Servidor {port}] Resultado enviado. Fechando conexão com {addr}.")

    except Exception as e:
        print(f"[Servidor {port}] Erro ao processar conexão de {addr}: {e}")
    finally:
        conn.close()


def main():
    if len(sys.argv) != 2:
        print(f"Uso: python {sys.argv[0]} <porta>")
        sys.exit(1)

    host = 'localhost'
    port = int(sys.argv[1])

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    s.bind((host, port))
    s.listen(5)

    print(f"[Servidor na porta {port}] Aguardando conexões... (Ctrl+C para sair)")

    try:
        while True:
            conn, addr = s.accept()

            t = threading.Thread(
                target=handle_connection,
                args=(conn, addr, port),
                daemon=True
            )
            t.start()

    except KeyboardInterrupt:
        print(f"\n[Servidor {port}] Encerrando por KeyboardInterrupt...")
    finally:
        s.close()
        print(f"[Servidor {port}] Socket fechado.")


if __name__ == "__main__":
    main()
