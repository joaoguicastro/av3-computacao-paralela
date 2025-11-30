import socket
import pickle
import numpy as np
import threading
import time
import argparse
from typing import List, Tuple, Optional

SERVERS: List[Tuple[str, int]] = [
    ('localhost', 9999),
    ('localhost', 9998),
    ('localhost', 9997),
]

def handle_server(
    server_address: Tuple[str, int],
    slice_A: np.ndarray,
    matrix_B: np.ndarray,
    result_list: List[Optional[np.ndarray]],
    index: int
) -> None:
    host, port = server_address
    start = time.perf_counter()

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((host, port))

        with s.makefile('wb') as file_wb:
            pickle.dump(slice_A, file_wb, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(matrix_B, file_wb, protocol=pickle.HIGHEST_PROTOCOL)

        with s.makefile('rb') as file_rb:
            result_slice = pickle.load(file_rb)

        elapsed = time.perf_counter() - start
        print(f"[Cliente <- {host}:{port}] Slice concluída em {elapsed:.4f}s "
              f"(linhas: {slice_A.shape[0]})")

        result_list[index] = result_slice

    except Exception as e:
        print(f"[Cliente] Erro ao conectar com {host}:{port}: {e}")
        result_list[index] = None
    finally:
        s.close()


def pedir_dimensoes(num_servers: int) -> Tuple[int, int, int]:
    print(f"--- CONFIGURAÇÃO DE MATRIZES ALEATÓRIAS ({num_servers} Servidores) ---")

    while True:
        try:
            rows_A = int(
                input(
                    f"Digite o número de LINHAS da Matriz A "
                    f"(deve ser múltiplo de {num_servers}): "
                )
            )
            if rows_A % num_servers == 0 and rows_A > 0:
                break
            else:
                print(f"Erro: O número de linhas deve ser divisível por {num_servers}.")
        except ValueError:
            print("Por favor, digite um número inteiro.")

    while True:
        try:
            cols_A = int(input("Digite o número de COLUNAS da Matriz A (e LINHAS da B): "))
            if cols_A > 0:
                break
            else:
                print("Valor deve ser maior que 0.")
        except ValueError:
            print("Por favor, digite um número inteiro.")

    while True:
        try:
            cols_B = int(input("Digite o número de COLUNAS da Matriz B: "))
            if cols_B > 0:
                break
            else:
                print("Valor deve ser maior que 0.")
        except ValueError:
            print("Por favor, digite um número inteiro.")

    return rows_A, cols_A, cols_B


def gerar_matrizes(rows_A: int, cols_A: int, cols_B: int, seed: Optional[int] = None):
    if seed is not None:
        np.random.seed(seed)

    print("\n[Cliente] Gerando matrizes aleatórias...")
    A = np.random.randint(-10, 10, size=(rows_A, cols_A))
    B = np.random.randint(-10, 10, size=(cols_A, cols_B))

    if rows_A <= 20 and cols_B <= 20:
        print("\n--- Matriz A Gerada ---")
        print(A)
        print("\n--- Matriz B Gerada ---")
        print(B)
    else:
        print("\n(Matrizes muito grandes para exibir no console, prosseguindo...)")

    return A, B


def multiplicacao_distribuida(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    num_servers = len(SERVERS)
    slices_A = np.array_split(A, num_servers, axis=0)

    threads = []
    results: List[Optional[np.ndarray]] = [None] * num_servers

    print(f"\n[Cliente] Distribuindo trabalho para {num_servers} servidores...")

    start_dist = time.perf_counter()

    for i in range(num_servers):
        server_addr = SERVERS[i]
        slice_a = slices_A[i]

        t = threading.Thread(
            target=handle_server,
            args=(server_addr, slice_a, B, results, i),
            daemon=True
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    elapsed_dist = time.perf_counter() - start_dist
    print(f"[Cliente] Todos os servidores terminaram em {elapsed_dist:.4f}s.")

    if any(r is None for r in results):
        raise RuntimeError("Erro: Falha ao receber um ou mais resultados dos servidores.")

    C_distribuida = np.vstack(results)
    return C_distribuida


def multiplicacao_serial(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, float]:
    print("\n[Cliente] Fazendo multiplicação serial (prova real)...")
    start = time.perf_counter()
    C = np.dot(A, B)
    elapsed = time.perf_counter() - start
    print(f"[Cliente] Multiplicação serial concluída em {elapsed:.4f}s.")
    return C, elapsed


def comparar_resultados(C_dist: np.ndarray, C_serial: np.ndarray) -> None:
    if np.array_equal(C_dist, C_serial):
        print("\n✅ Verificação: SUCESSO! O resultado distribuído é igual ao serial.")
    else:
        diff_norm = np.linalg.norm(C_dist - C_serial)
        print("\n❌ Verificação: FALHA! O resultado distribuído está diferente.")
        print(f"Norma da diferença ||C_dist - C_serial|| = {diff_norm:.6f}")


def main():
    parser = argparse.ArgumentParser(
        description="Cliente de multiplicação de matrizes distribuída (Computação Paralela/Concorrente)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed opcional para geração reprodutível das matrizes."
    )
    args = parser.parse_args()

    num_servers = len(SERVERS)
    if num_servers == 0:
        print("Nenhum servidor configurado em SERVERS. Encerrando.")
        return

    rows_A, cols_A, cols_B = pedir_dimensoes(num_servers)

    print("\n--- RESUMO DA CONFIGURAÇÃO ---")
    print(f"Matriz A: {rows_A} x {cols_A}")
    print(f"Matriz B: {cols_A} x {cols_B}")
    print(f"Número de servidores: {num_servers}")
    if args.seed is not None:
        print(f"Seed aleatória: {args.seed}")
    print("------------------------------")

    A, B = gerar_matrizes(rows_A, cols_A, cols_B, seed=args.seed)

    try:
        inicio_dist = time.perf_counter()
        C_distribuida = multiplicacao_distribuida(A, B)
        tempo_total_dist = time.perf_counter() - inicio_dist
    except RuntimeError as e:
        print(e)
        return

    if rows_A <= 20 and cols_B <= 20:
        print("\n--- RESULTADO FINAL (Distribuído) ---")
        print(C_distribuida)

    C_serial, tempo_serial = multiplicacao_serial(A, B)

    comparar_resultados(C_distribuida, C_serial)

    print("\n=== RESUMO DE DESEMPENHO ===")
    print(f"Tempo TOTAL (distribuído + comunicação): {tempo_total_dist:.4f}s")
    print(f"Tempo serial (numpy.dot local):          {tempo_serial:.4f}s")

    if tempo_total_dist > 0:
        speedup = tempo_serial / tempo_total_dist
        print(f"Speedup (serial / distribuído):          {speedup:.4f}x")
    else:
        print("Speedup: não foi possível calcular (tempo distribuído ≈ 0).")
    print("=============================")


if __name__ == "__main__":
    main()
