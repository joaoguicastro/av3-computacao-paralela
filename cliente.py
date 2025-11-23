import socket
import pickle
import numpy
import threading

# --- CONFIGURAÇÃO DOS SERVIDORES ---
SERVERS = [
    ('localhost', 9999),
    ('localhost', 9998)
    # Se adicionar mais servidores, lembre-se de rodar mais scripts server.py
]
# ------------------------------------

def handle_server(server_address, slice_A, matrix_B, result_list, index):
    host, port = server_address
    try:
        # print(f"[Cliente] Conectando ao servidor {host}:{port}...") 
        # (Comentei o print acima para não poluir se for muito rápido)
        
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((host, port))

        # Envia dados
        with s.makefile('wb') as file_wb:
            pickle.dump(slice_A, file_wb)
            pickle.dump(matrix_B, file_wb)

        # Recebe resultado
        with s.makefile('rb') as file_rb:
            result_slice = pickle.load(file_rb)

        # print(f"[Cliente <- {port}] Resultado parcial recebido.")
        result_list[index] = result_slice
        
    except Exception as e:
        print(f"[Cliente] Erro ao conectar com {host}:{port}: {e}")
        result_list[index] = None
    finally:
        s.close()

def main():
    num_servers = len(SERVERS)
    print(f"--- CONFIGURAÇÃO DE MATRIZES ALEATÓRIAS ({num_servers} Servidores) ---")

    # 1. Definição das Dimensões
    while True:
        try:
            rows_A = int(input(f"Digite o número de LINHAS da Matriz A (deve ser múltiplo de {num_servers}): "))
            if rows_A % num_servers == 0 and rows_A > 0:
                break
            else:
                print(f"Erro: O número de linhas deve ser divisível por {num_servers}.")
        except ValueError:
            print("Por favor, digite um número inteiro.")

    cols_A = int(input("Digite o número de COLUNAS da Matriz A (e Linhas da B): "))
    cols_B = int(input("Digite o número de COLUNAS da Matriz B: "))

    print("\n[Cliente] Gerando matrizes aleatórias...")

    # 2. Geração Aleatória (valores entre -10 e 10)
    A = numpy.random.randint(-10, 10, size=(rows_A, cols_A))
    B = numpy.random.randint(-10, 10, size=(cols_A, cols_B))

    # Se as matrizes forem pequenas, imprime na tela. Se forem gigantes, melhor não.
    if rows_A <= 20 and cols_B <= 20:
        print("\n--- Matriz A Gerada ---")
        print(A)
        print("\n--- Matriz B Gerada ---")
        print(B)
    else:
        print("\n(Matrizes muito grandes para exibir no console, prosseguindo...)")

    # 3. Divisão e Processamento
    slices_A = numpy.array_split(A, num_servers, axis=0)
    
    threads = []
    results = [None] * num_servers 

    print(f"\n[Cliente] Distribuindo trabalho para {num_servers} servidores...")

    for i in range(num_servers):
        server_addr = SERVERS[i]
        slice_a = slices_A[i]
        
        t = threading.Thread(target=handle_server, 
                             args=(server_addr, slice_a, B, results, i))
        threads.append(t)
        t.start() 
        
    for t in threads:
        t.join()

    print("[Cliente] Todos os servidores terminaram.")

    # 4. Montagem e Verificação
    if any(r is None for r in results):
        print("Erro: Falha ao receber um ou mais resultados. Abortando.")
        return
        
    C_distribuida = numpy.vstack(results)

    if rows_A <= 20 and cols_B <= 20:
        print("\n--- RESULTADO FINAL (Distribuído) ---")
        print(C_distribuida)

    # Prova Real (Serial)
    print("\n[Cliente] Fazendo verificação serial...")
    C_serial = numpy.dot(A, B)
    
    if numpy.array_equal(C_distribuida, C_serial):
        print("\n✅ Verificação: SUCESSO! O resultado distribuído é igual ao serial.")
    else:
        print("\n❌ Verificação: FALHA! O resultado distribuído está diferente.")

if __name__ == "__main__":
    main()