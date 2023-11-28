# Nome do arquivo a ser verificado
nome_do_arquivo = 'D:\QProj\datasetFinalStable-Beluga.txt'

# Abrir o arquivo e iterar pelas linhas
with open(nome_do_arquivo, 'r', encoding='utf-8') as arquivo:
    for numero_linha, linha in enumerate(arquivo, start=1):
        # Verificar se a linha tem mais de 1200 caracteres
        if len(linha) > 1000:
            # Imprimir a linha e o número correspondente
            print(f'Linha {numero_linha}: {linha}')