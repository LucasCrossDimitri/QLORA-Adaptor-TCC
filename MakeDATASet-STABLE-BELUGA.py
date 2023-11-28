input_path = './conversation2.txt'
output_path = './dataset.txt'

try:
    with open(input_path, 'r', encoding='utf-8') as file:
        input_text = file.read()
except FileNotFoundError as e:
    print(f"Erro ao ler o arquivo de entrada: {e}")
    exit()

transformed_lines = []

for line in input_text.split('\n'):
    if line.startswith('User:'):
        transformed_lines.append(f"### Comment: {line[6:]} ")
    elif line.startswith('Assistant:'):
        transformed_lines[-1] += f"### REPLY: {line[11:]} ### END."

transformed_text = '\n'.join(filter(None, transformed_lines))

try:
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(transformed_text)
except IOError as e:
    print(f"Erro ao escrever no arquivo de saída: {e}")
    exit()

print('Transformação completa. Resultado salvo em dataset.txt.')
