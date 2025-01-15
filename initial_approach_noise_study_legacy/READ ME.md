Os notebooks nessa pasta consideram uma abordagem inicial para o estudo do efeito do ruído no espectrograma.

Nessa abordagem, cria-se datasets com métricas do espectro, espectrograma e histograma do espectrograma obtidos de transientes com diferentes níveis de ruído.

A ideia é conseguir estudar como tais métricas variam com o nível de ruído adicionado aos transientes. Busca-se obter uma compreensão de transformações na imagem provocadas pelo ruído mais elevado.

Métricas consideradas:

Métricas do espectro: SNR, nivel de ruído (STD) e valor máximo do espectro na região do GABA.
Métricas do espectrograma: média, mediana, desvio padrão, trace, TV anisotrópica, TV isotrópica, soma - dividindo a imagem em 3 regiões: imagem completa, parte principal (até 0,4s e entre 1 e 8 ppm) e parte final (região do ruído, após 0,6s, entre 1 e 8 ppm).
Métricas do histograma: moda, frequência da moda, largura do pico central, skewness e kurtosis.

Os arquivos csv são exemplos desses datasets gerados considerando a parte real do espectrograma (p = real) normalizado pelo maior valor absoluto (norm = abs). Utilizou-se hop = 10, tamanho da janela = 256 e quantidade de frequências na FT = 446 - na abordagem inicial, utilizou-se zero-padding para aumentar a resolução frequencial. Com a evolução do trabalho essa técnica foi desconsiderada, pois a literatura a julga ineficiente. No restante dos estudos realizados nesse projeto, considera-se win = tamanho da janela = quantidade de frequências na FT. 

Em razão dos dados terem sido gerados dessa maneira antiga, a função de geração do espectrograma e demais funções de análise presentes nesses notebooks estão desatualizadas com relação ao que se obtém em utils.py. 

Os resultados obtidos da presente análise não foram muito esclarecedores, e, por isso, essa abordagem foi deixada de lado. No entanto, mantém-se essa pasta caso algum leitor interessado queira se inspirar na ideia e propor abordagens similares, porém com métricas mais significativas. Porém, ao leitor interessado, atenção as funções desatualizadas aqui presentes!!!
