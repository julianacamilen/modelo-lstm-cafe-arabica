# LSTM para previsão de preço de café arábica
Aplicação do modelo Long Time-Short Memory (LSTM) para a previsão diária de preço (em reais) de café arábica no Brasil, considerando dados históricos do CEPEA no período de 01/01/2015 à 01/01/2025.

# Etapas do projeto
Processamento dos dados: Adequações gerais dos dados para treinamento e teste do modelo.

Engenharia de atributos: Aplicação técnicas de normalização das variáveis, divisão do conjunto de dados para treinamento (80%) e teste (20%) e remodelação dos dados para o formato 3D conforme exigido pelo LSTM (`amostras`, `passos_de_tempo`, `características`).

Desenvolvimento e treinamento do modelo: Construção de um modelo LSTM multicamadas, considerando `relu` para ativação, o otimizador `adam`, a função de perda `mean_squared_error` e a métrica de performance `mae`. 

Avaliação do modelo: Cálculo das métricas de performance MAE, MAPE e RMSE.

Visualização: Gráfico de linha para visualização dos preços de treinamento, validação e previstos para análise de desempenho.

# Ferramentas
1. Python (`pandas`, `numpy`, `seaborn`, `matplotlib`) para manipulação e visualização dos dados.
2. Keras/TensorFlow para construção do modelo LSTM.
3. Scikit-learn para pré-processamento, incluindo escalonamento e divisão dos dados, e para cálculo de métricas de performance.
4. Google Colab para treinamento do modelo.

# Conjunto de dados

Compreende dez anos (01/01/2015 - 01/01/2025) de dados referentes ao preço do café arábica no Brasil. Os dados não são necessariamente diários, apresentando, em geral, uma nova observação a cada dois ou três dias. O conjunto de dados possui as seguintes colunas:
1. data: data de coleta da respectiva observação.
2. preco_reais: preço em reais (R$) no Brasil para a saca de 60kg de café arábica beneficiado.
3. preco_dolares: preço em doláres (US$) no Brasil para a saca de 60kg de café arábica beneficiado.

# Fluxo do projeto

1. **Importação de Bibliotecas**:
   - Importação de bibliotecas necessárias como `pandas`, `numpy`, `seaborn`, `matplotlib`, e ferramentas de aprendizado de máquina do Keras e Scikit-learn.

2. **Carregamento e Pré-processamento dos Dados**:
   - Leitura dos dados a partir de um arquivo CSV.
   - Conversão da coluna de datas para o formato datetime.
   - Filtragem dos dados com base no período 01/01/2015 - 01/01/2025 e ordenação cronológica.

3. **Visualização dos Dados**:
   - Criação e plotagem de gráfico de linha para visualizar o histórico dos preços do café em reais.

4. **Preparação dos Dados para o Modelo LSTM**:
   - Normalização dos dados para uma escala 0-1 utilizando o `MinMaxScaler`.
   - Divisão dos dados em conjuntos de treinamento (80%) e teste (20%).
   - Criação de sequências de dados para treinamento e teste do modelo LSTM com uma janela de tempo fixa (`window_size`) de 60 dias.

5. **Construção do Modelo LSTM**:
   - Definição do modelo LSTM com camadas sequenciais:
     - Uma camada LSTM com 512 neurônios e ativação `relu`.
     - Uma camada de `Dropout` para evitar overfitting.
     - Outra camada LSTM com 256 neurônios.
     - Outra camada de `Dropout`.
     - Uma camada densa (fully connected) para gerar a previsão final.
   - Compilação do modelo utilizando o otimizador `adam`, a função de perda `mean_squared_error` e a métrica de performance `mae`.

6. **Treinamento do Modelo**:
   - Treinamento do modelo com os dados (`X_train` e `y_train`).
   - Definição dos hiperparâmetros número de épocas (`epochs=35`) e tamanho do lote (`batch_size=32`).

7. **Predição e Avaliação do Modelo**:
   - O modelo treinado é utilizado para prever os valores no conjunto de teste (`X_test`).
   - Avaliação do desempenho utilizando métricas como:
     - **MAE (Erro Médio Absoluto)**: Mede o erro médio entre valores reais e previstos.
     - **MAPE (Erro Percentual Absoluto Médio)**: Mede o erro percentual médio.
     - **RMSE (Raiz do Erro Quadrático Médio)**: Mede a raiz quadrada da média dos erros ao quadrado.

8. **Visualização das Previsões**:
   - Criação de um gráfico comparativo para ilustrar:
     - Dados reais de preços.
     - Previsões feitas pelo modelo.
     - Divisão dos dados reais entre os conjuntos de treinamento e teste.

9. **Workflow Principal**:
   - Todo o fluxo de trabalho é organizado em um bloco `if __name__ == "__main__"` para facilitar a execução direta do script.
   - A sequência de etapas descrita acima é automatizada para execução contínua.

# Resultados

A avaliação do modelo aponta os seguintes resultados para as métricas de performance:

1. **MAE (Mean Absolute Error)**:
O MAE é de R$ 26.91, ou seja, em média, as previsões do modelo estão desviando dos dados reais por R$ 26.91. Esse valor é relativamente pequeno, sugerindo que o modelo está fazendo previsões próximas aos dados reais na maioria das vezes.
2. **MAPE (Mean Absolute Percentage Error)**:
O MAPE é de 2.07% e, portanto, indica um erro de 2.07% em média entre os dados reais e os dados previstos. Da mesma forma, essa métrica sugere que o modelo produz previsões precisas, com erros percentuais pequenos.
3. **RMSE (Root Mean Squared Error)**:
O RMSE é de R$ 44.42 e, uma vez que o RMSE é mais sensível a grandes erros do que o MAE, esse resultado mais alto mostra que, embora a média dos erros seja pequena, há algumas previsões em que o modelo erra significativamente, provavelmente em casos isolados ou extremos (outliers).
