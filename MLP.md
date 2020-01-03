## Olár :)

Nesse tutorial, vamos botar em prática o que foi visto nessa breve introdução de redes perceptron.
Você poderá aprender como implementar um perceptron para o dataset seguindo este notebook, para isso utilizaremos o dataset Iris. Tentarei explicar tudo em detalhes sobre a implementação deste notbook. Aproveite.

## Pacotes como pré-requisito

Considerando que estamos usando o sistema Ubuntu, basta instalar o pip (sudo apt install python-pip), e logo depois:

    sudo pip install numpy
    sudo pip install pandas
    sudo pip install keras
    sudo pip install theano
    sudo pip install sklearn
    sudo pip install np_utils
    sudo pip install tensorflow-gpu ou sudo pip install tensorflow
    sudo pip install seaborn

## Dataset

O dataset das flores Iris ou Iris de Fisher é um dataset mutivariável introduzido pelo estatistico e biólogo britânico Ronald Fisher em um artigo de 1936. O dataset consiste de 50 exemplos de cada uma das 3 espécies de Iris (Iris setosa, Iris virginica e Iris versicolor). Quatro features foram medidas para cada amostra : o tamanho e comprimento das pétalas e sépala, em centimetros.

## Visualização

A melhor forma de inspecionarmos os dados é visualizando-os. Para fazer isso, utilizaremos a biblioteca seaborn para gerar um gráfico de dispersão — Scatter plot, em inglês. Isso nos permitirá verificar se os dados de medição das pétalas e sépalas estão bem distribuídos.

```python
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="ticks")


df = sns.load_dataset("iris")

sns.pairplot(df, hue="species")

plt.show()
```
