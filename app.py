# PROJETO - DATA APP - SEGMENTAÇÃO DE CLIENTES

# IMPORTANTO OS PACOTES
import pandas as pd
import numpy as np
import streamlit as st
import pickle
from PIL import Image
import scipy.stats
import matplotlib as mat
import matplotlib.pyplot as plt
import colorsys
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import sklearn
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist, pdist
from sklearn.metrics import homogeneity_completeness_v_measure
from sklearn.preprocessing import MinMaxScaler
plt.style.use('seaborn-talk')
import warnings
warnings.filterwarnings("ignore")

######################################################################################################################

# CARREGAMENTO DOS DADOS

# Define o título do Projeto
st.title("PROJETO DE SEGMENTAÇÃO DE CLIENTES")
st.subheader("---------------------------------------------------------------------------------------")

# Escrevendo a barra lateral
st.sidebar.title('ESTRUTURA ANALÍTICA DE PROJETO')

# Escrevendo o objetivo do projeto
eap1= st.sidebar.checkbox('1- OBJETIVO')
if eap1:
    st.header('1- OBJETIVO')
    st.write('O Projeto tem como objetivo segmentar os clientes de determinada empresa, ou seja, utilizar informações coletadas dos clientes e após análises identificar padrões para construir os grupos (clusters) com base em características comuns.')

# Escrevendo o dicionário dos atributos
eap2= st.sidebar.checkbox('2- DICIONÁRIO DE ATRIBUTOS')
dicionario = Image.open('imagem1.png')

if eap2:
    st.header('2- DICIONÁRIO DE ATRIBUTOS')
    st.image(dicionario, caption=None, width= 700, use_column_width=False, output_format='PNG')

# Carregando o dataframe
df = pd.read_csv('dataset.csv', sep = ',', encoding = 'utf-8')

#######################################################################################################################

# ANÁLISE EXPLORATÓRIA, LIMPEZA E TRANSFORMAÇÃO
eap3= st.sidebar.checkbox('3- ANÁLISE EXPLORATÓRIA')
st.sidebar.write('-----3.1- Visualização do Cabeçalho do Dataframe')
st.sidebar.write('-----3.2- Resumo dos Dados')
st.sidebar.write('-----3.3- Sumário Estatístico das Variáveis Numéricas')
st.sidebar.write('-----3.4- Sumário Estatístico das Variáveis Categóricas')

# Ajsutando a coluna ID
df.reset_index(inplace=True, drop=False)
df.drop(columns=['ID'], inplace=True)

# Renomeando as colunas
colunas = ['Id', 'Sexo', 'Estado_Civil', 'Idade', 'Educacao', 'Renda', 'Ocupacao', 'Tamanho_Cidade']
df.columns = colunas

# Convertendo as variáveis Id, Sexo, Estado_Civil, Educacao, Ocupacacao e Tamanho_Cidade para categorias
df['Id'] = pd.Categorical(df['Id'])
df['Sexo'] = pd.Categorical(df['Sexo'])
df['Estado_Civil'] = pd.Categorical(df['Estado_Civil'])
df['Educacao'] = pd.Categorical(df['Educacao'])
df['Ocupacao'] = pd.Categorical(df['Ocupacao'])
df['Tamanho_Cidade'] = pd.Categorical(df['Tamanho_Cidade'])

# Criando uma cópia do dataframne
df1 = df.copy()

# Colunas numéricas (quantitativas) e categóricas 
colunas_numericas = ['Idade', 'Renda']
colunas_categoricas = ['Id', 'Sexo', 'Estado_Civil', 'Educacao', 'Ocupacao', 'Tamanho_Cidade']

# Dataframes com os tipos diferentes de variáveis
df_num = df1[colunas_numericas]
df_cat = df1[colunas_categoricas]

# Sumário estatístico das variáveis numéricas
sumario_num = pd.DataFrame(index = df_num.columns)
sumario_num['Não Nulos'] = df_num.count().values
sumario_num['Valores Únicos'] = df_num.nunique().values
sumario_num['Média'] = df_num.mean()
sumario_num['DesvPad'] = df_num.std()
sumario_num['Min'] = df_num.min()
sumario_num['Max'] = df_num.max()

sumario_num= sumario_num.astype(int)

# Sumário estatístico das variáveis categóricas
sumario_cat = pd.DataFrame(index = df_cat.columns)
sumario_cat['Não Nulos'] = df_cat.count().values
sumario_cat['% Populado'] = round(sumario_cat['Não Nulos'] / df_cat.shape[0]*100,2)
sumario_cat['Valores Únicos'] = df_cat.nunique().values

# Adiciona mais uma coluna com valores mais comuns
temp = []
for coluna in colunas_categoricas:
    temp.append(df_cat[coluna].value_counts().idxmax())
sumario_cat['Valores Mais Comuns'] = temp

sumario_cat= sumario_cat.astype(int)

if eap3:
    st.header('3- ANÁLISE EXPLORATÓRIA')
    st.subheader('3.1- Visualização do Cabeçalho do Dataframe')
    st.write(df.head(5))
    st.subheader('3.2- Resumo dos Dados')
    st.write("Linhas: ", df1.shape[0])
    st.write("Colunas: ", df1.shape[1])
    coluna1, coluna2 = st.columns(2)
    with coluna1:
       st.write("\nValores Ausentes: \n" , df1.isnull().sum())
    with coluna2:
       st.write("\nValores Únicos: \n", df1.nunique())
    st.subheader('3.3- Sumário Estatístico das Variáveis Numéricas')
    st.write(sumario_num)
    st.subheader('3.4- Sumário Estatístico das Variáveis Categóricas')
    st.write(sumario_cat)

######################################################################################################################

# VISUALIZAÇÃO DAS VARIÁVEIS
eap4= st.sidebar.checkbox('4- VISUALIZAÇÃO DAS VARIÁVEIS')
st.sidebar.write('-----4.1- Sexo')
st.sidebar.write('-----4.2- Estado Civil')
st.sidebar.write('-----4.3- Idade')
st.sidebar.write('-----4.4- Educação')
st.sidebar.write('-----4.5- Renda')
st.sidebar.write('-----4.6- Ocupação')
st.sidebar.write('-----4.7- Tamanho da Cidade')

# Variável 2 - Sexo (0= Masculino  1= Feminino)
sexo = df1['Sexo'].value_counts().rename_axis('Sexo').reset_index(name = 'Total')

# Criando a coluna com o nome das categorias
for item in range(0,len(sexo['Sexo']),1):
   
    if sexo['Sexo'][item] == 0:
        sexo['sexo_cat']= 'Masculino'
    elif sexo['Sexo'][item] == 1:
        sexo['sexo_cat']= 'Feminino'
    
for item in range(0,len(sexo['Sexo']),1):
   
    if sexo['Sexo'][item] == 0:
        sexo['sexo_cat'][item]= 'Masculino'
    elif sexo['Sexo'][item] == 1:
        sexo['sexo_cat'][item]= 'Feminino'
    

fig = px.bar(sexo, x='sexo_cat', y='Total', title="Total de Clientes por Sexo", color='sexo_cat',
			 labels={'Total':'Contagem', 'sexo_cat': 'Sexo'}, height=480, width= 700, text= 'Total', template = "seaborn")
fig.update_traces(textposition='outside')
fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', showlegend= False)

#--------------------------------------------------------------------------------------------------------------------#

# Variável 3 - Estado_Civil (0= Solteiro  1= Não Solteiro [Casado/Divorciado/Separado/Viúvo])
estado_civil = df1['Estado_Civil'].value_counts().rename_axis('Estado_Civil').reset_index(name = 'Total')

# Criando a coluna com o nome das categorias
for item in range(0,len(estado_civil['Estado_Civil']),1):
   
    if estado_civil['Estado_Civil'][item] == 0:
        estado_civil['estado_cat']= 'Solteiro'
    elif estado_civil['Estado_Civil'][item] == 1:
        estado_civil['estado_cat']= 'Não Solteiro'
    
for item in range(0,len(sexo['Sexo']),1):
   
    if estado_civil['Estado_Civil'][item] == 0:
        estado_civil['estado_cat'][item]= 'Solteiro'
    elif estado_civil['Estado_Civil'][item] == 1:
        estado_civil['estado_cat'][item]= 'Não Solteiro'

fig2 = px.bar(estado_civil, x='estado_cat', y='Total', title="Total de Clientes por Estado Civil", color='estado_cat',
			 labels={'Total':'Contagem', 'estado_cat': 'Estado Civil'}, height=480, width= 700, text= 'Total', template = "seaborn")
fig2.update_traces(textposition='outside')
fig2.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', showlegend= False)

#--------------------------------------------------------------------------------------------------------------------#

# Variável 4 - Idade
resumo_idade= df1['Idade'].describe().round(2)
resumo_idade= resumo_idade.astype(int)

fig3 = px.box(df1, x="Idade", height=400, width= 700, title="Boxplot Idade", template = "seaborn")

x = df1['Idade'].values
dados = [x]
labels = ['Idade']

fig5 = ff.create_distplot(dados, labels, bin_size=1, show_rug=False)
fig5.data[1].line.color = 'indianred'
fig5.update_layout(width=700, 
                  height=480,
                  bargap=0.01, template = "seaborn", title_text='Distplot Idade', showlegend=False
                  )

# Calculando a Assimetria e Curtose
assimetria_idade = scipy.stats.skew(df1['Idade'])
assimetria_idade = round(assimetria_idade, 4)
curtose_idade = scipy.stats.kurtosis(df1['Idade'])
curtose_idade = round(curtose_idade, 4)

#--------------------------------------------------------------------------------------------------------------------#

# Variável 5 - Educação (0= Outro/Desconhecido  1= Ensino Médio  2= Universitário  3= Graduado)
educacao = df1['Educacao'].value_counts().rename_axis('Educacao').reset_index(name = 'Total')

# Criando a coluna com o nome das categorias
for item in range(0,len(educacao['Educacao']),1):
   
    if educacao['Educacao'][item] == 0:
        educacao['educacao_cat']= 'Outro'
    elif educacao['Educacao'][item] == 1:
        educacao['educacao_cat']= 'Ensino Médio'
    elif educacao['Educacao'][item] == 2:
    	educacao['educacao_cat']= 'Universitário'
    elif educacao['Educacao'][item] == 3:
    	educacao['educacao_cat']= 'Graduado'

for item in range(0,len(educacao['Educacao']),1):
   
    if educacao['Educacao'][item] == 0:
        educacao['educacao_cat'][item]= 'Outro'
    elif educacao['Educacao'][item] == 1:
        educacao['educacao_cat'][item]= 'Ensino Médio'
    elif educacao['Educacao'][item] == 2:
        educacao['educacao_cat'][item]= 'Universitário'
    elif educacao['Educacao'][item] == 3:
        educacao['educacao_cat'][item]= 'Graduado'   

fig6 = px.bar(educacao, x='educacao_cat', y='Total', title="Total de Clientes por Educação", color='educacao_cat',
             labels={'Total':'Contagem', 'educacao_cat': 'Educação'}, height=480, width= 700, text= 'Total', template = "seaborn")
fig6.update_traces(textposition='outside')
fig6.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', showlegend= False)

def percentile(n):

    def percentile_(w):
        return np.percentile(w, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_

resumo_educacao_renda = df1.groupby('Educacao')['Renda'].agg([('Contagem', 'count'), ('Média', 'mean'), ('DesvPad', 'std'), 
('Mínimo', 'min'), ('P25', percentile(25)), ('P50', percentile(50)), ('P75', percentile(75)),
('Máximo', 'max')]) 
resumo_educacao_renda= resumo_educacao_renda.astype(int)
resumo_educacao_renda['Educação'] = ['Outro', 'Ensino Médio', 'Universitário', 'Graduado']
cols = resumo_educacao_renda.columns.tolist()
cols = cols[-1:] + cols[:-1]
resumo_educacao_renda = resumo_educacao_renda[cols]

fig7 = px.bar(resumo_educacao_renda, x='Educação', y='Média', title="Média de Renda por Educação", color='Educação',
             labels={'Média':'Média', 'Educação': 'Educação'}, height=480, width= 700, text= 'Média', template = "seaborn")
fig7.update_traces(textposition='outside')
fig7.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', showlegend= False)
#st.plotly_chart(fig7)

#--------------------------------------------------------------------------------------------------------------------#

# Variável 6 - Renda
resumo_renda= df1['Renda'].describe().round(2)
resumo_renda= resumo_renda.astype(int)
#st.write(resumo_renda)

fig8 = px.box(df1, x="Renda", height=400, width= 700, title="Boxplot Renda", template = "seaborn")

fig9 = px.histogram(df1, x="Renda", nbins=30, title="Histograma Renda", 
                   labels={'count':'Contagem', 'Renda': 'Renda'}, height=480, width= 700, template = "seaborn")
fig9.update_layout(bargap=0.01)

# Calculando a Assimetria e Curtose
assimetria_renda = scipy.stats.skew(df1['Renda'])
assimetria_renda = round(assimetria_renda, 4)
curtose_renda = scipy.stats.kurtosis(df1['Renda'])
curtose_renda = round(curtose_renda, 4)

#--------------------------------------------------------------------------------------------------------------------#

# Variável 7 - Ocupação (0= Desempregado  1= Funcionário  2= Gestão/Autônomo)

ocupacao = df1['Ocupacao'].value_counts().rename_axis('Ocupacao').reset_index(name = 'Total')
#st.write(ocupacao)

# Criando a coluna com o nome das categorias
for item in range(0,len(ocupacao['Ocupacao']),1):
   
    if ocupacao['Ocupacao'][item] == 0:
        ocupacao['ocupacao_cat']= 'Desempregado'
    elif ocupacao['Ocupacao'][item] == 1:
        ocupacao['ocupacao_cat']= 'Funcionário'
    elif ocupacao['Ocupacao'][item] == 2:
        ocupacao['ocupacao_cat']= 'Gestão/Autônomo'

for item in range(0,len(ocupacao['Ocupacao']),1):
   
    if ocupacao['Ocupacao'][item] == 0:
        ocupacao['ocupacao_cat'][item]= 'Desempregado'
    elif ocupacao['Ocupacao'][item] == 1:
        ocupacao['ocupacao_cat'][item]= 'Funcionário'
    elif ocupacao['Ocupacao'][item] == 2:
        ocupacao['ocupacao_cat'][item]= 'Gestão/Autônomo'
       
fig11 = px.bar(ocupacao, x='ocupacao_cat', y='Total', title="Total de Clientes por Ocupação", color='ocupacao_cat',
             labels={'Total':'Contagem', 'ocupacao_cat': 'Ocupacao'}, height=480, width= 700, text= 'Total', template = "seaborn")
fig11.update_traces(textposition='outside')
fig11.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', showlegend= False)

resumo_ocupacao_renda = df1.groupby('Ocupacao')['Renda'].agg([('Contagem', 'count'), ('Média', 'mean'), ('DesvPad', 'std'), 
('Mínimo', 'min'), ('P25', percentile(25)), ('P50', percentile(50)), ('P75', percentile(75)),
('Máximo', 'max')]) 
resumo_ocupacao_renda= resumo_ocupacao_renda.astype(int)
resumo_ocupacao_renda['Ocupação'] = ['Desempregado', 'Funcionário', 'Gestão/Autônomo']
cols2 = resumo_ocupacao_renda.columns.tolist()
cols2 = cols2[-1:] + cols2[:-1]
resumo_ocupacao_renda = resumo_ocupacao_renda[cols2]

fig12 = px.bar(resumo_ocupacao_renda, x='Ocupação', y='Média', title="Média de Renda por Ocupação", color='Ocupação',
             labels={'Média':'Média', 'Ocupação': 'Ocupação'}, height=480, width= 700, text= 'Média', template = "seaborn")
fig12.update_traces(textposition='outside')
fig12.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', showlegend= False)

#--------------------------------------------------------------------------------------------------------------------#

# Variável 8 - Tamanho da Cidade (0= Pequena  1= Média  2= Grande)

tamanho_cidade = df1['Tamanho_Cidade'].value_counts().rename_axis('Tamanho_Cidade').reset_index(name = 'Total')

# Criando a coluna com o nome das categorias
for item in range(0,len(tamanho_cidade['Tamanho_Cidade']),1):
   
    if tamanho_cidade['Tamanho_Cidade'][item] == 0:
        tamanho_cidade['cidade_cat']= 'Pequena'
    elif tamanho_cidade['Tamanho_Cidade'][item] == 1:
        tamanho_cidade['cidade_cat']= 'Média'
    elif tamanho_cidade['Tamanho_Cidade'][item] == 2:
        tamanho_cidade['cidade_cat']= 'Grande'

for item in range(0,len(tamanho_cidade['Tamanho_Cidade']),1):
   
    if tamanho_cidade['Tamanho_Cidade'][item] == 0:
        tamanho_cidade['cidade_cat'][item]= 'Pequena'
    elif tamanho_cidade['Tamanho_Cidade'][item] == 1:
        tamanho_cidade['cidade_cat'][item]= 'Média'
    elif tamanho_cidade['Tamanho_Cidade'][item] == 2:
        tamanho_cidade['cidade_cat'][item]= 'Grande'
       
fig11 = px.bar(tamanho_cidade, x='cidade_cat', y='Total', title="Total de Clientes por Tamanho da Cidade", color='cidade_cat',
             labels={'Total':'Contagem', 'cidade_cat': 'Tamanho da Cidade'}, height=480, width= 700, text= 'Total', template = "seaborn")
fig11.update_traces(textposition='outside')
fig11.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', showlegend= False)

resumo_cidade_renda = df1.groupby('Tamanho_Cidade')['Renda'].agg([('Contagem', 'count'), ('Média', 'mean'), ('DesvPad', 'std'), 
('Mínimo', 'min'), ('P25', percentile(25)), ('P50', percentile(50)), ('P75', percentile(75)),
('Máximo', 'max')]) 
resumo_cidade_renda= resumo_cidade_renda.astype(int)
resumo_cidade_renda['TamanhoDaCidade'] = ['Desempregado', 'Funcionário', 'Gestão/Autônomo']
cols3 = resumo_cidade_renda.columns.tolist()
cols3 = cols3[-1:] + cols3[:-1]
resumo_cidade_renda = resumo_cidade_renda[cols3]

fig12 = px.bar(resumo_cidade_renda, x='TamanhoDaCidade', y='Média', title="Média de Renda por Tamanho da Cidade", color='TamanhoDaCidade',
             labels={'Média':'Média', 'TamanhoDaCidade': 'Tamanho da Cidade'}, height=480, width= 700, text= 'Média', template = "seaborn")
fig12.update_traces(textposition='outside')
fig12.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', showlegend= False)

if eap4:
    st.header('4- VISUALIZAÇÃO DAS VARIÁVEIS')
    st.subheader('4.1- Sexo')
    coluna3, coluna4 = st.columns([1,4])
    with coluna3:
        st.write('0- Masculino')
    with coluna4:
        st.write('1- Feminino')
    st.write(sexo)
    st.plotly_chart(fig)
    st.subheader('4.2- Estado Civil')
    coluna5, coluna6 = st.columns([1,4])
    with coluna5:
        st.write('0- Solteiro')
    with coluna6:
        st.write('1- Não Solteiro (Casado/Divorciado/Separado/Viúvo)')
    st.write(estado_civil)
    st.plotly_chart(fig2)
    st.subheader('4.3- Idade')
    st.write(resumo_idade)
    st.plotly_chart(fig3)
    st.plotly_chart(fig5)
    coluna7, coluna8 = st.columns([2,2])
    with coluna7:
        st.write("Assimetria: \n", assimetria_idade)
    with coluna8:
        st.write("Curtose: \n", curtose_idade)
    st.subheader('4.4- Educação')
    coluna9, coluna10, coluna11, coluna12 = st.columns([1,1,1,1])
    with coluna9:
        st.write('0- Outro')
    with coluna10:
        st.write('1- Ensino Médio')
    with coluna11:
        st.write('2- Universitário')
    with coluna12:
        st.write('3- Graduado')
    st.write(educacao)
    st.plotly_chart(fig6)
    st.write('Resumo Educação x Renda')
    st.write(resumo_educacao_renda)
    st.plotly_chart(fig7)
    st.subheader('4.5- Renda')
    st.write(resumo_renda)
    st.plotly_chart(fig8)
    st.plotly_chart(fig9)
    coluna13, coluna14 = st.columns([2,2])
    with coluna13:
        st.write("Assimetria: \n", (assimetria_renda))
    with coluna14:
        st.write("Curtose: \n", curtose_renda)
    st.subheader('4.5- Ocupação')
    coluna15, coluna16, coluna17 = st.columns([1,1,2])
    with coluna15:
        st.write('0- Desempregado')
    with coluna16:
        st.write('1- Funcionário')
    with coluna17:
        st.write('2- Gestão/Autônomo')
    st.write(ocupacao)
    st.plotly_chart(fig11)
    st.write('Resumo Ocupação x Renda')
    st.write(resumo_ocupacao_renda)
    st.plotly_chart(fig12)
    st.subheader('4.6- Tamanho da Cidade')
    coluna18, coluna19, coluna20 = st.columns([1,1,2])
    with coluna18:
        st.write('0- Pequena')
    with coluna19:
        st.write('1- Média')
    with coluna20:
        st.write('2- Grande')
    st.write(tamanho_cidade)
    st.plotly_chart(fig11)
    st.write('Resumo Tamanho da Cidade x Renda')
    st.write(resumo_cidade_renda)
    st.plotly_chart(fig12)

######################################################################################################################

# MODELAGEM PREDITIVA

# Aplicando redução de dimensionalidade
pca = PCA(n_components = 2).fit_transform(df1)

# Determinando um range de K
k_range = range(1,10)

# Aplicando o modelo K-Means para cada valor de K (esta célula pode levar bastante tempo para ser executada)
k_means_var = [KMeans(n_clusters = k).fit(pca) for k in k_range]

# Ajustando o centróide do cluster para cada modelo
centroids = [X.cluster_centers_ for X in k_means_var]

# Calculando a distância euclidiana de cada ponto de dado para o centróide
k_euclid = [cdist(pca, cent, 'euclidean') for cent in centroids]
dist = [np.min(ke, axis = 1) for ke in k_euclid]

# Soma dos quadrados das distâncias dentro do cluster
soma_quadrados_intra_cluster = [sum(d**2) for d in dist]

# Soma total dos quadrados
soma_total = sum(pdist(pca)**2)/pca.shape[0]

# Soma dos quadrados entre clusters
soma_quadrados_inter_cluster = soma_total - soma_quadrados_intra_cluster

# Construindo o dataframe para a Curva de Elbow
k_range1 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
celbow = pd.DataFrame(soma_quadrados_intra_cluster)
celbow['soma_total'] = soma_total
celbow['range'] = k_range1
colunas2 = ['inter_cluster', 'soma_total', 'range']
celbow.columns = colunas2
celbow['variancia'] = (celbow['inter_cluster'] / celbow['soma_total']) * 100

# Curva de Elbow
fig13 = px.line(celbow, x='range', y=(soma_quadrados_inter_cluster/soma_total*100), title="Variância Explicada x Número de Cluster",
             labels={'y':' Percentual Variância Explicada', 'range': 'Número de Clusters'}, height=480, width= 700, template = "seaborn")
fig13.data[0].update(mode='markers+lines')
fig13.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', showlegend= False)
#st.plotly_chart(fig13)

# Criação do modelo
modelo_v1 = KMeans(n_clusters = 4, 
                     init = 'k-means++', 
                     n_init = 10, 
                     max_iter = 300, 
                     tol = 0.0001,   
                     algorithm = 'elkan')


# Treinamento do modelo
modelo_v1.fit(pca)

# Extração dos labels
labels1 = modelo_v1.labels_
#labels1
valores_k = range(2, 10)
# Lista para receber as métricas
HCVs = []

# Um resultado de cluster satisfaz a homogeneidade se todos os seus clusters contiverem apenas pontos de 
# dados que são membros de uma única classe.

# Um resultado de cluster satisfaz a completude se todos os pontos de dados que são membros 
# de uma determinada classe são elementos do mesmo cluster.

# Ambas as pontuações têm valores positivos entre 0,0 e 1,0, sendo desejáveis valores maiores.

# Medida V é a média entre homogeneidade e completude.

previsoes = modelo_v1.fit_predict(pca)

eap5= st.sidebar.checkbox('5- MODELAGEM PREDITIVA')
st.sidebar.write('-----5.1- Curva de Elbow')

if eap5:
    st.header('5- MODELAGEM PREDITIVA')
    st.subheader('Algoritmo K-means')
    st.subheader('5.1- Curva de Elbow')
    st.plotly_chart(fig13)

######################################################################################################################

# RESULTADOS

# Converte o array para dataframe
df_labels1 = pd.DataFrame(labels1)

# Vamos fazer o merge de pca e os labels (clusters) encontrados pelo modelo
df_final1 = df1.merge(df_labels1, left_index = True, right_index = True)
df_final1.rename(columns = {0:"Cluster"}, inplace = True)

cluster1 = df_final1['Cluster'].value_counts().rename_axis('Cluster').reset_index(name = 'Total')

df_final2 = df_final1.copy()
cols3 = df_final2.columns.tolist()
cols3 = cols3[-1:] + cols3[:-1]
df_final2 = df_final2[cols3]

# Calcula a média de idade por cluster
resumo_cluster = df_final1.groupby('Cluster')['Idade'].agg([('Clientes', 'count'),('Media_Idade', 'mean')]) 
resumo_cluster= resumo_cluster.astype(int)
resumo_cluster['Cluster']= [0, 1, 2, 3]
cols4 = resumo_cluster.columns.tolist()
cols4 = cols4[-1:] + cols4[:-1]
resumo_cluster = resumo_cluster[cols4]

# Calcula a média de renda por cluster
resumo_renda = df_final1.groupby('Cluster')['Renda'].agg([('Clientes', 'count'),('Media_Renda', 'mean')])
resumo_renda= resumo_renda.astype(int)
resumo_renda.drop(columns=["Clientes"],inplace=True)
resumo_cluster = resumo_cluster.merge(resumo_renda, left_index = True, right_index = True)

# Calcula a quantidade de clientes de cada genero por cluster
sexo_cluster = df_final2.groupby('Cluster')['Sexo'].value_counts(sort=False)
contagem_sexo= sexo_cluster.values
colunas_sexo = ['Cluster', 'Masculino', 'Feminino']
cluster_sexo = [0, 1, 2, 3]

df_sexo = pd.DataFrame(columns= colunas_sexo)
df_sexo['Cluster'] = cluster_sexo

df_sexo.iloc[[0],1] = contagem_sexo[0]
df_sexo.iloc[[0],2] = contagem_sexo[1]
df_sexo.iloc[[1],1] = contagem_sexo[2]
df_sexo.iloc[[1],2] = contagem_sexo[3]
df_sexo.iloc[[2],1] = contagem_sexo[4]
df_sexo.iloc[[2],2] = contagem_sexo[5]
df_sexo.iloc[[3],1] = contagem_sexo[6]
df_sexo.iloc[[3],2] = contagem_sexo[7]
df_sexo.drop(columns=["Cluster"],inplace=True)
resumo_cluster = resumo_cluster.merge(df_sexo, left_index = True, right_index = True)

# Calcula a quantidade de clientes de cada estado civil por cluster
civil_cluster = df_final2.groupby('Cluster')['Estado_Civil'].value_counts(sort=False)
contagem_civil= civil_cluster.values
colunas_civil = ['Cluster', 'Solteiro', 'Nao_Solteiro']
cluster_civil = [0, 1, 2, 3]

df_civil = pd.DataFrame(columns= colunas_civil)
df_civil['Cluster'] = cluster_civil

df_civil.iloc[[0],1] = contagem_civil[0]
df_civil.iloc[[0],2] = contagem_civil[1]
df_civil.iloc[[1],1] = contagem_civil[2]
df_civil.iloc[[1],2] = contagem_civil[3]
df_civil.iloc[[2],1] = contagem_civil[4]
df_civil.iloc[[2],2] = contagem_civil[5]
df_civil.iloc[[3],1] = contagem_civil[6]
df_civil.iloc[[3],2] = contagem_civil[7]
df_civil.drop(columns=["Cluster"],inplace=True)
resumo_cluster = resumo_cluster.merge(df_civil, left_index = True, right_index = True)

# Calcula a quantidade de clientes de cada nível de formação por cluster
formacao_cluster = df_final2.groupby('Cluster')['Educacao'].value_counts(sort=False)
contagem_formacao= formacao_cluster.values
colunas_formacao = ['Cluster', 'Outro', 'Ensino_Medio', 'Universitario', 'Graduado']
cluster_formacao = [0, 1, 2, 3]

df_formacao = pd.DataFrame(columns= colunas_formacao)
df_formacao['Cluster'] = cluster_formacao

df_formacao.iloc[[0],1] = contagem_formacao[0]
df_formacao.iloc[[0],2] = contagem_formacao[1]
df_formacao.iloc[[0],3] = contagem_formacao[2]
df_formacao.iloc[[0],4] = contagem_formacao[3]
df_formacao.iloc[[1],1] = contagem_formacao[4]
df_formacao.iloc[[1],2] = contagem_formacao[5]
df_formacao.iloc[[1],3] = contagem_formacao[6]
df_formacao.iloc[[1],4] = contagem_formacao[7]
df_formacao.iloc[[2],1] = contagem_formacao[8]
df_formacao.iloc[[2],2] = contagem_formacao[9]
df_formacao.iloc[[2],3] = contagem_formacao[10]
df_formacao.iloc[[2],4] = contagem_formacao[11]
df_formacao.iloc[[3],1] = contagem_formacao[12]
df_formacao.iloc[[3],2] = contagem_formacao[13]
df_formacao.iloc[[3],3] = contagem_formacao[14]
df_formacao.iloc[[3],4] = contagem_formacao[15]
df_formacao.drop(columns=["Cluster"],inplace=True)
resumo_cluster = resumo_cluster.merge(df_formacao, left_index = True, right_index = True)

# Calcula a quantidade de clientes de cada ocupação por cluster
ocupacao_cluster = df_final2.groupby('Cluster')['Ocupacao'].value_counts(sort=False)
contagem_ocupacao= ocupacao_cluster.values
colunas_ocupacao = ['Cluster', 'Desempregado', 'Funcionario', 'Gestao/Autonomo']
cluster_ocupacao = [0, 1, 2, 3]

df_ocupacao = pd.DataFrame(columns= colunas_ocupacao)
df_ocupacao['Cluster'] = cluster_ocupacao

df_ocupacao.iloc[[0],1] = contagem_ocupacao[0]
df_ocupacao.iloc[[0],2] = contagem_ocupacao[1]
df_ocupacao.iloc[[0],3] = contagem_ocupacao[2]
df_ocupacao.iloc[[1],1] = contagem_ocupacao[3]
df_ocupacao.iloc[[1],2] = contagem_ocupacao[4]
df_ocupacao.iloc[[1],3] = contagem_ocupacao[5]
df_ocupacao.iloc[[2],1] = contagem_ocupacao[6]
df_ocupacao.iloc[[2],2] = contagem_ocupacao[7]
df_ocupacao.iloc[[2],3] = contagem_ocupacao[8]
df_ocupacao.iloc[[3],1] = contagem_ocupacao[9]
df_ocupacao.iloc[[3],2] = contagem_ocupacao[10]
df_ocupacao.iloc[[3],3] = 0
df_ocupacao.drop(columns=["Cluster"],inplace=True)
resumo_cluster = resumo_cluster.merge(df_ocupacao, left_index = True, right_index = True)

# Calcula a quantidade de clientes de cada tamanho de cidade por cluster
cidade_cluster = df_final2.groupby('Cluster')['Tamanho_Cidade'].value_counts(sort=False)
contagem_cidade= cidade_cluster.values
colunas_cidade = ['Cluster', 'Pequena', 'Media', 'Grande']
cluster_cidade = [0, 1, 2, 3]

df_cidade = pd.DataFrame(columns= colunas_cidade)
df_cidade['Cluster'] = cluster_cidade

df_cidade.iloc[[0],1] = contagem_cidade[0]
df_cidade.iloc[[0],2] = contagem_cidade[1]
df_cidade.iloc[[0],3] = contagem_cidade[2]
df_cidade.iloc[[1],1] = contagem_cidade[3]
df_cidade.iloc[[1],2] = contagem_cidade[4]
df_cidade.iloc[[1],3] = contagem_cidade[5]
df_cidade.iloc[[2],1] = contagem_cidade[6]
df_cidade.iloc[[2],2] = contagem_cidade[7]
df_cidade.iloc[[2],3] = contagem_cidade[8]
df_cidade.iloc[[3],1] = contagem_cidade[9]
df_cidade.iloc[[3],2] = contagem_cidade[10]
df_cidade.iloc[[3],3] = contagem_cidade[11]
df_cidade.drop(columns=["Cluster"],inplace=True)
resumo_cluster = resumo_cluster.merge(df_cidade, left_index = True, right_index = True)

# Gráficos de Comparação entre os clusters

# 1- Cluster x Média de Idade
fig14 = px.bar(resumo_cluster, x='Cluster', y='Media_Idade', title="Média de Idade por Cluster", color=['blue', 'orange', 'green', 'red'],
             labels={'Media_Idade':'Média', 'Cluster': 'Cluster'}, height=480, width= 700, text= 'Media_Idade', template = "seaborn")
fig14.update_traces(textposition='outside')
fig14.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', showlegend= False)
#st.plotly_chart(fig14)

# 2- Cluster x Média de Renda
fig15 = px.bar(resumo_cluster, x='Cluster', y='Media_Renda', title="Média de Renda por Cluster", color=['blue', 'orange', 'green', 'red'],
             labels={'Media_Renda':'Média', 'Cluster': 'Cluster'}, height=480, width= 700, text= 'Media_Renda', template = "seaborn")
fig15.update_traces(textposition='outside')
fig15.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', showlegend= False)
#st.plotly_chart(fig15)

# 3- Cluster x Sexo
fig16 = px.bar(resumo_cluster, x='Cluster', y='Masculino', title="Masculino por Cluster", color=['blue', 'orange', 'green', 'red'],
             labels={'Masculino':'Contagem', 'Cluster': 'Cluster'}, height=480, width= 700, text= 'Masculino', template = "seaborn")
fig16.update_traces(textposition='outside')
fig16.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', showlegend= False)
#st.plotly_chart(fig16)

fig17 = px.bar(resumo_cluster, x='Cluster', y='Feminino', title="Feminino por Cluster", color=['blue', 'orange', 'green', 'red'],
             labels={'Feminino':'Contagem', 'Cluster': 'Cluster'}, height=480, width= 700, text= 'Feminino', template = "seaborn")
fig17.update_traces(textposition='outside')
fig17.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', showlegend= False)
#st.plotly_chart(fig17)

fig18 = px.bar(resumo_cluster, x='Cluster', y='Solteiro', title="Solteiro por Cluster", color=['blue', 'orange', 'green', 'red'],
             labels={'Solteiro':'Contagem', 'Cluster': 'Cluster'}, height=480, width= 700, text= 'Solteiro', template = "seaborn")
fig18.update_traces(textposition='outside')
fig18.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', showlegend= False)
#st.plotly_chart(fig18)

fig19 = px.bar(resumo_cluster, x='Cluster', y='Nao_Solteiro', title="Não Solteiro por Cluster", color=['blue', 'orange', 'green', 'red'],
             labels={'Nao_Solteiro':'Contagem', 'Cluster': 'Cluster'}, height=480, width= 700, text= 'Nao_Solteiro', template = "seaborn")
fig19.update_traces(textposition='outside')
fig19.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', showlegend= False)
#st.plotly_chart(fig19)

fig20 = px.bar(resumo_cluster, x='Cluster', y='Outro', title="Educacação (Outro) por Cluster", color=['blue', 'orange', 'green', 'red'],
             labels={'Outro':'Contagem', 'Cluster': 'Cluster'}, height=480, width= 700, text= 'Outro', template = "seaborn")
fig20.update_traces(textposition='outside')
fig20.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', showlegend= False)
#st.plotly_chart(fig20)

fig21 = px.bar(resumo_cluster, x='Cluster', y='Ensino_Medio', title="Educacação (Ensino Médio) por Cluster", color=['blue', 'orange', 'green', 'red'],
             labels={'Ensino_Medio':'Contagem', 'Cluster': 'Cluster'}, height=480, width= 700, text= 'Ensino_Medio', template = "seaborn")
fig21.update_traces(textposition='outside')
fig21.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', showlegend= False)
#st.plotly_chart(fig21)

fig22 = px.bar(resumo_cluster, x='Cluster', y='Universitario', title="Educacação (Universitário) por Cluster", color=['blue', 'orange', 'green', 'red'],
             labels={'Universitario':'Contagem', 'Cluster': 'Cluster'}, height=480, width= 700, text= 'Universitario', template = "seaborn")
fig22.update_traces(textposition='outside')
fig22.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', showlegend= False)
#st.plotly_chart(fig22)

fig23 = px.bar(resumo_cluster, x='Cluster', y='Graduado', title="Educacação (Graduado) por Cluster", color=['blue', 'orange', 'green', 'red'],
             labels={'Graduado':'Contagem', 'Cluster': 'Cluster'}, height=480, width= 700, text= 'Graduado', template = "seaborn")
fig23.update_traces(textposition='outside')
fig23.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', showlegend= False)
#st.plotly_chart(fig23)

fig24 = px.bar(resumo_cluster, x='Cluster', y='Desempregado', title="Desempregado por Cluster", color=['blue', 'orange', 'green', 'red'],
             labels={'Desempregado':'Contagem', 'Cluster': 'Cluster'}, height=480, width= 700, text= 'Desempregado', template = "seaborn")
fig24.update_traces(textposition='outside')
fig24.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', showlegend= False)
#st.plotly_chart(fig24)

fig25 = px.bar(resumo_cluster, x='Cluster', y='Funcionario', title="Funcionário por Cluster", color=['blue', 'orange', 'green', 'red'],
             labels={'Funcionario':'Contagem', 'Cluster': 'Cluster'}, height=480, width= 700, text= 'Funcionario', template = "seaborn")
fig25.update_traces(textposition='outside')
fig25.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', showlegend= False)
#st.plotly_chart(fig25)

fig26 = px.bar(resumo_cluster, x='Cluster', y='Gestao/Autonomo', title="Gestão/Autônomo por Cluster", color=['blue', 'orange', 'green', 'red'],
             labels={'Gestao/Autonomo':'Contagem', 'Cluster': 'Cluster'}, height=480, width= 700, text= 'Gestao/Autonomo', template = "seaborn")
fig26.update_traces(textposition='outside')
fig26.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', showlegend= False)
#st.plotly_chart(fig26)

fig27 = px.bar(resumo_cluster, x='Cluster', y='Pequena', title="Cidade Pequena por Cluster", color=['blue', 'orange', 'green', 'red'],
             labels={'Pequena':'Contagem', 'Cluster': 'Cluster'}, height=480, width= 700, text= 'Pequena', template = "seaborn")
fig27.update_traces(textposition='outside')
fig27.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', showlegend= False)
#st.plotly_chart(fig27)

fig28 = px.bar(resumo_cluster, x='Cluster', y='Media', title="Cidade Média por Cluster", color=['blue', 'orange', 'green', 'red'],
             labels={'Media':'Contagem', 'Cluster': 'Cluster'}, height=480, width= 700, text= 'Media', template = "seaborn")
fig28.update_traces(textposition='outside')
fig28.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', showlegend= False)
#st.plotly_chart(fig28)

fig29 = px.bar(resumo_cluster, x='Cluster', y='Grande', title="Cidade Grande por Cluster", color=['blue', 'orange', 'green', 'red'],
             labels={'Grande':'Contagem', 'Cluster': 'Cluster'}, height=480, width= 700, text= 'Grande', template = "seaborn")
fig29.update_traces(textposition='outside')
fig29.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', showlegend= False)
#st.plotly_chart(fig29)


# Criando as consultas de cada cluster
resumo_c0 = df_final2[df_final2.Cluster.eq(0)]
resumo_c1 = df_final2[df_final2.Cluster.eq(1)]
resumo_c2 = df_final2[df_final2.Cluster.eq(2)]
resumo_c3 = df_final2[df_final2.Cluster.eq(3)]

# Resultado Cluster 0 
fig30 = px.pie(resumo_c0, values= 'Id', names= 'Sexo', title="Sexo",
            height=430, width= 420, template = "seaborn")
fig30.update_traces(textposition='inside')
fig30.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', legend=dict(
        title=None, orientation="h", y=-0.2, yanchor="bottom", x=0.5, xanchor="center"
    ))


idade_cluster0 = resumo_cluster.iloc[[0],2:4]
total_cluster0 = resumo_cluster.iloc[[0],1]


fig31 = px.pie(resumo_c0, values= 'Id', names= 'Estado_Civil', title="Estado Civil",
            height=430, width= 430, template = "seaborn")
fig31.update_traces(textposition='inside')
fig31.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', legend=dict(
        title=None, orientation="h", y=-0.2, yanchor="bottom", x=0.5, xanchor="center"
    ))


fig32 = px.pie(resumo_c0, values= 'Id', names= 'Educacao', title="Educação",
            height=430, width= 430, template = "seaborn")
fig32.update_traces(textposition='inside')
fig32.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', legend=dict(
        title=None, orientation="h", y=-0.2, yanchor="bottom", x=0.5, xanchor="center"
    ))

fig33 = px.pie(resumo_c0, values= 'Id', names= 'Ocupacao', title="Ocupação",
            height=430, width= 430, template = "seaborn")
fig33.update_traces(textposition='inside')
fig33.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', legend=dict(
        title=None, orientation="h", y=-0.2, yanchor="bottom", x=0.5, xanchor="center"
    ))


fig34 = px.pie(resumo_c0, values= 'Id', names= 'Tamanho_Cidade', title="Tamanho da Cidade",
            height=430, width= 430, template = "seaborn")
fig34.update_traces(textposition='inside')
fig34.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', legend=dict(
        title=None, orientation="h", y=-0.2, yanchor="bottom", x=0.5, xanchor="center"
    ))

# Resultado Cluster 1 
fig35 = px.pie(resumo_c1, values= 'Id', names= 'Sexo', title="Sexo",
            height=430, width= 420, template = "seaborn")
fig35.update_traces(textposition='inside')
fig35.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', legend=dict(
        title=None, orientation="h", y=-0.2, yanchor="bottom", x=0.5, xanchor="center"
    ))

idade_cluster1 = resumo_cluster.iloc[[1],2:4]
total_cluster1 = resumo_cluster.iloc[[1],1]


fig36 = px.pie(resumo_c1, values= 'Id', names= 'Estado_Civil', title="Estado Civil",
            height=430, width= 430, template = "seaborn")
fig36.update_traces(textposition='inside')
fig36.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', legend=dict(
        title=None, orientation="h", y=-0.2, yanchor="bottom", x=0.5, xanchor="center"
    ))


fig37 = px.pie(resumo_c1, values= 'Id', names= 'Educacao', title="Educação",
            height=430, width= 430, template = "seaborn")
fig37.update_traces(textposition='inside')
fig37.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', legend=dict(
        title=None, orientation="h", y=-0.2, yanchor="bottom", x=0.5, xanchor="center"
    ))

fig38 = px.pie(resumo_c1, values= 'Id', names= 'Ocupacao', title="Ocupação",
            height=430, width= 430, template = "seaborn")
fig38.update_traces(textposition='inside')
fig38.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', legend=dict(
        title=None, orientation="h", y=-0.2, yanchor="bottom", x=0.5, xanchor="center"
    ))


fig39 = px.pie(resumo_c1, values= 'Id', names= 'Tamanho_Cidade', title="Tamanho da Cidade",
            height=430, width= 430, template = "seaborn")
fig39.update_traces(textposition='inside')
fig39.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', legend=dict(
        title=None, orientation="h", y=-0.2, yanchor="bottom", x=0.5, xanchor="center"
    ))


# Resultado Cluster 2 
fig40 = px.pie(resumo_c2, values= 'Id', names= 'Sexo', title="Sexo",
            height=430, width= 420, template = "seaborn")
fig40.update_traces(textposition='inside')
fig40.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', legend=dict(
        title=None, orientation="h", y=-0.2, yanchor="bottom", x=0.5, xanchor="center"
    ))

idade_cluster2 = resumo_cluster.iloc[[2],2:4]
total_cluster2 = resumo_cluster.iloc[[2],1]

fig41 = px.pie(resumo_c2, values= 'Id', names= 'Estado_Civil', title="Estado Civil",
            height=430, width= 430, template = "seaborn")
fig41.update_traces(textposition='inside')
fig41.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', legend=dict(
        title=None, orientation="h", y=-0.2, yanchor="bottom", x=0.5, xanchor="center"
    ))

fig42 = px.pie(resumo_c2, values= 'Id', names= 'Educacao', title="Educação",
            height=430, width= 430, template = "seaborn")
fig42.update_traces(textposition='inside')
fig42.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', legend=dict(
        title=None, orientation="h", y=-0.2, yanchor="bottom", x=0.5, xanchor="center"
    ))
 

fig43 = px.pie(resumo_c2, values= 'Id', names= 'Ocupacao', title="Ocupação",
            height=430, width= 430, template = "seaborn")
fig43.update_traces(textposition='inside')
fig43.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', legend=dict(
        title=None, orientation="h", y=-0.2, yanchor="bottom", x=0.5, xanchor="center"
    ))


fig44 = px.pie(resumo_c2, values= 'Id', names= 'Tamanho_Cidade', title="Tamanho da Cidade",
            height=430, width= 430, template = "seaborn")
fig44.update_traces(textposition='inside')
fig44.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', legend=dict(
        title=None, orientation="h", y=-0.2, yanchor="bottom", x=0.5, xanchor="center"
    ))


# Resultado Cluster 3 
fig45 = px.pie(resumo_c3, values= 'Id', names= 'Sexo', title="Sexo",
            height=430, width= 420, template = "seaborn")
fig45.update_traces(textposition='inside')
fig45.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', legend=dict(
        title=None, orientation="h", y=-0.2, yanchor="bottom", x=0.5, xanchor="center"
    ))

idade_cluster3 = resumo_cluster.iloc[[3],2:4]
total_cluster3 = resumo_cluster.iloc[[3],1]


fig46 = px.pie(resumo_c3, values= 'Id', names= 'Estado_Civil', title="Estado Civil",
            height=430, width= 430, template = "seaborn")
fig46.update_traces(textposition='inside')
fig46.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', legend=dict(
        title=None, orientation="h", y=-0.2, yanchor="bottom", x=0.5, xanchor="center"
    ))


fig47 = px.pie(resumo_c3, values= 'Id', names= 'Educacao', title="Educação",
            height=430, width= 430, template = "seaborn")
fig47.update_traces(textposition='inside')
fig47.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', legend=dict(
        title=None, orientation="h", y=-0.2, yanchor="bottom", x=0.5, xanchor="center"
    ))

   
fig48 = px.pie(resumo_c3, values= 'Id', names= 'Ocupacao', title="Ocupação",
            height=430, width= 430, template = "seaborn")
fig48.update_traces(textposition='inside')
fig48.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', legend=dict(
        title=None, orientation="h", y=-0.2, yanchor="bottom", x=0.5, xanchor="center"
    ))


fig49 = px.pie(resumo_c3, values= 'Id', names= 'Tamanho_Cidade', title="Tamanho da Cidade",
            height=430, width= 430, template = "seaborn")
fig49.update_traces(textposition='inside')
fig49.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', legend=dict(
        title=None, orientation="h", y=-0.2, yanchor="bottom", x=0.5, xanchor="center"
    ))


eap6= st.sidebar.checkbox('6- RESULTADOS')
st.sidebar.write('-----6.1- Resumo dos Clusters')
st.sidebar.write('-----6.2- Resultado da Clusterização')
st.sidebar.write('-----6.3- Comparativo entre Clusters')
st.sidebar.write('-------6.3.1- Gráficos de Comparação')
st.sidebar.write('-----6.4- Resumo por Cluster')
st.sidebar.write('-------6.4.1- Cluster 0')
st.sidebar.write('-------6.4.1- Cluster 1')
st.sidebar.write('-------6.4.1- Cluster 2')
st.sidebar.write('-------6.4.1- Cluster 3')

if eap6:
    st.header('6.1- Resumo dos Clusters')
    st.write(cluster1)
    st.subheader('6.2- Resultado da Clusterização')
    st.write(df_final2)
    st.subheader('6.3- Comparativo entre Clusters')
    st.write(resumo_cluster)
    st.subheader('6.3.1- Gráficos de Comparação')
    st.plotly_chart(fig14)
    st.plotly_chart(fig15)
    st.plotly_chart(fig16)
    st.plotly_chart(fig17)
    st.plotly_chart(fig18)
    st.plotly_chart(fig19)
    st.plotly_chart(fig20)
    st.plotly_chart(fig21)
    st.plotly_chart(fig22)
    st.plotly_chart(fig23)
    st.plotly_chart(fig24)
    st.plotly_chart(fig25)
    st.plotly_chart(fig26)
    st.plotly_chart(fig27)
    st.plotly_chart(fig28)
    st.plotly_chart(fig29)
    st.subheader('6.4- Resumo por Cluster')
    st.subheader('6.4.1- Cluster 0')
    coluna26, coluna27 = st.columns(2)
    with coluna26:
        st.write(total_cluster0) 
        st.write(idade_cluster0) 
    with coluna27:
        st.plotly_chart(fig30)
    coluna28, coluna29 = st.columns(2)
    with coluna28:  
        st.plotly_chart(fig31) 
    with coluna29:
        st.plotly_chart(fig32)
    coluna30, coluna31 = st.columns(2)
    with coluna30:  
        st.plotly_chart(fig33) 
    with coluna31:
        st.plotly_chart(fig34)
    st.subheader('6.4.2- Cluster 1')
    coluna32, coluna33 = st.columns(2)
    with coluna32:
        st.write(total_cluster1) 
        st.write(idade_cluster1)    
    with coluna33:
        st.plotly_chart(fig35)
    coluna34, coluna35 = st.columns(2)
    with coluna34:  
        st.plotly_chart(fig36) 
    with coluna35:
        st.plotly_chart(fig37)
    coluna36, coluna37 = st.columns(2)
    with coluna36:  
        st.plotly_chart(fig38) 
    with coluna37:
        st.plotly_chart(fig39)
    st.subheader('6.4.3- Cluster 2')
    coluna38, coluna39 = st.columns(2)
    with coluna38:
        st.write(total_cluster2) 
        st.write(idade_cluster2) 
    with coluna39:
        st.plotly_chart(fig40)
    coluna40, coluna41 = st.columns(2)
    with coluna40:  
        st.plotly_chart(fig41) 
    with coluna41:
        st.plotly_chart(fig42)
    coluna42, coluna43 = st.columns(2)
    with coluna42:  
        st.plotly_chart(fig43) 
    with coluna43:
        st.plotly_chart(fig44)
    st.subheader('6.4.4- Cluster 3')
    coluna44, coluna45 = st.columns(2)
    with coluna44:
        st.write(total_cluster3) 
        st.write(idade_cluster3) 
    with coluna45:
        st.plotly_chart(fig45)
    coluna46, coluna47 = st.columns(2)
    with coluna46:  
        st.plotly_chart(fig46) 
    with coluna47:
        st.plotly_chart(fig47)
    coluna48, coluna49 = st.columns(2)
    with coluna48:  
        st.plotly_chart(fig48) 
    with coluna49:
        st.plotly_chart(fig49)
