#Problema das n.Rainhas com Algoritmos Genéticos
#####################################################################
# Autor: Vilson Soares de Siqueira                                  #
# Data: 30/09/2018                                                  #
#####################################################################

import numpy as np
import random
from operator import itemgetter

class Nrainhas:
    def __init__(self,n):
        self.n = n
        self.rainhas = [i for i in range(0,self.n)]

    def GerarIndividuo(self):
        i = 0
        coluna = self.rainhas[:]
        individuo = []
        i = 0
        while coluna != [] and len(coluna) > 0:
            gene = random.randrange(0,len(coluna))
            individuo.append(coluna[gene])
            coluna.pop(gene)
            i += 1
        return individuo


    def RepresentarIndividuo(self,individuo):

        representacao = np.zeros((self.n, self.n), dtype=np.int64)
        for i in range(0,len(individuo)):
            representacao[i][individuo[i]] = 1
        return representacao

    def buscaDiagPrincipal(self,RepresentacaoIndividuo,linha,coluna):
        res = False
        # O canto inferior direito e canto superior esquerdo não possui apenas a sua posição
        if (linha == 0 and coluna == self.n - 1) or (coluna == 0 and linha == self.n-1):
            res = True
            return res
        else:

            # Sobe ma diagonal principal
            l,c = linha, coluna
            while l > 0 and c > 0:
                l -= 1
                c -= 1
                if RepresentacaoIndividuo[l][c] == 1:
                    res = True
                    return res

            # Desce ma diagonal principal
            l, c = linha, coluna
            while l < self.n - 1 and c < self.n-1:
                l += 1
                c += 1
                if RepresentacaoIndividuo[l][c] == 1:
                    res = True
                    return res
        return res

    def buscaDiagSecundaria(self,RepresentacaoIndividuo,linha,coluna):
        res = False
        if (linha == 0 and coluna == 0) or (linha == self.n-1 and coluna == self.n-1):
            res = True
            return res
        else:

            # Sobe na diagonal secundaria
            l,c = linha, coluna

            while l > 0 and c > self.n-1:
                l -= 1
                c += 1
                if RepresentacaoIndividuo[l][c] == 1:
                    res = True
                    return res

            # Sobe na diagonal secundaria
            l, c = linha, coluna
            while l < self.n - 1 and c > 0:
                l += 1
                c -= 1
                if RepresentacaoIndividuo[l][c] == 1:
                    res = True
                    return res
        return res

    def buscaLinha(self,RepresentacaoIndividuo,linha,coluna):
        res = False
        i = 0
        while i < len(RepresentacaoIndividuo):
            if coluna != i:
                if RepresentacaoIndividuo[linha][i] == 1:
                    res = True
                    return res
            i += 1
        return res

    def buscaColuna(self,RepresentacaoIndividuo,linha,coluna):
        res = False
        i = 0
        while i < len(RepresentacaoIndividuo):
            if linha != i:
                if RepresentacaoIndividuo[i][coluna] == 1:
                    res = True
                    return res
            i += 1
        return res

    def fitnes(self,RepresentacaoIndividuo,individuo):
        score = 0
        for i in range(0,len(individuo)):
            dp = Nrainhas.buscaDiagPrincipal(self,RepresentacaoIndividuo, i, individuo[i])
            ds = Nrainhas.buscaDiagSecundaria(self,RepresentacaoIndividuo, i, individuo[i])
            l = Nrainhas.buscaLinha(self,RepresentacaoIndividuo, i, individuo[i])
            c = Nrainhas.buscaColuna(self,RepresentacaoIndividuo, i, individuo[i])
            if dp == True:
                score += 1
            if ds == True:
                score += 1
            if l == True:
                score += 1
            if c == True:
                score += 1
        return score

class AlgoritmoGenetico(Nrainhas):

    # Cria a População Inicial
    def PopulacaoInicial(self,TamPop):
        Populacao = []
        for i in range(0,TamPop):
            ind = Nrainhas.GerarIndividuo(self)
            Populacao.append(ind)
        return Populacao

    # Avaliação da População
    def AvaliacaoPopulacao(self,Populacao):
        Avaliacao = []
        populacaoOrdenada = []
        for i in range(0,len(Populacao)):
            RI = AlgoritmoGenetico.RepresentarIndividuo(self,Populacao[i])
            avaliacaoIndividuo = AlgoritmoGenetico.fitnes(self,RI,Populacao[i])
            Avaliacao.append([i, avaliacaoIndividuo])

        # ordenar as melhores solucoes: Elitismo
        Avaliacao = sorted(Avaliacao, key=itemgetter(1, 0))
        for i in range(0, len(Populacao)):
            # Organiza os Individuos da População de acordo com sua Avaliação
            populacaoOrdenada.append(Populacao[Avaliacao[i][0]])

        return Avaliacao[0:100], populacaoOrdenada[0:100]

    # Seleção por Torneio
    def selecaoTorneio(self, avaliacao, K=3):
        pais = []

        for torneio in range(0, len(avaliacao)):
            competidores = []

            for i in range(K):
                indice = random.randrange(0, len(avaliacao))
                competidores.append(indice)

            vencedor = competidores[0]
            mfitness = avaliacao[vencedor][1]
            for i in range(1, K):
                fitness = avaliacao[competidores[i]][1]
                if fitness <= mfitness:
                    mfitness = fitness
                    vencedor = competidores[i]
                pais.append(vencedor)
        return pais

    # Operador CX - Cycle Crossover. (OLIVER; SMITH; HOLLAND,1987)
    def OperadorCX(self, pai1, pai2):

        # realiza o crossover dos pais selecionados
        for v in range(0, len(pai1)):
            p1 = pai1[:]  # Copia dos pais
            p2 = pai2[:]

            filho1, filho2 = p1, p2  # copia o genes dos pais para o filhos

            i, j = 0, 0
            index1, index2 = None, None

            while index1 != 0:
                if i == 0:
                    filho1[0] = p1[0]
                    index1 = p2.index(filho1[0])
                    i += 1
                else:
                    filho1[index1] = p1[index1]
                    index1 = p2.index(filho1[index1])

            while index2 != 0:
                if j == 0:
                    filho2[0] = p2[0]
                    index2 = p1.index(filho2[0])
                    j += 1
                else:
                    filho2[index2] = p2[index2]
                    index2 = p1.index(filho2[index2])

        return filho1, filho2

    # Operador PMX (Partially Mapped Crossover) (Goldberg e Lingle Jr 1985) Original
    def OperadorPMX(self, pai1, pai2):

            Sfilho1, Sfilho2 = [], []  # filhos

            # realiza o crossover em cada sub-lista
            for v in range(0, len(pai1)):
                p1 = pai1[:]  # Pega os pais de uma sub-lista
                p2 = pai2[:]

                filho1, filho2 = pai1[:], pai2[:]  # copia o genes dos pais para o filhos
                Genesfal1, Genesfal2 = [], []  # genes faltantes vazio

                # sorteia aleatoreamente 2 pontos para criar a partição sendo que i < j
                i = random.randrange(0, len(p1) - 1)
                j = random.randrange(i + 1, len(p1))

                # troca os genes do intervado definido (i,j) do pai1 para o filho2 e do pai2 para o filho1
                for a in range(i, j):
                    filho1[a] = p2[a]
                    filho2[a] = p1[a]

                # Pega os genes faltantes dos pais que estão faltando nos filhos
                [Genesfal1.append(p1[i]) for i in range(0, len(p1)) if p1[i] not in filho1]
                [Genesfal2.append(p2[i]) for i in range(0, len(p2)) if p2[i] not in filho2]

                k1, k2 = 0, 0  # variaveis de controle para os genes menores que i

                # Troca os genes duplicados
                for m in range(0, len(p1)):
                    if m < i:  # genes faltantes antes do conte i
                        if p1[m] in Genesfal2:  # se o gene do filho1 for duplicado troca pelo gene faltante do pai1
                            filho1[m] = Genesfal1[k1]
                            k1 += 1

                        if p2[m] in Genesfal1:
                            filho2[m] = Genesfal2[k2]
                            k2 += 1

                    elif m >= j:  # genes faltantes depois do conte j
                        if p1[m] in Genesfal2:  # se o gene do filho2 for duplicado troca pelo gene faltante do pai2
                            filho1[m] = Genesfal1[k1]
                            k1 += 1

                        if p2[m] in Genesfal1:
                            filho2[m] = Genesfal2[k2]
                            k2 += 1

            return filho1, filho2
    
    # Define o Crossover que será aplicado
    def Crossover(self,populacao,pais, tam_pop, taxa_crossover = 0.7):

        nova_pop = []
        for i in range((tam_pop // 2)-1):

            pai1 = random.choice(pais)
            pai2 = random.choice(pais)

            if random.random() < taxa_crossover:
                # opções implementadas Operadores CX e PMX
                #filho1,filho2 = AlgoritmoGenetico.OperadorCX(self,populacao[pai1],populacao[pai2])
                filho1,filho2 = AlgoritmoGenetico.OperadorPMX(self,populacao[pai1],populacao[pai2])
                nova_pop.append(filho1)
                nova_pop.append(filho2)
            else:
                nova_pop.append(populacao[pai1])
                nova_pop.append(populacao[pai2])

        return nova_pop
    # Mutação SWAP
    def Mutacao(self,novaPopulacao,pbmut = 0.05):
        # Aplica a Mutação dos individuos da população, se probabilidade defenida for True
        for m in range(2,len(novaPopulacao)):
            if random.random() < pbmut:
                gene1 = random.randrange(0,len(novaPopulacao[m])-1)
                gene2 = random.randrange(0,len(novaPopulacao[m])-1)

                novaPopulacao[m][gene1], novaPopulacao[m][gene2] = novaPopulacao[m][gene2], novaPopulacao[m][gene1]

        return novaPopulacao
    
    # Passar apenas os 2 melhores individuos para a próxima geração
    def Elitismo(self,populacao,n):
        Elistismo = populacao[0:n]
        return Elistismo

    def AG(self,TamPop=100,geracoes=100,K=3,pbcx=0.8,pbmut=0.5):
        # Cria a população
        Populacao = AlgoritmoGenetico.PopulacaoInicial(self,TamPop)
    
        # avaliação dos indivíduos da População
        Avaliacao, PopulacaoOrdenada = AlgoritmoGenetico.AvaliacaoPopulacao(self,Populacao)
        
        for i in range(geracoes):
            # seleção por torneio
            pais = AlgoritmoGenetico.selecaoTorneio(self,Avaliacao,K)
            
            # Aplica o Crossover e Mutação
            NPopulacao = AlgoritmoGenetico.Crossover(self, PopulacaoOrdenada, pais, TamPop, pbcx)
            NPopulacao = AlgoritmoGenetico.Mutacao(self, NPopulacao, pbmut)
            
            # Retirar o comentário para usar o metódo de seleção para a próxima geração
            
            # Metódo Ranking, ordena e passa os melhores individuos para a próxima geração
            # se usar está opção : definir a taxa de  mutação para (0.1 - 0.5)
            #populacao = PopulacaoOrdenada + NPopulacao
            
            # Passa n melhores individuos para a próxima geração
            # se usar está opção : definir a taxa de mutação para (0.05 - 0.1)
            Elitismo = AlgoritmoGenetico.Elitismo(self, PopulacaoOrdenada,n=2)
            populacao = Elitismo + NPopulacao
            
            # Avaliação da População que passará para a próxima geração
            Avaliacao, PopulacaoOrdenada = AlgoritmoGenetico.AvaliacaoPopulacao(self, populacao)

            print('Geracao:', i + 1, '| Evolução -> Melhor:', Avaliacao[0][1], 'Pior: ', Avaliacao[-1][1])
            if Avaliacao[0][1] == 0:
                print("\n#### Criou uma solução na Geração: ", i + 1, '#########\n\n')
                individuo, fitness = AlgoritmoGenetico.RepresentarIndividuo(self, PopulacaoOrdenada[0]), Avaliacao[1]
                return individuo, fitness

        individuo, fitness = AlgoritmoGenetico.RepresentarIndividuo(self, PopulacaoOrdenada[0]), Avaliacao[1]
        return individuo, fitness

        individuo, fitness = AlgoritmoGenetico.RepresentarIndividuo(self, PopulacaoOrdenada[0]), Avaliacao[1]
        return individuo, fitness


rainhas = AlgoritmoGenetico(60)
ind,fit = rainhas.AG(100,1000,3,0.8,0.1)
# imprime o resultado
for i in range(0,len(ind)):
    print(ind[i])
