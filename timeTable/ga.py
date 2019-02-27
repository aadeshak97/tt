import threading
import numpy as np
import pandas as pd 
import random
from organization.models import Faculty
all_faculty=Faculty.objects.all()
Faculty_id

class FitnessUtil:
   def __init__(self):
       self.nu_class=2
       self.nu_days=2
       self.nu_hours=2
       
       self.rows=[]
       for i in range(1,self.nu_class+1):
           for j in range(1,self.nu_days+1):
               for k in range(1,self.nu_hours+1):
                   self.rows.append(str(i)+str(j)+str(k))
       
       
       
       self.cols=np.array(['111','212','323','424'])
       self.teacher_load=np.array([2,2,2,2])
       self.subject_load=np.array([1,1,1,1])
       self.matrix=pd.DataFrame(np.zeros((self.nu_class*self.nu_days*self.nu_hours,len(self.cols)),dtype=int),index=[self.rows],columns=[self.cols])
       
   def getFitness(self,pop):
       
       lng=len(pop)
       fitness = []
       for p_i in range(lng):
           chrm_ind = pop[p_i,:]
           matrix1=pd.DataFrame(np.zeros((self.nu_class*self.nu_days*self.nu_hours,len(self.cols)),dtype=int),index=[self.rows],columns=[self.cols])
           chrom = self.cols[chrm_ind]
           print(chrom)
           r=0
           
           for i in range(0,len(chrom)):
               index=chrom[i]
               index_load=int(index[0:1])-1
               load=0
               for j in range(1,self.nu_class+1):
                   sub_load=0
                   for k in range(1,self.nu_days+1):
                       if((load==self.teacher_load[index_load]) or (sub_load==self.subject_load[index_load])):
                           break;
                       a=self.nu_hours+2
                       for l in range(1,self.nu_hours+1):
                           b=str(j)+str(k)+str(l)
                           if(matrix1.loc[b][index].values[0]==0):
                               matrix1.loc[b,:].values[0]=-1
                               print("hello")
                               matrix1.loc[b][index].values[0]=1
                               print(matrix1.loc[b][index].values[0])
                               load=load+1
                               a=l
                               sub_load=sub_load+1
                               break;
                       if(a<=self.nu_hours+1):
                           for n in range(1,self.nu_class+1):
                               if(n!=j):
                                   q=str(n)+str(k)+str(a)
                                   matrix1.loc[q][index].values[0]=-1
                          
                    
               r=r+((matrix1[:][index].values)>0).sum()
               
                    
           print(matrix1) 
           matrix1=self.matrix
           fitness.append(r/self.teacher_load.sum())
       print(fitness)       
       return fitness     
   

    
    
    
class SGA(object):
    _instance = None
    _lock = threading.Lock()

    def __new__(cls,min, max, sol_per_pop, num_weights, num_generations):
        if SGA._instance is None:
            with SGA._lock:
                if SGA._instance is None:
                    SGA._instance = super(SGA, cls).__new__(cls)
        return SGA._instance

    def __init__(self, min, max, sol_per_pop, num_weights, num_generations):
        self.min = min
        self.max = max
        self.sol_per_pop = sol_per_pop
        self.num_weights = num_weights
        self.num_generations = num_generations
        self.fitUtil = FitnessUtil()
   
    #@staticmethod 
    def getInstance(min, max, sol_per_pop, num_weights, num_generations):
       """ Static access method. """
       if SGA._instance == None:
          SGA._instance = SGA(min, max, sol_per_pop, num_weights, num_generations)
       return SGA._instance
   

    def callGA(self):
        #print("hi")
        
        lst = []
        for x in range(self.sol_per_pop):
            a=list(range(self.num_weights))
            random.shuffle(a)
            lst.append(a)
        new_population = np.array(lst)
        num_parents_mating = self.sol_per_pop
        best_outputs = []
        
        for generation in range(self.num_generations):
            #print("Generation : ", generation)
            # Measuring the fitness of each chromosome in the population.
            fitness = self.cal_pop_fitness(new_population)
            #print("Fitness")
            #print(fitness)

            #best_outputs.append(np.max(np.sum(new_population*equation_inputs, axis=1)))
            # The best result in the current iteration.
            #print("Best result : ", np.max(np.sum(new_population*equation_inputs, axis=1)))
            
            # Selecting the best parents in the population for mating.
            parents = self.select_mating_pool(new_population, fitness, num_parents_mating)
            #print("Parents")
            #print(parents)

            # Generating next generation using crossover.
            offspring_crossover = self.crossover(parents, offspring_size=(self.sol_per_pop[0]-parents.shape[0], self.num_weights))
            #print("Crossover")
            #print(offspring_crossover)

            # Adding some variations to the offspring using mutation.
            offspring_mutation = self.mutation(offspring_crossover)
            #print("Mutation")
            #print(offspring_mutation)

            # Creating the new population based on the parents and offspring.
            new_population[0:parents.shape[0], :] = parents
            new_population[parents.shape[0]:, :] = offspring_mutation

    #At first, the fitness is calculated for each solution in the final generation.
        fitness = self.cal_pop_fitness(equation_inputs, new_population)
    # Then return the index of that solution corresponding to the best fitness.
        best_match_idx = np.where(fitness == np.max(fitness))
        return best_match_idx;



    def cal_pop_fitness(self, pop):
    # Calculating the fitness value of each solution in the current population.
    # The fitness function calulates the sum of products between each input and its corresponding weight.
        return self.fitUtil.getFitness(pop)

    def select_mating_pool(self, pop, fitness, num_parents):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
        parents = np.empty((num_parents, pop.shape[1]))
        for parent_num in range(num_parents):
            max_fitness_idx = np.where(fitness == np.max(fitness))
            max_fitness_idx = max_fitness_idx[0][0]
            parents[parent_num, :] = pop[max_fitness_idx, :]
            fitness[max_fitness_idx] = -99999999999
        return parents

    def crossover(self, parents, offspring_size):
        offspring = np.empty(offspring_size)
    # The point at which crossover takes place between two parents. Usually, it is at the center.
        crossover_point = np.uint8(offspring_size[1]/2)

        for k in range(offspring_size[0]):
        # Index of the first parent to mate.
            parent1_idx = k%parents.shape[0]
        # Index of the second parent to mate.
            parent2_idx = (k+1)%parents.shape[0]
        # The new offspring will have its first half of its genes taken from the first parent.
            offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        # The new offspring will have its second half of its genes taken from the second parent.
            offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
        return offspring

    def mutation(self, offspring_crossover):
        # Mutation changes a single gene in each offspring randomly.
        for idx in range(offspring_crossover.shape[0]):
            # The random value to be added to the gene.
            random_value = np.random.uniform(-1.0, 1.0, 1)
            offspring_crossover[idx, 4] = offspring_crossover[idx, 4] + random_value
        return offspring_crossover

