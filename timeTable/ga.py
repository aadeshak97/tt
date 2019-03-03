import threading
import numpy as np
import pandas as pd
import random
from course.models import Course
from organization.models import Faculty
from timeTable.models import time_table

class FitnessUtil:
   def __init__(self):
       self.nu_class= Course.objects.all().count()#hould be collected from db
       
       b=Course.objects.values_list('working_days')
       self.nu_days=(b[0])[0]
       
       c=Course.objects.values_list('working_hrs')
       self.nu_hours=(c[0])[0]

       self.rows=[]
       self.ll=Course.objects.values_list('pk',flat=True)
       for i in range(self.nu_class):

           for j in range(1,self.nu_days+1):
               for k in range(1,self.nu_hours+1):
                   self.rows.append(str(self.ll[i])+str(j)+str(k))


       u=Assignment.objects.filter(is_assign=True).values_list('faculty__id','subject__subject_type','subject')
      
       k=[]
       for x in u:
         if x[0]>=1 and x[0]<=9:
           p="0"+str(x[0])
         else:
           p=str(x[0])
         if x[2]>=1 and x[2]<=9:
           q="0"+str(x[2])
         else:
           q=str(x[2])
    
        k.append(p+str(x[1])+q)
    
       self.cols=k
       d=Faculty.objects.values_list('Faculty_load',flat=True)
       self.teacher_load=np.array(d)
       e=Subject.objects.values_list('period_per_week',flat=True)
       self.subject_load=np.array(e)
       self.matrix=pd.DataFrame(np.zeros((self.nu_class*self.nu_days*self.nu_hours,len(self.cols)),dtype=int),index=[self.rows],columns=[self.cols])
       for cha in range(len(self.cols)):
           ids=self.cols[cha]
           ids=int(ids[3:])
           for j_class in range(len(self.ll)):
                uu=Subject.objects.filter(id=ids).values_list('course__id')
              mm=(u[0])[0]
              if(self.ll[j_class]!=mm):
                f=str(self.ll[j_class])+str(1)+str(1)
                l=str(self.ll[j_class]+1)+str(1)+str(1)
                self.matrix.at[f:l,self.cols[cha]]=-1 

    def getFitness(self,pop):

       lng=len(pop)#calculate the number of chromosome in population
       fitness = []
       for p_i in range(lng):
           chrm_ind = pop[p_i,:]
           self.matrix1=pd.DataFrame(np.zeros((self.nu_class*self.nu_days*self.nu_hours,len(self.cols)),dtype=int),index=[self.rows],columns=[self.cols])
           self.chrom = self.cols[chrm_ind]
           #print(chrom)
           r=0

           for c_i in range(0,len(self.chrom)):
               index=self.chrom[c_i]
               index_load=int(index[0:2])-1
               load=0
               for j in range(self.nu_class):
                   sub_load=0
                   for k in range(1,self.nu_days+1):
                       if((load==self.teacher_load[index_load]) or (sub_load==self.subject_load[index_load])):
                           break;
                       a=self.nu_hours+2
                       for l in range(1,self.nu_hours+1):
                           b=str(self.ll[j])+str(k)+str(l)
                           if(self.matrix1.loc[b][index].values[0]==0):
                               #self.matrix1.at[b,:].values[0]=-1
                               #print(self.matrix1.loc[b,:].values[0])
                               #self.matrix1.loc[b][index].values[0]=1
                               self.matrix1.at[b,:]=-1
                               self.matrix1.at[b,index]=1

                               load=load+1
                               a=l
                               sub_load=sub_load+1
                               break;
                       if(a<=self.nu_hours+1):
                           for n in range(self.nu_class):
                               if(self.ll[n]!=self.ll[j]):
                                   q=str(self.ll[n])+str(k)+str(a)
                                   self.matrix1.at[q,index]=-1


               r=r+((self.matrix1[:][index].values)>0).sum()
           if(r/(self.nu_class*self.subject_load.sum())==1):
               p=settle(self.matrix1,self.chrom,self.nu_class,self.nu_days,self.nu_hours,self.ll)
               p.class_data()



           print(self.matrix1)
           #self.matrix1=self.matrix

           fitness.append(r/(self.nu_class*self.subject_load.sum()))
       print(fitness)
       return fitness

class settle:
    def __init__(self,matrix1,chrom,nu_class,nu_days,nu_hours,ll):
        self.matrix1=matrix1
        self.chrom=chrom
        self.nu_class=nu_class
        self.nu_days=nu_days
        self.nu_hours=nu_hours
        self.ll=ll
    def class_data(self):
        lis_data=[]
        for i in range(self.nu_class+1):
            lis_class=[]
            for j in range(1,self.nu_days+1):
                for k in range(1,self.nu_hours+1):
                    for l in range(len(self.chrom)):
                        index_col=self.chrom[l]
                        index_row=str(self.ll[i])+str(j)+str(k)
                        if(self.matrix1.loc[index_row][index_col].values[0]==1):
                                lis_class.append(index_row+index_col)
                                break;
            lis_data.append(lis_class)
        print("again")
        print(lis_data)

        for cl in lis_data:
            for ch in cl:
                class_id=int(ch[0:1])
                day=int(ch[1:2])
                hour=int(ch[2:3])
                lec_id=int(ch[3:5])
                typ=int(ch[5:6])
                subj_id=int(ch[6:8])
                b=time_table(semester=1,period=1,days=1,lect=1,types=1,sub=1)
                b.save()





class SGA(object):
    _instance = None
    _lock = threading.Lock()

    def __new__(cls,sol_per_pop, num_weights, num_generations):
        if SGA._instance is None:
            with SGA._lock:
                if SGA._instance is None:
                    SGA._instance = super(SGA, cls).__new__(cls)
        return SGA._instance

    def __init__(self, sol_per_pop, num_weights, num_generations):
        self.sol_per_pop = sol_per_pop
        self.num_weights = num_weights
        self.num_generations = num_generations
        self.fitUtil = FitnessUtil()

    #@staticmethod
    def getInstance(sol_per_pop, num_weights, num_generations):
       """ Static access method. """
       if SGA._instance == None:
          SGA._instance = SGA(sol_per_pop, num_weights, num_generations)
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
            print("Parents")
            print(parents)

            # Generating next generation using crossover.
            #offspring_size=(self.sol_per_pop-parents.shape[0], self.num_weights)
            offspring_crossover = self.crossover(parents, offspring_size=(2*self.sol_per_pop-parents.shape[0], self.num_weights))
            print("Crossover")
            print(offspring_crossover)

            # Adding some variations to the offspring using mutation.
            offspring_mutation = self.mutation(offspring_crossover)
            print("Mutation")
            print(offspring_mutation)

            # Creating the new population based on the parents and offspring.
            new_population[0:parents.shape[0], :] = parents
            #new_population[parents.shape[0]:, :] = offspring_mutation

    #At first, the fitness is calculated for each solution in the final generation.
        fitness = self.cal_pop_fitness(new_population)
    # Then return the index of that solution corresponding to the best fitness.
        #self.fitUtil=class_data()
        best_match_idx = np.where(fitness == np.max(fitness))
        print("best")
        print(best_match_idx)
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

        for k in range(offspring_size[0]-1):
        # Index of the first parent to mate.

            parent1_idx = k%parents.shape[0]
        # Index of the second parent to mate.
            #print(parent1_idx[0])
            print("parr")
            parent2_idx = (k+1)%parents.shape[0]
        # The new offspring will have its first half of its genes taken from the first parent.
            cycle=[]
            key=parents[parent1_idx,0]
            temp=parents[parent2_idx,0]
            cycle.append(0)
            for c_j in range(99999999999):
                if(parents[parent1_idx,c_j%self.num_weights]==temp):
                    temp=parents[parent2_idx,c_j%self.num_weights]
                    cycle.append(c_j%self.num_weights)
                if(temp==key):
                    break;
            for cy in range(len(cycle)):
                r=cycle[cy]
                #temp=parent1_idx[k]
                #offspring[k,:]=parent2_idx[k]
                #offspring[k+1,:]=temp
                temp=parents[parent1_idx,r]
                parents[parent1_idx,r]=parents[parent2_idx,r]
                parents[parent2_idx,r]=temp

            offspring[k,:]=parents[parent1_idx,:]
            offspring[k+1,:]=parents[parent2_idx,:]
            k=k+1

            #offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        # The new offspring will have its second half of its genes taken from the second parent.
            #offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
        return offspring

    def mutation(self, offspring_crossover):
        # Mutation changes a single gene in neach offspring randomly.
        for idx in range(offspring_crossover.shape[0]):
            # The random value to be added to the gene.
            random_value = np.random.randint(0, self.num_weights)#data base$$$$$$$
            random_value1=np.random.randint(0,self.num_weights)
            #print(offspring_crossover[idx,2])
            print(random_value)
            print(random_value1)
            temp=offspring_crossover[idx,random_value1]
            offspring_crossover[idx,random_value1]=offspring_crossover[idx,random_value]
            offspring_crossover[idx,random_value]=temp
            #   offspring_crossover[idx, 2] = offspring_crossover[idx, 2] + random_value
        return offspring_crossover

#obj = SGA.getInstance(7,4,11)
#obj.callGA()
