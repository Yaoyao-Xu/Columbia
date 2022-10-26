import random
from collections import deque
import numpy as np


iteration = 1
time = 1
best = float('inf')
MEA_record = []


f = open(r"data.txt")
line = f.readline()
x = []
y = []
while line:
    num = list(map(float,line.split(',')))
    x.append(round(num[0],5))
    y.append(round(num[1],5))
    line = f.readline()
f.close()

x_train, y_train = np.array(x), np.array(y)
#x_valid, y_valid = np.array(x[600 :]), np.array(y[ 600 :])

def constant():
    p = random.random()
    if p > 0.3:
        return float(random.randint(-10, 10))
    else:
        return 'x'

def add(a, b):
    return np.add(a, b) 

def minus(a,b):
    return np.subtract(a,b)

def mul(a,b):
    return np.multiply(a,b)

def divd(a ,b):
        return np.divide(a, np.add(b,0.001))

def sin(a):
    return np.sin(a)

def cos(a):
    return np.cos(a)

operator = [add, minus, mul, divd, sin, cos]
population = 300

def fitness(a, b):
    difference = np.subtract(a,b)
    fitness = np.absolute(difference)
    total = np.sum(fitness, axis = 0)
    avg = np.divide(total, 600)
    return round(avg,6)

def generate(operator):
    p = random.random()
    if p > 0.35:
        return operator[random.randint(0,5)]
    else:
        return constant()

def createheap(operator):
    depth = 5
    output = []
    cur = 0
    
    q = deque([operator[random.randint(0,5)]])
        
    while cur <= depth:
        size = len(q)
        for _ in range(size):
            node = q.popleft()
            output.append(node)
            if node not in operator:
                q.append(None)
                q.append(None)
            elif cur == depth - 1:
                if node == operator[4] or node == operator[5]:
                    q.append('x')
                    q.append(None)
                else:
                    q.append(constant())
                    q.append('x')
            elif node == operator[4] or node == operator[5]:
                left_node = generate(operator)
                q.append(left_node)
                q.append(None)
            else:
                left_node = generate(operator)
                right_node = generate(operator)
                q.append(left_node)
                q.append(right_node)
        cur += 1 
        equation = output.copy()
    return output,  equation
            
def findconstant(matrix):
    candidate = []
    for i in range(len(matrix)):
        if type(matrix[i]) == float:
            candidate.append(i)

    return candidate

def mutation(matrix, candidate, operator, t):
    
    limit = 3
    for _ in range(limit):
        if len(candidate) != 0:
            for i in range(len(candidate)):
                idx = candidate[i]
                matrix[idx] += random.random()* random.randint(-10, 10)
                    
        
        idx = random.randint(0, 40)
        while matrix[idx] not in operator:
            idx = random.randint(0, 40)
      
        if matrix[idx] in operator[:4]:
            new = random.randint(0, 3)
            matrix[idx] = operator[new]
        elif matrix[idx] in operator[4:]:
            new = random.randint(4, 5)
            matrix[idx] = operator[new]
        
    return  matrix


def evaluate(matrix, x_train, operator):
    for i in range(len(matrix)-1, -1, -1):
        if matrix[i] == 'x':
            matrix[i] = x_train
        else:
            if matrix[i] == operator[4] or matrix[i] == operator[5]:
                if i == 0:
                    matrix[0] = matrix[0](matrix[1])
                else:
                    matrix[i] = matrix[i](matrix[2*i + 1])
            elif matrix[i] == operator[0] or matrix[i] == operator[1] or matrix[i] == operator[2] or matrix[i] == operator[3]:
                if i == 0:
                    matrix[0] = matrix[0](matrix[1], matrix[2])
                else:
                    matrix[i] = matrix[i](matrix[2*i + 1], matrix[2*i + 2])
    return matrix[0]

def find_crossingidx(matrix1, matrix2, operator):
    for j in range(len(matrix1) - 1, -1, -1):
        if matrix1[j] in operator:
            function1_idx = j
            break
        
    
    for i in range(len(matrix2) - 1, -1, -1):
        if matrix2[i] in operator:
            function2_idx = i
            break
    return [function1_idx, function2_idx]

def crossing(i, j, matrix1, matrix2):
    children = matrix1.copy()
    if children[i] not in operator[4:] and matrix2[j] not in operator[4:]:
        children[i], children[2*i + 1], children[2*i + 2] = matrix2[j], matrix2[2*j + 2], matrix2[2 *j +1]
    else:
        children[i], children[2*i + 1], children[2*i + 2] = matrix2[j], matrix2[2*j + 1], matrix2[2 *j + 2]
    return children
    
def rank(dic, y_train, x_train, operator):
    rank_dic = []
    output_dic = []
    totalpoint = []
    for i in range(600):
        result = evaluate(dic[i].copy(), x_train, operator)
        MEA = fitness(y_train, result)
        rank_dic.append([MEA, dic[i]])
        totalpoint.append(MEA)
    rank_dic = sorted(rank_dic, key = lambda x: x[0])
    
    count = 0
    for j in range(population):
        output_dic.append(rank_dic[j][1])
        if rank_dic[j][0] <= 0.8:
            count += 1 / 300
    return output_dic, rank_dic[0][0], round(count, 3), totalpoint
    

initial_dictionary = []

for i in range(population):
    output, equation = createheap(operator)
    initial_dictionary.append(output)
#print(dictionary[0])
#print('-----------------------------------------------------------')
#print(dictionary)
#print('---------------------------------------------------------------')

for _ in range(1):
    cur = []
    valid = []
    convergence = []
    dictionary = initial_dictionary.copy()
    complexityList = []
    dotList = []
    for t in range(10000):

        for _ in range(population//3):
            idx_0 = random.randint(0, 199)
            idx_1 = random.randint(0, 299)
            i , j = find_crossingidx(dictionary[ idx_0 ], dictionary[ idx_1 ], operator)
            first_children = crossing(i, j, dictionary[idx_0], dictionary[ idx_1 ])
            i2 , j2 = find_crossingidx(dictionary[ idx_1 ], dictionary[ idx_0 ], operator)
            second_children = crossing(i2, j2, dictionary[ idx_1 ], dictionary[ idx_0 ])

            candidate0 = findconstant(first_children)
            candidate1 = findconstant(second_children)
            mutated0 = mutation(first_children, candidate0, operator,t)
            mutated1 = mutation(second_children, candidate1, operator, t)
            dictionary.append(mutated0)
            dictionary.append(mutated1)
            
           
        for _ in range(100):
            fresh, fresh_equation = createheap(operator)
            dictionary.append(fresh)
        
        
        dictionary, Best_MEA, propotion, dot = rank(dictionary, y_train, x_train, operator)
        #if t in [0, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]:
        #    result = evaluate(dictionary[0].copy(), x_valid, operator)
        #    MEA_valid = fitness(y_valid, result)
        #    valid.append(MEA_valid)
        dotList.append(dot)
        cur.append((Best_MEA))
        convergence.append(propotion)
        empty = 0
        for slot in dictionary[0].copy():
            if slot == None:
                empty+=1
        complexity = len(dictionary[0].copy()) - empty
        complexityList.append(complexity)
        print('evol : {},  MEA is {},  convergence : {}'.format(t, Best_MEA, propotion))
    MEA_record.append(cur)
        #print(dictionary)
        #print(sorted_order)
        #print('------------------------------------------------------')
    np.savetxt("MAE.mat", np.array(cur), fmt="%s")

#print(MEA_record)
#print(dictionary[0])
EA_RANK = evaluate(dictionary[0].copy(), x_train, operator)
#print(MEA_record)
#print(predict_y[:5])
np.savetxt("EA_rank_final.mat", np.array(EA_RANK), fmt="%s")
np.savetxt("convergence.mat", np.array(convergence), fmt="%s")


# In[42]:


import matplotlib.pyplot as plt


# In[43]:


plt.scatter(x_train, y_train)
plt.scatter(x_train, EA_RANK)


# In[58]:


x = [i for i in range(10000)]
plt.plot(x, cur)


# In[48]:


x = []
y = []
for i in range(len(dotList)):
    for j in range(len(dotList[i])):
        x.append(i)
        y.append(dotList[i][j])


# In[55]:


plt.ylim([0, 100000])
plt.scatter(x, y, s=3.0)


# In[50]:


plt.scatter(complexityList, cur, s=10)


# In[ ]:




