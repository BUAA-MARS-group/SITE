# %% import modules
import geppy as gep
from deap import creator, base, tools
import numpy as np
import random
import operator 
import time
import scipy.io as scio
import sympy as sp

# For reproduction
s = 0
random.seed(s)
np.random.seed(s)

import SITE
from simplification import p_symbol, linker_add, protected_div_symbol


# %% define functions for tensor operators 
def tensor_add(*args):
    args = [arg for arg in args]
    sum = 0
    for arg in args:
        sum += arg
    return sum

def tensor_sub(a, b):
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.shape == b.shape:
        return a - b
    else:
        return False

def tensor_inner_product(a, b):
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.shape == b.shape:
        result = np.empty_like(a)
        for i in range(a.shape[0]):
            result[i] = np.matmul(a[i], b[i])
        return result
    else:
        return False
        
# define a protected division to avoid dividing by zero
def protected_div(x1, x2):
    if isinstance(x2, np.ndarray):
        abs_x2 = np.maximum(x2,-x2)
        if (abs_x2 < 1e-10).any():
            return 0
        return x1 / x2
    else:
        if abs(x2) < 1e-10:
            return 0
        return x1 / x2
    
def transpose(tensor):
    if isinstance(tensor, np.ndarray):
        transposed = np.empty_like(tensor)
        for i in range(tensor.shape[0]):
            transposed[i] = tensor[i].T
        return transposed
    else:
        return False

def trace_free(tensor):
    if isinstance(tensor, np.ndarray):
        trace = np.trace(tensor, axis1=1, axis2=2) / tensor.shape[1]
        trace_matrix = np.zeros_like(tensor)
        for i in range(tensor.shape[0]):
            trace_matrix[i] = np.eye(tensor.shape[1]) * trace[i]
        return tensor - trace_matrix
    else:
        return False

    
# a placeholder function
def p_(tensor):
    pass

# %%
data = scio.loadmat('data/processed_data.mat')
op_data = np.load('data/output_data.npy')
x = data['x']
y = data['y']
z = data['z']
t = data['t']
u = data['u']
v = data['v']
w = data['w']
R11 = data['R11']
R12 = data['R12']
R13 = data['R13']
R22 = data['R22']
R23 = data['R23']
R33 = data['R33']
epsilon = data['epsilon']
k = data['k']
nut = data['nut']
rho = data['rho']
ddt = op_data[:,4:10]
div = op_data[:,10:16]
gradU = op_data[:,16:25]
laplacian = op_data[:,25:31]
num_points = R11.shape[0]
dim = 3

# tensors
def create_tensor(c11, c12, c13, c21, c22, c23, c31, c32, c33):
    tensor = np.zeros((num_points, dim, dim))
    tensor[:, 0, 0] = c11
    tensor[:, 0, 1] = c12
    tensor[:, 0, 2] = c13
    tensor[:, 1, 0] = c21
    tensor[:, 1, 1] = c22
    tensor[:, 1, 2] = c23
    tensor[:, 2, 0] = c31
    tensor[:, 2, 1] = c32
    tensor[:, 2, 2] = c33
    return tensor
rR_t = create_tensor(ddt[:, 0], ddt[:, 1], ddt[:, 2],
                     ddt[:, 1], ddt[:, 3], ddt[:, 4],
                     ddt[:, 2], ddt[:, 4], ddt[:, 5])
div_ij = create_tensor(div[:, 0], div[:, 1], div[:, 2],
                       div[:, 1], div[:, 3], div[:, 4],
                       div[:, 2], div[:, 4], div[:, 5])
u_x_ij = create_tensor(gradU[:, 0], gradU[:, 3], gradU[:, 6],
                       gradU[:, 1], gradU[:, 4], gradU[:, 7],
                       gradU[:, 2], gradU[:, 5], gradU[:, 8])
R_ij = create_tensor(R11[:,0], R12[:,0], R13[:,0],
                     R12[:,0], R22[:,0], R23[:,0],
                     R13[:,0], R23[:,0], R33[:,0])
D_ij = create_tensor(laplacian[:, 0], laplacian[:, 1], laplacian[:, 2],
                     laplacian[:, 1], laplacian[:, 3], laplacian[:, 4],
                     laplacian[:, 2], laplacian[:, 4], laplacian[:, 5])

# Left: material derivative
DR_t = rR_t + div_ij

# subsample
data_size = 200
indices = np.random.choice(R_ij.shape[0], data_size, replace=False)

u_x_ij = u_x_ij[indices]
u_x_ji = transpose(u_x_ij)
R_ij = R_ij[indices]
D_ij = D_ij[indices]
DR_t = DR_t[indices]
epsilon = epsilon[indices]
k = k[indices]
rho = rho[indices]

delta_ij = np.zeros((R_ij.shape[0], dim, dim))                        
delta_ij[:, 0, 0] = 1
delta_ij[:, 1, 1] = 1
delta_ij[:, 2, 2] = 1

Y = DR_t
P_ij = - tensor_inner_product(R_ij, u_x_ji) - tensor_inner_product(u_x_ij, R_ij)
Y = rho[:,None] * P_ij - 1.8 * rho[:,None] * (epsilon[:,None]/k[:,None])*trace_free(R_ij) - 0.6 * rho[:,None] * trace_free(P_ij) - (2/3)* rho[:,None] * epsilon[:,None]*delta_ij + D_ij

# %%
dict_of_variables = {'R_ij':R_ij,
                     'u_x_ij':u_x_ij,
                     'D_ij':D_ij,
                    'delta_ij':delta_ij,
                    'epsilon':epsilon,
                    'rho':rho,
                    'k':k,}

symbolic_function_map = {
        'tensor_add': linker_add,
        'tensor_sub': operator.sub,
        'tensor_inner_product': operator.mul,
        'transpose': sp.transpose,
        'trace_free': sp.sin,   # Using log as a placeholder for trace_free
        'p_': p_symbol,
        operator.add.__name__: operator.add,
        operator.sub.__name__: operator.sub,
        operator.mul.__name__: operator.mul,
        'protected_div': protected_div_symbol,
    }

dict_of_operators = {'tensor_add':tensor_add,
                    'tensor_sub':tensor_sub,
                    'tensor_inner_product':tensor_inner_product,
                    'transpose':transpose,
                    'trace_free':trace_free,
                    operator.add.__name__: operator.add,
                    operator.sub.__name__: operator.sub,
                    operator.mul.__name__: operator.mul,
                    'protected_div':protected_div,
                    'linker_add':linker_add}

# [M, L, T]
dict_of_dimension = {'R_ij':[0, 2, -2],
                     'u_x_ij':[0, 0, -1],
                     'D_ij':[1, -1, -3],
                    'delta_ij':[0, 0, 0],
                    'rho':[1, -3, 0],
                    'epsilon':[0, 2,-3],
                    'k':[0, 2, -2],} 


NUM_UNITS = 3  # length of dimension list

target_dimension = [1,-1,-3]

# %% Creating the primitives set
# Define the operators
host_pset = gep.PrimitiveSet('Host', input_names=['R_ij', 'u_x_ij', 'D_ij', 'delta_ij'])
host_pset.add_function(tensor_add, 2)
host_pset.add_function(tensor_sub, 2)
host_pset.add_function(tensor_inner_product, 2)
host_pset.add_function(transpose, 1)
host_pset.add_function(trace_free, 1)
host_pset.add_function(p_, 1)


plasmid_pset = gep.PrimitiveSet('Plasmid', input_names=['epsilon','k','rho'])
plasmid_pset.add_function(operator.add, 2)
plasmid_pset.add_function(operator.sub, 2)
plasmid_pset.add_function(operator.mul, 2)
plasmid_pset.add_function(protected_div, 2)

# %% Create the individual and population
# Define the indiviudal class, a subclass of gep.Chromosome
creator.create("FitnessMax", base.Fitness, weights=(-1,))  # weights=(-1,)/weights=(1,) means to minimize/maximize the objective (fitness).
creator.create("Host_Individual", gep.Chromosome, fitness=creator.FitnessMax, plasmid=[])
creator.create("Plasmid_Individual", gep.Chromosome) 

# Register the individual and population creation operations
h_host = 5            # head length
h_plasmid = 10
n_genes_host = 6  # number of genes in a chromosome for hosts
n_genes_plasmid = 1      # number of genes in a chromosome for plasmids

toolbox = gep.Toolbox()

toolbox.register('host_gene_gen', gep.Gene, pset=host_pset, head_length=h_host)
toolbox.register('host_individual', creator.Host_Individual, gene_gen=toolbox.host_gene_gen, n_genes=n_genes_host, linker=tensor_add)
toolbox.register("host_population", tools.initRepeat, list, toolbox.host_individual)

creator.create("tinder", gep.Chromosome, plasmid=[])
toolbox.register('tinder_individual', creator.tinder, gene_gen=toolbox.host_gene_gen, n_genes=1, linker=tensor_add)

toolbox.register('plasmid_gene_gen', gep.Gene, 
                 pset=plasmid_pset, head_length=h_plasmid)
toolbox.register('plasmid_individual', creator.Plasmid_Individual, 
                 gene_gen=toolbox.plasmid_gene_gen, 
                 n_genes=n_genes_plasmid, 
                 linker=linker_add)
toolbox.register("plasmid_population", SITE.plasmid_generate, 
                 toolbox.plasmid_individual)
     
toolbox.register('compile', SITE.my_compile, 
                 dict_of_operators = dict_of_operators, 
                 symbolic_function_map = symbolic_function_map, 
                 dict_of_variables = dict_of_variables,
                 Y = Y)
toolbox.register('dimensional_verification', SITE.dimensional_verification, 
                 dict_of_dimension = dict_of_dimension, 
                 num_units = NUM_UNITS, 
                 target_dimension = target_dimension)
toolbox.register('evaluate', SITE.evaluate, 
                 tb = toolbox, 
                 dict_of_operators = dict_of_operators,
                 dict_of_variables = dict_of_variables, 
                 Y = Y)

# %% Register genetic operators
toolbox.register('select', tools.selTournament, tournsize=200) # Selection operator
# 1. general operators for host population
toolbox.register('mut_uniform', SITE.mutate_uniform, host_pset = host_pset, 
                 func = toolbox.plasmid_individual, ind_pb=0.2, pb=1)
toolbox.register('mut_invert', SITE.invert, pb=0.2)
toolbox.register('mut_is_transpose', SITE.is_transpose, pb=0.2)
toolbox.register('mut_ris_transpose', SITE.ris_transpose, pb=0.2)
toolbox.register('mut_gene_transpose', SITE.gene_transpose, pb=0.2)
toolbox.register('cx_1p', SITE.crossover_one_point, pb=0.2)
toolbox.register('cx_2p', SITE.crossover_two_point, pb=0.2)
toolbox.register('cx_gene', SITE.crossover_gene, pb=0.2)
# 2. general operators for plasmid population
toolbox.register('mut_uniform_plasmid', gep.mutate_uniform, pset = plasmid_pset, ind_pb=0.05, pb=1)
toolbox.register('mut_invert_plasmid', gep.invert, pb=0.1)
toolbox.register('mut_is_transpose_plasmid', gep.is_transpose, pb=0.1)
toolbox.register('mut_ris_transpose_plasmid', gep.ris_transpose, pb=0.1)
toolbox.register('mut_gene_transpose_plasmid', gep.gene_transpose, pb=0.1)

# %% Statistics to be inspected
stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
stats.register("avg", np.mean)
# stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

# %% Launch evolution

# Define size of population and number of generations
n_pop = 1600             # Number of individuals in a host_population
n_gen = 3000            # Maximum Generation
tol = 1e-6               # Threshold to terminate the evolution
output_type = f'RST_no_Pij'     # Name of the problem

host_pop = toolbox.host_population(n=n_pop) 
plasmid_pop = toolbox.plasmid_population(host_pop)
for ind_host, ind_plasmid in zip(host_pop, plasmid_pop):
    ind_host.plasmid = ind_plasmid 

# Only record the best three individuals ever found in all generations
champs = 3 
hof = tools.HallOfFame(champs)   


# %%
# Evolve
start_time = time.time()
pop, log = SITE.gep_simple(host_pop, plasmid_pop, toolbox, n_generations=n_gen, n_elites=1, n_alien_inds=100,
                          stats=stats, hall_of_fame=hof, verbose=True,tolerance = tol,GEP_type = output_type)
print(time.time() - start_time)