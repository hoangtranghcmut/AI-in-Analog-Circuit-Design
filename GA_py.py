# genetic algorithm search for continuous function optimization
from numpy.random import randint
from numpy.random import rand
import sys
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt

# calFitness function
#def calFitness(x):
#  print("hihi\n")
#  return (x[0]-1)**2.0 + 2*(x[1]-3)**2.0 + 4*(x[2]-5)**2.0 + 4*(x[3]-5)**2.0 + 4*(x[4]-5)**2.0 

import os
import numpy as np
def calFitness(pos2,kt):
  m = 5
  n = 6
  minGain = 35
  maxError = 2
  minSlew = 10
  minVicMax = 0.6
  maxVicMin = -0.2
  um = 1e-6
  nm = 1e-9
  u = 1e-6
  p = 1e-12
  L12 = 0.5
  L34 = 1
  L56 = 0.5
  W6 = 3
  pos0 = np.array([[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1]])
  pos2 = np.array(pos2)
  sizePos = np.prod(pos2.shape)
  pos = np.full((6,5),2, dtype ='f')
  if sizePos == 5:
    pos[0,:] = pos2
  else:
    pos = pos2
  #print(pos)
  CL_a			= pos[0,0]	* p			
  CL_b			= pos[1,0] 	* p	
  CL_c			= pos[2,0] 	* p	
  CL_d			= pos[3,0]	* p	
  CL_e			= pos[4,0] 	* p	
  CL_f			= pos[5,0] 	* p	
  
  I_a			    = pos[0,1] 	* u 	
  I_b			    = pos[1,1] 	* u	
  I_c			    = pos[2,1] 	* u	
  I_d			    = pos[3,1] 	* u	
  I_e			    = pos[4,1] 	* u	
  I_f			    = pos[5,1] 	* u	
  L1_a			= L12 	    * um
  L1_b			= L12 	    * um
  L1_c			= L12 	    * um
  L1_d			= L12 	    * um
  L1_e			= L12 	    * um
  L1_f			= L12 	    * um
  L2_a			= L12 	    * um
  L2_b			= L12 	    * um
  L2_c			= L12 	    * um
  L2_d			= L12 	    * um
  L2_e			= L12 	    * um
  L2_f			= L12 	    * um
  L3_a			= L34	    * um	
  L3_b			= L34	    * um	
  L3_c			= L34	    * um	
  L3_d			= L34	    * um	
  L3_e			= L34	    * um	
  L3_f			= L34	    * um	
  L4_a			= L34	    * um	
  L4_b			= L34	    * um	
  L4_c			= L34	    * um	
  L4_d			= L34	    * um	
  L4_e			= L34	    * um	
  L4_f			= L34	    * um	
  L5_a			= L56	    * um
  L5_b			= L56	    * um
  L5_c			= L56	    * um
  L5_d			= L56	    * um
  L5_e			= L56	    * um
  L5_f			= L56	    * um
  L6_a			= L56	    * um
  L6_b			= L56	    * um
  L6_c			= L56	    * um
  L6_d			= L56	    * um
  L6_e			= L56	    * um
  L6_f			= L56	    * um
  W1_a			= pos[0,2]	* um	
  W1_b			= pos[1,2]	* um	
  W1_c			= pos[2,2]	* um	
  W1_d			= pos[3,2]	* um	
  W1_e			= pos[4,2]	* um	
  W1_f			= pos[5,2]	* um	
  W2_a			= pos[0,2]  * um	
  W2_b			= pos[1,2]  * um	
  W2_c			= pos[2,2]  * um	
  W2_d			= pos[3,2]  * um	
  W2_e			= pos[4,2]  * um	
  W2_f			= pos[5,2]  * um	
  
  W3_a			= pos[0,3]	* um	
  W3_b			= pos[1,3]	* um	
  W3_c			= pos[2,3]	* um	
  W3_d			= pos[3,3]	* um	
  W3_e			= pos[4,3]	* um	
  W3_f			= pos[5,3]	* um	
  W4_a			= pos[0,3]	* um	
  W4_b			= pos[1,3]	* um	
  W4_c			= pos[2,3]	* um	
  W4_d			= pos[3,3]	* um	
  W4_e			= pos[4,3]	* um	
  W4_f			= pos[5,3]	* um	
  
  W5_a			= pos[0,4]	* um	
  W5_b			= pos[1,4]	* um	
  W5_c			= pos[2,4]	* um	
  W5_d			= pos[3,4]	* um	
  W5_e			= pos[4,4]	* um	
  W5_f			= pos[5,4]	* um
  	
  W6_a			= W6	* um	#pos[0,4]/2	* um	
  W6_b			= W6	* um	
  W6_c			= W6    * um	
  W6_d			= W6	* um	
  W6_e			= W6	* um	
  W6_f			= W6	* um	
  ac_vminus	    = 0	
  ac_vplus		= 1e-3	
  dc_vplus		= 0	
  tran_vminus	    = -0.4
  tran_vplus	    = 0.4	
  vdd			    = 1.2	
  vss			    = -1.2
  
  # xuat vo ocean
  filethongso = "/mnt/hgfs/shared/GA_thongso.txt"
  f = open(filethongso,'w')
  f.write("\n%s" %CL_a		)
  f.write("\n%s" %CL_b		)
  f.write("\n%s" %CL_c		)
  f.write("\n%s" %CL_d		)
  f.write("\n%s" %CL_e		)
  f.write("\n%s" %CL_f		)
  f.write("\n%s" %I_a			)
  f.write("\n%s" %I_b			)
  f.write("\n%s" %I_c			)
  f.write("\n%s" %I_d			)
  f.write("\n%s" %I_e			)
  f.write("\n%s" %I_f			)
  f.write("\n%s" %L1_a		)
  f.write("\n%s" %L1_b		)
  f.write("\n%s" %L1_c		)
  f.write("\n%s" %L1_d		)
  f.write("\n%s" %L1_e		)
  f.write("\n%s" %L1_f		)
  f.write("\n%s" %L2_a		)
  f.write("\n%s" %L2_b		)
  f.write("\n%s" %L2_c		)
  f.write("\n%s" %L2_d		)
  f.write("\n%s" %L2_e		)
  f.write("\n%s" %L2_f		)
  f.write("\n%s" %L3_a		)
  f.write("\n%s" %L3_b		)
  f.write("\n%s" %L3_c		)
  f.write("\n%s" %L3_d		)
  f.write("\n%s" %L3_e		)
  f.write("\n%s" %L3_f		)
  f.write("\n%s" %L4_a		)
  f.write("\n%s" %L4_b		)
  f.write("\n%s" %L4_c		)
  f.write("\n%s" %L4_d		)
  f.write("\n%s" %L4_e		)
  f.write("\n%s" %L4_f		)
  f.write("\n%s" %L5_a		)
  f.write("\n%s" %L5_b		)
  f.write("\n%s" %L5_c		)
  f.write("\n%s" %L5_d		)
  f.write("\n%s" %L5_e		)
  f.write("\n%s" %L5_f		)
  f.write("\n%s" %L6_a		)
  f.write("\n%s" %L6_b		)
  f.write("\n%s" %L6_c		)
  f.write("\n%s" %L6_d		)
  f.write("\n%s" %L6_e		)
  f.write("\n%s" %L6_f		)
  f.write("\n%s" %W1_a		)
  f.write("\n%s" %W1_b		)
  f.write("\n%s" %W1_c		)
  f.write("\n%s" %W1_d		)
  f.write("\n%s" %W1_e		)
  f.write("\n%s" %W1_f		)
  f.write("\n%s" %W2_a		)
  f.write("\n%s" %W2_b		)
  f.write("\n%s" %W2_c		)
  f.write("\n%s" %W2_d		)
  f.write("\n%s" %W2_e		)
  f.write("\n%s" %W2_f		)
  f.write("\n%s" %W3_a		)
  f.write("\n%s" %W3_b		)
  f.write("\n%s" %W3_c		)
  f.write("\n%s" %W3_d		)
  f.write("\n%s" %W3_e		)
  f.write("\n%s" %W3_f		)
  f.write("\n%s" %W4_a		)
  f.write("\n%s" %W4_b		)
  f.write("\n%s" %W4_c		)
  f.write("\n%s" %W4_d		)
  f.write("\n%s" %W4_e		)
  f.write("\n%s" %W4_f		)
  f.write("\n%s" %W5_a		)
  f.write("\n%s" %W5_b		)
  f.write("\n%s" %W5_c		)
  f.write("\n%s" %W5_d		)
  f.write("\n%s" %W5_e		)
  f.write("\n%s" %W5_f		)
  f.write("\n%s" %W6_a		)
  f.write("\n%s" %W6_b		)
  f.write("\n%s" %W6_c		)
  f.write("\n%s" %W6_d		)
  f.write("\n%s" %W6_e		)
  f.write("\n%s" %W6_f		)
  f.write("\n%s" %ac_vminus	)
  f.write("\n%s" %ac_vplus	)
  f.write("\n%s" %dc_vplus	)
  f.write("\n%s" %tran_vminus	)
  f.write("\n%s" %tran_vplus	)
  f.write("\n%s" %vdd			)
  f.write("\n%s" %vss			)
  
  f.close()
  
  os.system("ocean -nograph -restore GA_ocn.ocn")
  ketqua = np.array([])
  with open("/mnt/hgfs/shared/GA_ketqua.txt") as f:
    ketquara = f.readlines()
  ketquara = np.array(ketquara)
  ketqua = ketquara.astype(np.float)
  ketqua= ketqua.reshape(n,m)
  
  minGain = np.full((n,1),35)
  maxError = np.full((n,1),2)
  minSlew = np.full((n,1),8)
  minVicMax = np.full((n,1),0.6)
  maxVicMin = np.full((n,1),-0.2)
  
  expectGain = 41
  expectError = 1 #1%
  expectSlew = 16
  expectVicMax = 0.9
  expectVicMin = -0.4
  
  merGain   = np.transpose(ketqua[:,0]).reshape(-1,1)
  merError  = np.transpose(ketqua[:,1]).reshape(-1,1)
  merSlew   = np.transpose(ketqua[:,2]).reshape(-1,1)
  merVicMax = np.transpose(ketqua[:,3]).reshape(-1,1)
  merVicMin = np.transpose(ketqua[:,4]).reshape(-1,1)
  
  hesoGain   = 4
  hesoError  = 2
  hesoSlew   = 3
  hesoVicMax = 1
  hesoVicMin = 4
  
  #xu ly de ra fitness
  fitness = np.zeros((n,1), dtype='f')
  fitness[:,] = (hesoGain*(merGain-expectGain)**2 + hesoError*(merError-expectError)**2 + hesoGain*(merSlew-expectSlew)**2 + hesoVicMax*(merVicMax-expectVicMax)**2 + hesoVicMin*(merVicMin-expectVicMin)**2).reshape(-1,1)
    

  #kiem tra co du tieu chuan chua
  #fitness[:,] = ((merGain >= minGain)*fitness + (merGain < minGain)*999         )
  #fitness[:,] = ((merError <= maxError)*fitness + (merError > maxError)*999     )
  #fitness[:,] = ((merSlew >= minSlew)*fitness + (merSlew < minSlew)*999         )
  #fitness[:,] = ((merVicMax >= minVicMax)*fitness + (merVicMax < minVicMax)*999 )
  #fitness[:,] = ((merVicMin <= maxVicMin)*fitness + (merVicMin > maxVicMin)*999 )
  #return
  
  fitness =  fitness.flatten()
  if kt==1:
    fitness2 = fitness[0]
  else:
    fitness2 = fitness.tolist()
  return fitness2, ketqua

# decode bitstring to numbers
def decode(bounds, n_bits, bitstring):
  decoded = list()
  largest = 2**n_bits
  for i in range(len(bounds)):
    # extract the substring
    start, end = i * n_bits, (i * n_bits)+n_bits
    substring = bitstring[start:end]
    # convert bitstring to a string of chars
    chars = ''.join([str(s) for s in substring])
    # convert string to integer
    integer = int(chars, 2)
    # scale integer to desired range
    value = bounds[i][0] + (integer/largest) * (bounds[i][1] - bounds[i][0])
    # store
    decoded.append(value)
  return decoded
 
# tournament selection
def selection(pop, scores, k=3):
  # first random selection
  selection_ix = randint(len(pop))
  for ix in randint(0, len(pop), k-1):
    # check if better (e.g. perform a tournament)
    if scores[ix] < scores[selection_ix]:
      selection_ix = ix
  return pop[selection_ix]
 
# crossover two parents to create two children
def crossover(p1, p2, r_cross):
  # children are copies of parents by default
  c1, c2 = p1.copy(), p2.copy()
  # check for recombination
  if rand() < r_cross:
    # select crossover point that is not on the end of the string
    pt = randint(1, len(p1)-2)
    # perform crossover
    c1 = p1[:pt] + p2[pt:]
    c2 = p2[:pt] + p1[pt:]
  return [c1, c2]
 
# mutation operator
def mutation(bitstring, r_mut):
  for i in range(len(bitstring)):
    # check for a mutation
    if rand() < r_mut:
      # flip the bit
      bitstring[i] = 1 - bitstring[i]
 
# genetic algorithm
def genetic_algorithm(calFitness, bounds, n_bits, n_iter, n_pop, r_cross, r_mut,df):
  # initial population of random bitstring
  pop = [randint(0, 2, n_bits*len(bounds)).tolist() for _ in range(n_pop)]
  print(decode(bounds, n_bits, pop[0]))
  # keep track of best solution
  best = 0
  best_eval, ketqua = calFitness(decode(bounds, n_bits, pop[0]),1)
  
  # enumerate generations
  for gen in range(n_iter):
    begin_time = dt.datetime.now().strftime("%H:%M:%S")
    # decode population
    decoded = [decode(bounds, n_bits, p) for p in pop]
    # evaluate all candidates in the population
    scores,ketqua = calFitness(decoded,0)
    #print("decoded\n",decoded)
    #print("scores\n",scores)
    # check for new best solution
    for i in range(n_pop):
      #print("best_eval",best_eval)
      if scores[i] < best_eval:
        best, best_eval= pop[i], scores[i]
        #print(">%d, new best f(%s) = %f" % (gen,  decoded[i], scores[i]))
    # select parents
    selected = [selection(pop, scores) for _ in range(n_pop)]
    # create the next generation
    children = list()
    for i in range(0, n_pop, 2):
      # get selected parents in pairs
      p1, p2 = selected[i], selected[i+1]
      # crossover and mutation
      for c in crossover(p1, p2, r_cross):
        # mutation
        mutation(c, r_mut)
        # store for next generation
        children.append(c)
    # replace population
    pop = children
    if gen==0:
      xuatketqua = ketqua[np.argmin(scores)] 
    else:
      xuatketqua = (np.amin(scores)<=best_eval)*ketqua[np.argmin(scores)] + (np.amin(scores)>best_eval)*xuatketqua
    print("xuatketqua",xuatketqua)
    gbest_pos = decode(bounds, n_bits, best)
    new_row =[
    {
      "Lan chay": gen,
      "Begin": begin_time,
      "End": dt.datetime.now().strftime("%H:%M:%S"),
      "Time (s)": dt.datetime.strptime(dt.datetime.now().strftime("%H:%M:%S"), '%H:%M:%S') - dt.datetime.strptime(begin_time, '%H:%M:%S'),
      "Fitness": best_eval,
      "CL (p)": gbest_pos[0],
      "Id (u)" : gbest_pos[1],
      "W1 (um)": gbest_pos[2],
      "W3 (um)": gbest_pos[3],
      "W5 (um)": gbest_pos[4],
      "Open Loop Gain (dB)": xuatketqua[0],
      "Error (%)": xuatketqua[1],
      "Slew rate (V/us)": xuatketqua[2],
      "vIcMax (V)": xuatketqua[3],
      "vIcMin (V)": xuatketqua[4],
    }]
    df = df.append(new_row, ignore_index=True)
    print(df)
  return [best, best_eval,df]
 
# define range for input
bounds = [[1, 10.0], [0.1, 40], [0.2, 10], [0.2, 10], [0.2, 10]]
#thiet lap
um = 1e-6
nm = 1e-9
u = 1e-6
p = 1e-12

m = 5  #so bien
n = 6  #swarm size
wMax = 0.9
wMin = 0.4
c1 = 0.3
c2 = 0.9


# define the total iterations
n_iter = int(sys.argv[1])
# bits per variable
n_bits = 16
# define the population size
n_pop = 6
# crossover rate
r_cross = 0.9
# mutation rate
r_mut = 1.0 / (float(n_bits) * len(bounds))
column_names = ["Lan chay","Begin","End","Time (s)","Fitness","CL (p)","Id (u)","W1 (um)","W3 (um)","W5 (um)","Open Loop Gain (dB)","Error (%)","Slew rate (V/us)","vIcMax (V)","vIcMin (V)"]
df = pd.DataFrame(columns = column_names)
# perform the genetic algorithm search
best, score,df = genetic_algorithm(calFitness, bounds, n_bits, n_iter, n_pop, r_cross, r_mut,df)
print('Done!')
print(df)
decoded = decode(bounds, n_bits, best)
print('f(%s) = %f' % (decoded, score))
df.to_csv("/mnt/hgfs/shared/ketqua4GA.csv")
df.to_excel("/mnt/hgfs/shared/ketqua4GA.xlsx", sheet_name="GA")
#os.system("ocean -restore final.ocn")
calFitness(decoded,0)
os.system("ocean -restore GA_final.ocn")
