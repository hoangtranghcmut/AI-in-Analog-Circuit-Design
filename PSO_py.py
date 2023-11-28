import numpy as np
import os
import csv
import PSO1_fitnessRun
import sys
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt

#thiet lap
um = 1e-6
nm = 1e-9
u = 1e-6
p = 1e-12

m = 5	#so bien
n = 6	#swarm size
wMax = 0.9
wMin = 0.4
c1 = 0.3
c2 = 0.9
lap = int(sys.argv[1])



#sys.stdout("ls")
# gioi han
LB = np.array([1, 5, 0.2, 0.2, 0.2] , dtype = 'f')   #CL, I, W1, W3, W5
UB = np.array([5, 40, 10, 10,  10 ] , dtype = 'f')
minGain = 35
maxError = 2
minSlew = 10
minVicMax = 0.6
maxVicMin = -0.2

pos = np.zeros((n,m), dtype='f')

# khoi tao gia tri dau cho cac bien
# khoi tao POS
# tao 5 bay, moi bay 3 bien
##QUAN TRONG
#for bay in range(n):    # moi con tu nhan ban thanh 5 con la 5 bay
  # for con in range(m)  # moi bay co 3 con khac nhau tuc la 3 bien
  #pos[bay, :] = LB + np.random.rand(1,m)*(UB - LB)
pos[0, :] = np.array([1, 10, 1, 1, 1.0])    #CL, I, W1, W3, W5
pos[1, :] = np.array([1, 1, 5, 1, 1.0])    #CL, I, W1, W3, W5
pos[2, :] = np.array([5, 2, 4, 1, 1.0])    #CL, I, W1, W3, W5
pos[3, :] = np.array([2, 10, 5, 1, 1.0])    #CL, I, W1, W3, W5
pos[4, :] = np.array([1, 5, 5, 1, 6.0])    #CL, I, W1, W3, W5
pos[5, :] = np.array([1, 10, 1, 1, 4.0])    #CL, I, W1, W3, W5
# khoi tao van toc VEL
#pos = LB + np.random.rand(n,m)*(UB - LB)
vel = 0.1*pos
# khoi tao FITNESS
fitness = np.zeros((n,1), dtype='f')
pbest_val = np.full((n, 1), 999999999, dtype ='f')
pbest_pos = np.zeros((n,m), dtype='f')
gbest_val = np.full((1,1), 999999999, dtype='f')
gbest_pos = np.zeros((1,m), dtype='f')
#print(fitness[:,])
print(pos)
#print(pbest_val)
#print(vel)

#gbest_pos_save = np.empty_like(gbest_pos)
#gbest_val_save = np.array([1])

column_names = ["Lan chay","Begin","End","Time (s)","Fitness","CL (p)","Id (u)","W1 (um)","W3 (um)","W5 (um)","Open Loop Gain (dB)","Error (%)","Slew rate (V/us)","vIcMax (V)","vIcMin (V)"]
df = pd.DataFrame(columns = column_names)



for chay in range(lap):
  #bat dau lap
  #luu start time
  begin_time = dt.datetime.now().strftime("%H:%M:%S")
  #thay doi w
  w = wMax - (chay/lap)*(wMax-wMin)

  # tinh gia tri voi moi bo so
  fitness, ketqua = PSO1_fitnessRun.calFitness(pos)
  
  #cap nhat pbest
  pbest_pos = (fitness <= pbest_val)*pos + (fitness > pbest_val)*pbest_pos
  pbest_val = (fitness <= pbest_val)*fitness + (fitness > pbest_val)*pbest_val
  
  #cap nhat gbest
  gbest_pos = pbest_pos[np.argmin(pbest_val)]
  gbest_val = pbest_val[np.argmin(pbest_val)]
  
  #cap nhat vel, pos
  vel = w*vel+ c1*np.random.rand(n,m)*(pbest_pos - pos) + c2*np.random.rand(n,m)*(gbest_pos - pos)
  pos = pos + vel

  #kiem tra dieu kien bien
  pos = (pos >= LB )*pos + (pos < LB)*LB 
  pos = (pos <= UB )*pos + (pos > UB)*UB

  #neu gbest duoc cap nhat moi thi ket qua la gia tri moi, con khong thi thoi
  if chay==0:
    xuatketqua = ketqua[np.argmin(fitness)] 
  else:
     xuatketqua = (np.amin(fitness)<=gbest_val[0])*ketqua[np.argmin(fitness)] + (np.amin(fitness)>gbest_val[0])*xuatketqua
  #luu vao bang ghi
  new_row =[
    {
      "Lan chay": chay,
      "Begin": begin_time,
      "End": dt.datetime.now().strftime("%H:%M:%S"),
      "Time (s)": dt.datetime.strptime(dt.datetime.now().strftime("%H:%M:%S"), '%H:%M:%S') - dt.datetime.strptime(begin_time, '%H:%M:%S'),
      "Fitness": gbest_val[0],
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

df.to_csv("/mnt/hgfs/shared/PSO1.csv")
df.to_excel("/mnt/hgfs/shared/PSO1.xlsx", sheet_name="PSO")
#xuat ra
pos[0, :] = np.transpose(gbest_pos)   #CL, I, W1, W3, W5
os.system("ocean -restore PSO1_final.ocn")
