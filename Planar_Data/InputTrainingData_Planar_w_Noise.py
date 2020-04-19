import numpy as np
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
plt.ion()
ax = plt.axes(projection="3d")


#This program implements a single neuron of a neural net.  The structure is 3 inputs to a single node to a single output.
#Code is based on example presented at: https://urldefense.proofpoint.com/v2/url?u=https-3A__medium.com_themlblog_a-2Dsimple-2Dneural-2Dnetwork-2Dwith-2Da-2Dsingle-2Dneuron-2D8a4e3b0a4148&d=DwIGAg&c=c6MrceVCY5m5A_KAUkrdoA&r=MqWbypA6xJWFEpQLXyroUCv17OsLpWNnvZ1UzmavYE8&m=v7kMgQh71qyhs8GJRQBuFBkdH0YZzQ7h7s0L126B7qY&s=xv5Z06ECMGNc33BZsJIbrCeH7reJ_obzCaT3dfqEYl8&e= 
#    x1
#     \
#      \
#        --->  |^^^|
#   x2 ----->  |   |------> Output
#        --->  |___|
#       /
#      /
#    x3
#
# 
# %Actually, make the data planar data with noise
#Data was created in Matlab as follows:
#A = 0.3; B=0.4; C=0.5; nnoise=randn(1,100);max(nnoise),min(nnoise)
#xx = rand(1,100); y = rand(1,100);
#[max(xx),min(xx),max(yy), min(yy)]
#xx = rand(1,100); yy = rand(1,100);
#[max(xx),min(xx),max(yy), min(yy)]
#xx=10*xx;yy=10*yy;
#zz = A + B*xx+C*yy + 0.01*nnoise;
#plot3(xx,yy,zz)
#plot3(xx,yy,zz,'.')



train_inputs = np.array([[ 1, 6.819719 , 0.942293  ], 

[ 1, 0.424311 , 5.985237  ], 

[ 1, 0.714455 , 4.709243  ], 

[ 1, 5.216498 , 6.959493  ], 

[ 1, 0.967300 , 6.998878  ], 

[ 1, 8.181486 , 6.385308  ], 

[ 1, 8.175471 , 0.336038  ], 

[ 1, 7.224396 , 0.688061  ], 

[ 1, 1.498654 , 3.195997  ], 

[ 1, 6.596053 , 5.308643  ], 

[ 1, 5.185949 , 6.544457  ], 

[ 1, 9.729746 , 4.076192  ], 

[ 1, 6.489915 , 8.199812  ], 

[ 1, 8.003306 , 7.183589  ], 

[ 1, 4.537977 , 9.686493  ], 

[ 1, 4.323915 , 5.313339  ], 

[ 1, 8.253138 , 3.251457  ], 

[ 1, 0.834698 , 1.056292  ], 

[ 1, 1.331710 , 6.109587  ], 

[ 1, 1.733886 , 7.788022  ], 

[ 1, 3.909378 , 4.234529  ], 

[ 1, 8.313797 , 0.908233  ], 

[ 1, 8.033644 , 2.664715  ], 

[ 1, 0.604712 , 1.536567  ], 

[ 1, 3.992578 , 2.810053  ], 

[ 1, 5.268758 , 4.400851  ], 

[ 1, 4.167995 , 5.271427  ], 

[ 1, 6.568599 , 4.574244  ], 

[ 1, 6.279734 , 8.753716  ], 

[ 1, 2.919841 , 5.180521  ], 

[ 1, 4.316512 , 9.436226  ], 

[ 1, 0.154871 , 6.377091  ], 

[ 1, 9.840637 , 9.576939  ], 

[ 1, 1.671684 , 2.407070  ], 

[ 1, 1.062163 , 6.761223  ], 

[ 1, 3.724097 , 2.890646  ], 

[ 1, 1.981184 , 6.718082  ], 

[ 1, 4.896876 , 6.951405  ], 

[ 1, 3.394934 , 0.679928  ], 

[ 1, 9.516305 , 2.547902  ], 

[ 1, 9.203320 , 2.240400  ], 

[ 1, 0.526770 , 6.678327  ], 

[ 1, 7.378581 , 8.443922  ], 

[ 1, 2.691194 , 3.444624  ], 

[ 1, 4.228356 , 7.805197  ], 

[ 1, 5.478709 , 6.753321  ], 

[ 1, 9.427370 , 0.067153  ], 

[ 1, 4.177441 , 6.021705  ], 

[ 1, 9.830525 , 3.867712  ], 

[ 1, 3.014549 , 9.159912  ], 

[ 1, 7.010988 , 0.011511  ], 

[ 1, 6.663389 , 4.624492  ], 

[ 1, 5.391265 , 4.243490  ], 

[ 1, 6.981055 , 4.609164  ], 

[ 1, 6.665279 , 7.701597  ], 

[ 1, 1.781325 , 3.224718  ], 

[ 1, 1.280144 , 7.847393  ], 

[ 1, 9.990804 , 4.713572  ], 

[ 1, 1.711211 , 0.357627  ], 

[ 1, 0.326008 , 1.758744  ], 

[ 1, 5.611998 , 7.217580  ], 

[ 1, 8.818665 , 4.734860  ], 

[ 1, 6.691753 , 1.527212  ], 

[ 1, 1.904333 , 3.411246  ], 

[ 1, 3.689165 , 6.073892  ], 

[ 1, 4.607259 , 1.917453  ], 

[ 1, 9.816380 , 7.384268  ], 

[ 1, 1.564050 , 2.428496  ], 

[ 1, 8.555228 , 9.174243  ], 

[ 1, 6.447645 , 2.690616  ], 

[ 1, 3.762722 , 7.655000  ], 

[ 1, 1.909237 , 1.886620  ], 

[ 1, 4.282530 , 2.874982  ], 

[ 1, 4.820221 , 0.911135  ], 

[ 1, 1.206116 , 5.762094  ], 

[ 1, 5.895075 , 6.833632  ], 

[ 1, 2.261877 , 5.465931  ], 

[ 1, 3.846191 , 4.257288  ], 

[ 1, 5.829864 , 6.444428  ], 

[ 1, 2.518061 , 6.476176  ], 

[ 1, 2.904407 , 6.790168  ], 

[ 1, 6.170909 , 6.357867  ], 

[ 1, 2.652809 , 9.451741  ], 

[ 1, 8.243763 , 2.089349  ], 

[ 1, 9.826634 , 7.092817  ], 

[ 1, 7.302488 , 2.362306  ], 

[ 1, 3.438770 , 1.193962  ], 

[ 1, 5.840693 , 6.073039  ], 

[ 1, 1.077690 , 4.501377  ], 

[ 1, 9.063082 , 4.587255  ], 

[ 1, 8.796537 , 6.619448  ], 

[ 1, 8.177606 , 7.702855  ], 

[ 1, 2.607280 , 3.502180  ], 

[ 1, 5.943563 , 6.620096  ], 

[ 1, 0.225126 , 4.161586  ], 

[ 1, 4.252593 , 8.419292  ], 

[ 1, 3.127189 , 8.329168  ], 

[ 1, 1.614847 , 2.564410  ], 

[ 1, 1.787662 , 6.134607  ], 

[ 1, 4.228857 , 5.822492  ]])

train_outputs = np.array([[3.504411   , 
 3.480682   , 
 2.917815   , 
 5.874968   , 
 4.189547   , 
 6.752171   , 
 3.733872   , 
 3.537215   , 
 2.533244   , 
 5.620437   , 
 5.633109   , 
 6.260343   , 
 7.003126   , 
 7.092486   , 
 6.965585   , 
 4.684186   , 
 5.225742   , 
 1.176922   , 
 3.901568   , 
 4.901738   , 
 3.987731   , 
 4.067561   , 
 4.852987   , 
 1.326471   , 
 3.306947   , 
 4.618276   , 
 4.610180   , 
 5.211527   , 
 7.191690   , 
 4.050324   , 
 6.753602   , 
 3.539023   , 
 9.014036   , 
 2.164114   , 
 4.076034   , 
 3.249346   , 
 4.454766   , 
 5.726904   , 
 2.011640   , 
 5.363357   , 
 5.100506   , 
 3.847457   , 
 7.476585   , 
 3.101918   , 
 5.885292   , 
 5.867843   , 
 4.102876   , 
 4.988106   , 
 6.176998   , 
 6.096869   , 
 3.101514   , 
 5.278375   , 
 4.566110   , 
 5.385869   , 
 6.816842   , 
 2.640215   , 
 4.728057   , 
 6.656821   , 
 1.161042   , 
 1.320949   , 
 6.142699   , 
 6.195222   , 
 3.745832   , 
 2.778362   , 
 4.828054   , 
 3.102489   , 
 7.903770   , 
 2.132445   , 
 8.298597   , 
 4.247871   , 
 5.626433   , 
 2.014485   , 
 3.448579   , 
 2.692542   , 
 3.655845   , 
 6.060823   , 
 3.923493   , 
 3.972003   , 
 5.852386   , 
 4.543352   , 
 4.871040   , 
 5.950213   , 
 6.088972   , 
 4.658057   , 
 7.769017   , 
 4.409114   , 
 2.280840   , 
 5.670360   , 
 2.983921   , 
 6.207202   , 
 7.116859   , 
 7.423519   , 
 3.101225   , 
 6.013328   , 
 2.464174   , 
 6.212556   , 
 5.714635   , 
 2.208814   , 
 4.077979   , 
 4.884842 ]]).T

np.random.seed(1)
synaptic_weights = 2 * np.random.random((3, 1)) - 1
#OR, you can set the synaptic weights to be quite close to the ideal values of 0.3, 0.4 and 0.5.  
#To do so, uncomment the following three lines
#synaptic_weights[0]=0.32
#synaptic_weights[1]=0.39
#synaptic_weights[2]=0.515

ax.scatter3D(train_inputs[:,1],train_inputs[:,2],train_outputs)
plt.show()