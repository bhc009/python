import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


rawData = pd.read_csv("D:/data/exterior/20200511.csv")

particleData = rawData.loc[ rawData["particle"]==1 , ["U", "V"] ]
defectData = rawData.loc[ rawData["particle"]==0 , ["U", "V"] ]



#
#
#
plt.plot( particleData["U"], particleData["V"], "bo", label="particle" )
plt.plot( defectData["U"], defectData["V"], "ro", label="defect" )

plt.title("Classify by UV")
plt.xlabel( "U @ UVW" )
plt.ylabel( "V @ UVW" )
plt.legend()

plt.show()



#
#
#
#plt.plot( particleData["StdU"], particleData["StdV"], "bo", label="particle" )
#plt.plot( defectData["StdU"], defectData["StdV"], "ro", label="defect" )

#plt.title("Classify by STD")
#plt.xlabel( "std @ U" )
#plt.ylabel( "std @ V" )
#plt.legend()

#plt.show()



