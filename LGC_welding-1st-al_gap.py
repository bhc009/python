import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score


def plot_decision_boundary(clf, X, y, axes=[0, 7.5, 0, 3], iris=True, legend=False, plot_training=True):
    # gen mesh data
    xs = np.linspace( axes[0], axes[1], 100 )
    ys = np.linspace( axes[2], axes[3], 100 )
    xx, yy = np.meshgrid( xs, ys )

    # predict
    x_new = np.c_[ xx.ravel(), yy.ravel() ]
    y_pred = clf.predict( x_new ).reshape( xx.shape )

    # gen color map
#    custom_cmap = ListedColormap(['#fafab0', '#9898ff', '#a0faa0'])
#    custom_cmap = ListedColormap([(0,0,0), (0.3,0,0), (0.6,0,0), (1,0,0), (0,0.3,0), (0,0.6,0), (0,1,0), (0,0,0.3), (0,0,0.6)])
    custom_cmap = ListedColormap([(0,0,1), (0,1,0), (0,1,0), (0,1,0), (1,0,0), (1,0,0)])
    plt.contourf(xx, yy, y_pred, alpha=0.3, cmap=custom_cmap)


#
# Read data
#
# OK
OK_1 = pd.read_csv("D:/data/LGC-welding/AL/OK/191115_143553_(No code)_NG.csv")
OK_2 = pd.read_csv("D:/data/LGC-welding/AL/OK/191115_143611_(No code)_NG.csv")
OK_3 = pd.read_csv("D:/data/LGC-welding/AL/OK/191115_143638_(No code)_NG.csv")
OK_4 = pd.read_csv("D:/data/LGC-welding/AL/OK/191115_143700_(No code)_NG.csv")
OK_5 = pd.read_csv("D:/data/LGC-welding/AL/OK/191115_143715_(No code)_NG.csv")

OKs = pd.concat([OK_1, OK_2, OK_3, OK_4, OK_5])  # merge
OKs["class"] = 0 # make result

# NG 1
NG_1 = pd.read_csv("D:/data/LGC-welding/AL/G1/191115_143823_(No code)_NG.csv")
NG_2 = pd.read_csv("D:/data/LGC-welding/AL/G1/191115_143956_(No code)_NG.csv")
NG_3 = pd.read_csv("D:/data/LGC-welding/AL/G1/191115_144006_(No code)_NG.csv")
NG_4 = pd.read_csv("D:/data/LGC-welding/AL/G1/191115_144020_(No code)_NG.csv")
NG_5 = pd.read_csv("D:/data/LGC-welding/AL/G1/191120_140735_(No code)_NG.csv")

NGs_1 = pd.concat([NG_1, NG_2, NG_3, NG_4, NG_5])
NGs_1["class"] = 1

# NG 2
NG_1 = pd.read_csv("D:/data/LGC-welding/AL/G2/191115_144042_(No code)_NG.csv")
NG_2 = pd.read_csv("D:/data/LGC-welding/AL/G2/191115_144058_(No code)_NG.csv")
NG_3 = pd.read_csv("D:/data/LGC-welding/AL/G2/191115_144114_(No code)_NG.csv")

NGs_2 = pd.concat([NG_1, NG_2, NG_3])
NGs_2["class"] = 2

# NG 3
NG_1 = pd.read_csv("D:/data/LGC-welding/AL/G3/191115_144545_(No code)_NG.csv")
NG_2 = pd.read_csv("D:/data/LGC-welding/AL/G3/191115_144603_(No code)_NG.csv")
NG_3 = pd.read_csv("D:/data/LGC-welding/AL/G3/191115_144641_(No code)_NG.csv")
NG_4 = pd.read_csv("D:/data/LGC-welding/AL/G3/191120_140842_(No code)_NG.csv")

NGs_3 = pd.concat([NG_1, NG_2, NG_3, NG_4])
NGs_3["class"] = 3

# NG 4
NG_1 = pd.read_csv("D:/data/LGC-welding/AL/G4/191115_144707_(No code)_NG.csv")
NG_2 = pd.read_csv("D:/data/LGC-welding/AL/G4/191115_144720_(No code)_NG.csv")
NG_3 = pd.read_csv("D:/data/LGC-welding/AL/G4/191115_144733_(No code)_NG.csv")
NG_4 = pd.read_csv("D:/data/LGC-welding/AL/G4/191115_144804_(No code)_NG.csv")
NG_5 = pd.read_csv("D:/data/LGC-welding/AL/G4/191120_141018_(No code)_NG.csv")

NGs_4 = pd.concat([NG_1, NG_2, NG_3, NG_4, NG_5])
NGs_4["class"] = 4

# NG 5
NG_1 = pd.read_csv("D:/data/LGC-welding/AL/G5/191115_144823_(No code)_NG.csv")
NG_2 = pd.read_csv("D:/data/LGC-welding/AL/G5/191115_144840_(No code)_NG.csv")
NG_3 = pd.read_csv("D:/data/LGC-welding/AL/G5/191115_144847_(No code)_NG.csv")
NG_4 = pd.read_csv("D:/data/LGC-welding/AL/G5/191115_144857_(No code)_NG.csv")
NG_5 = pd.read_csv("D:/data/LGC-welding/AL/G5/191115_144909_(No code)_NG.csv")

NGs_5 = pd.concat([NG_1, NG_2, NG_3, NG_4, NG_5])
NGs_5["class"] = 5


# mearge
All = pd.concat( [OKs, NGs_1, NGs_2, NGs_3, NGs_4, NGs_5] )


#
#
#
xIndex = "area"   # "Xxyz_mean" # "Shsv_mean"
yIndex = "index(welding)"   # "Yxyz_mean"   # "Vhsv_mean"
markSize = 5

temp = np.zeros((27,20),dtype=int)
precision = pd.DataFrame( temp )
recall = pd.DataFrame( temp )


#imageId = 26
for imageId in range(27) :
    #
    # train
    #
    X = All.loc[ (All["index(image)"]==imageId) , [xIndex, yIndex]]
    y = All.loc[ (All["index(image)"]==imageId) , "class"]

    rnd_clf = RandomForestClassifier(n_estimators=100, max_leaf_nodes=16, n_jobs=-1 )

    rnd_clf.fit( X, y )

    # estimate
    y_pred = rnd_clf.predict(X)

    # get score
    # level 별 score
    for level in range(6) :
        y_train_each = (y==level)
        y_pred_each = (y_pred==level)
        precision.iloc[imageId, level] = precision_score( y_train_each, y_pred_each )
        recall.iloc[imageId, level] = recall_score( y_train_each, y_pred_each)

    # block data
    y_train_last = ( (y==5) | (y==4) )
    y_pred_last  = ((y_pred==5) | (y_pred==4))
    precision.iloc[imageId, 10] = precision_score(y_train_last, y_pred_last)
    recall.iloc[imageId, 10] = recall_score(y_train_last, y_pred_last)

    #
    # show result
    #
    if imageId == 26 :
        #plot_decision_boundary( rnd_clf, X, y, axes=[0.3, 0.5, 0.3, 0.5] )  # xyz
        #plot_decision_boundary( rnd_clf, X, y, axes=[0.0, 0.6, 0.0, 0.6] )  # hsv
        plot_decision_boundary( rnd_clf, X, y, axes=[10000, 20000, 0, 9] )  # xyz

        plt.plot(
            All.loc[ (All["index(image)"]==imageId) & (All["class"]==0), xIndex ],
            All.loc[ (All["index(image)"]==imageId) & (All["class"]==0), yIndex ],
            "bs", label="OK", markersize=markSize
        )

        plt.plot(
            All.loc[ (All["index(image)"]==imageId) & (All["class"]==1), xIndex ],
            All.loc[ (All["index(image)"]==imageId) & (All["class"]==1), yIndex ],
            "g^", label="OK", markersize=markSize
        )

        plt.plot(
            All.loc[ (All["index(image)"]==imageId) & (All["class"]==2), xIndex ],
            All.loc[ (All["index(image)"]==imageId) & (All["class"]==2), yIndex ],
            "go", label="OK", markersize=markSize
        )

        plt.plot(
            All.loc[ (All["index(image)"]==imageId) & (All["class"]==3), xIndex ],
            All.loc[ (All["index(image)"]==imageId) & (All["class"]==3), yIndex ],
            "gs", label="OK", markersize=markSize
        )

        plt.plot(
            All.loc[ (All["index(image)"]==imageId) & (All["class"]==4), xIndex ],
            All.loc[ (All["index(image)"]==imageId) & (All["class"]==4), yIndex ],
            "r^", label="OK", markersize=markSize
        )

        plt.plot(
            All.loc[ (All["index(image)"]==imageId) & (All["class"]==5), xIndex ],
            All.loc[ (All["index(image)"]==imageId) & (All["class"]==5), yIndex ],
            "ro", label="OK", markersize=markSize
        )

        plt.show()


precision.to_csv("d:/test/precision.csv")
recall.to_csv("d:/test/recall.csv")
