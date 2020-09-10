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
    custom_cmap = ListedColormap([(0,0,1), (1,0,0)])
    plt.contourf(xx, yy, y_pred, alpha=0.3, cmap=custom_cmap)


#
# Read data
#
# OK
OK_1 = pd.read_csv("D:/data/LGC-welding/cu/OK/191115_145202_(No code)_OK.csv")
OK_2 = pd.read_csv("D:/data/LGC-welding/cu/OK/191115_145229_(No code)_OK.csv")
OK_3 = pd.read_csv("D:/data/LGC-welding/cu/OK/191115_145317_(No code)_OK.csv")
OK_4 = pd.read_csv("D:/data/LGC-welding/cu/OK/191120_141101_(No code)_OK.csv")
OK_5 = pd.read_csv("D:/data/LGC-welding/cu/OK/191120_141119_(No code)_OK.csv")

OKs = pd.concat([OK_1, OK_2, OK_3, OK_4, OK_5])  # merge
OKs["class"] = 0 # make result

# NG 1
NG_1 = pd.read_csv("D:/data/LGC-welding/cu/F_1/191115_161449_(No code)_NG.csv")
NG_2 = pd.read_csv("D:/data/LGC-welding/cu/F_1/191115_161458_(No code)_NG.csv")

NGs_1 = pd.concat([NG_1, NG_2])
NGs_1["class"] = 1

# NG 2
NG_1 = pd.read_csv("D:/data/LGC-welding/cu/F_2/191115_161541_(No code)_NG.csv")
NG_2 = pd.read_csv("D:/data/LGC-welding/cu/F_2/191115_161630_(No code)_NG.csv")
NG_3 = pd.read_csv("D:/data/LGC-welding/cu/F_2/191115_161646_(No code)_NG.csv")
NG_4 = pd.read_csv("D:/data/LGC-welding/cu/F_2/191120_141800_(No code)_OK.csv")
NG_5 = pd.read_csv("D:/data/LGC-welding/cu/F_2/191120_141821_(No code)_OK.csv")

NGs_2 = pd.concat([NG_1, NG_2, NG_3, NG_4, NG_5])
NGs_2["class"] = 1

# NG 3
NG_1 = pd.read_csv("D:/data/LGC-welding/cu/F_3/191115_161722_(No code)_NG.csv")
NG_2 = pd.read_csv("D:/data/LGC-welding/cu/F_3/191115_161746_(No code)_NG.csv")
NG_3 = pd.read_csv("D:/data/LGC-welding/cu/F_3/191120_141848_(No code)_OK.csv")

NGs_3 = pd.concat([NG_1, NG_2, NG_3])
NGs_3["class"] = 0

# NG 4
NG_1 = pd.read_csv("D:/data/LGC-welding/cu/F_4/191115_161834_(No code)_NG.csv")
NG_2 = pd.read_csv("D:/data/LGC-welding/cu/F_4/191115_161853_(No code)_NG.csv")
NG_3 = pd.read_csv("D:/data/LGC-welding/cu/F_4/191115_161911_(No code)_NG.csv")

NGs_4 = pd.concat([NG_1, NG_2, NG_3])
NGs_4["class"] = 0

# NG 5
NG_1 = pd.read_csv("D:/data/LGC-welding/cu/F_5/191115_161946_(No code)_NG.csv")
NG_2 = pd.read_csv("D:/data/LGC-welding/cu/F_5/191115_162005_(No code)_NG.csv")
NG_3 = pd.read_csv("D:/data/LGC-welding/cu/F_5/191115_162036_(No code)_NG.csv")
NG_4 = pd.read_csv("D:/data/LGC-welding/cu/F_5/191115_162048_(No code)_NG.csv")
NG_5 = pd.read_csv("D:/data/LGC-welding/cu/F_5/191120_141905_(No code)_OK.csv")

NGs_5 = pd.concat([NG_1, NG_2, NG_3, NG_4, NG_5])
NGs_5["class"] = 1

# NG 6
NG_1 = pd.read_csv("D:/data/LGC-welding/cu/F_6/191115_162148_(No code)_NG.csv")
NG_2 = pd.read_csv("D:/data/LGC-welding/cu/F_6/191120_141928_(No code)_OK.csv")

NGs_6 = pd.concat([NG_1, NG_2])
NGs_6["class"] = 1


# mearge
All = pd.concat( [OKs, NGs_1, NGs_2, NGs_3, NGs_4, NGs_5, NGs_6] )


temp = np.zeros((27,20),dtype=int)
precision = pd.DataFrame( temp )
recall = pd.DataFrame( temp )

#
#
#
xIndex = "Uluv_mean"   # "Xxyz_mean"    # "Hhsv_mean"   # "Uluv_mean"   #"area"
yIndex = "Vluv_mean"   # "Yxyz_mean"    # "Shsv_mean"   # "Vluv_mean"   #"index(welding)"
markSize = 5

#
#imageId = 26
#OK_X = OK.loc[ OK["index(image)"]==imageId, [xIndex, yIndex] ]
#OK_y = OK.loc[ NG["index(image)"]==imageId, "class" ]

#NG_X = NG.loc[ NG["index(image)"]==imageId, ["Xxyz_mean", "Yxyz_mean"] ]
#NG_y = NG.loc[ NG["index(image)"]==imageId, "class" ]

#plt.plot(OK_x.iloc[:,0], OK_x.iloc[:,1], 'bo', markersize=1)
#plt.plot(NG_x.iloc[:,0], NG_x.iloc[:,1], 'ro', markersize=1)
#plt.show()



#
#
#
#imageId = 26
#for imageId in range(27) :
for i in range(1):
    imageId = 26

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
    for level in range(7) :
        y_train_each = (y==level)
        y_pred_each = (y_pred==level)
        precision.iloc[imageId, level] = precision_score( y_train_each, y_pred_each )
        recall.iloc[imageId, level] = recall_score( y_train_each, y_pred_each)

    #
    # show result
    #
    if imageId == 26 :
        #plot_decision_boundary( rnd_clf, X, y, axes=[0.3, 0.5, 0.3, 0.5] )  # xyz
        plot_decision_boundary( rnd_clf, X, y, axes=[0.0, 1.0, 0.0, 1.0] )  # hsv
        #plot_decision_boundary( rnd_clf, X, y, axes=[3000, 9000, 0, 9] )  # xyz

        plt.plot(
            All.loc[ (All["index(image)"]==imageId) & (All["class"]==0), xIndex ],
            All.loc[ (All["index(image)"]==imageId) & (All["class"]==0), yIndex ],
            "bo", label="OK", markersize=markSize
        )

        plt.plot(
            All.loc[ (All["index(image)"]==imageId) & (All["class"]==1), xIndex ],
            All.loc[ (All["index(image)"]==imageId) & (All["class"]==1), yIndex ],
            "ro", label="OK", markersize=markSize
        )

        plt.show()


precision.to_csv("d:/test/precision.csv")
recall.to_csv("d:/test/recall.csv")

