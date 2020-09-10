import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score


def plot_decision_boundary(clf, X, y, axes=[0, 7.5, 0, 3], iris=True, legend=False, plot_training=True):
    # gen mesh data
    xs = np.linspace( axes[0], axes[1], 1000 )
    ys = np.linspace( axes[2], axes[3], 1000 )
    xx, yy = np.meshgrid( xs, ys )

    # predict
    x_new = np.c_[ xx.ravel(), yy.ravel() ]
    y_pred = clf.predict( x_new ).reshape( xx.shape )

    # gen color map
#    custom_cmap = ListedColormap(['#fafab0', '#9898ff', '#a0faa0'])
#    custom_cmap = ListedColormap([(0,0,0), (0.3,0,0), (0.6,0,0), (1,0,0), (0,0.3,0), (0,0.6,0), (0,1,0), (0,0,0.3), (0,0,0.6)])
#    custom_cmap = ListedColormap([(0,0,1), (0,0,1), (0,0,1), (0,1,0), (0,1,0), (0,1,0), (1,0,0), (1,0,0), (1,0,0)])
    custom_cmap = ListedColormap([(0,0,1), (1,0,0)])
    plt.contourf(xx, yy, y_pred, alpha=0.3, cmap=custom_cmap)


#
# Read data
#
# OK
OK_1 = pd.read_csv("D:/data/LGC-welding/image/20200529-2nd/Cu/gap/200524_0_1_2.csv")
OK_2 = pd.read_csv("D:/data/LGC-welding/image/20200529-2nd/Cu/gap/200524_0_3_4.csv")
OK_3 = pd.read_csv("D:/data/LGC-welding/image/20200529-2nd/Cu/gap/200524_0_5_6.csv")
OK_4 = pd.read_csv("D:/data/LGC-welding/image/20200529-2nd/Cu/gap/200524_0_7_8.csv")
OK_5 = pd.read_csv("D:/data/LGC-welding/image/20200529-2nd/Cu/gap/200524_0_9_10.csv")

OKs = pd.concat([OK_1, OK_2, OK_3, OK_4, OK_5])  # merge
OKs["class"] = 0 # make result

# NG 1
NG_1 = pd.read_csv("D:/data/LGC-welding/image/20200529-2nd/Cu/focus/200524_F1_1_2.csv")
NG_2 = pd.read_csv("D:/data/LGC-welding/image/20200529-2nd/Cu/focus/200524_F1_3_4.csv")
#NG_3 = pd.read_csv("D:/data/LGC-welding/image/20200529-2nd/Cu/focus/200524_P1_5_6.csv")
#NG_4 = pd.read_csv("D:/data/LGC-welding/image/20200529-2nd/Cu/focus/200524_P1_7_8.csv")
#NG_5 = pd.read_csv("D:/data/LGC-welding/image/20200529-2nd/Cu/focus/200524_P1_9_10.csv")

NGs_1 = pd.concat([NG_1, NG_2])
NGs_1["class"] = 0

# NG 2
NG_1 = pd.read_csv("D:/data/LGC-welding/image/20200529-2nd/Cu/focus/200524_F2_1_2.csv")
NG_2 = pd.read_csv("D:/data/LGC-welding/image/20200529-2nd/Cu/focus/200524_F2_3_4.csv")
NG_3 = pd.read_csv("D:/data/LGC-welding/image/20200529-2nd/Cu/focus/200524_F2_5_6.csv")
NG_4 = pd.read_csv("D:/data/LGC-welding/image/20200529-2nd/Cu/focus/200524_F2_7_8.csv")
NG_5 = pd.read_csv("D:/data/LGC-welding/image/20200529-2nd/Cu/focus/200524_F2_9_10.csv")

NGs_2 = pd.concat([NG_1, NG_2, NG_2, NG_4, NG_5])
NGs_2["class"] = 0

# NG 3
NG_1 = pd.read_csv("D:/data/LGC-welding/image/20200529-2nd/Cu/focus/200524_F3_1_2.csv")
NG_2 = pd.read_csv("D:/data/LGC-welding/image/20200529-2nd/Cu/focus/200524_F3_3_4.csv")
NG_3 = pd.read_csv("D:/data/LGC-welding/image/20200529-2nd/Cu/focus/200524_F3_5_6.csv")

NGs_3 = pd.concat([NG_1, NG_2, NG_3])
NGs_3["class"] = 1

# NG 4
NG_1 = pd.read_csv("D:/data/LGC-welding/image/20200529-2nd/Cu/focus/200524_F4_3_4.csv")
NG_2 = pd.read_csv("D:/data/LGC-welding/image/20200529-2nd/Cu/focus/200524_F4_5_6.csv")
#NG_3 = pd.read_csv("D:/data/LGC-welding/image/20200529-2nd/Cu/focus/200524_P4_5_6.csv")

NGs_4 = pd.concat([NG_1, NG_2])
NGs_4["class"] = 1

# NG 5
NG_1 = pd.read_csv("D:/data/LGC-welding/image/20200529-2nd/Cu/focus/200524_F5_1_2.csv")
NG_2 = pd.read_csv("D:/data/LGC-welding/image/20200529-2nd/Cu/focus/200524_F5_3_4.csv")
NG_3 = pd.read_csv("D:/data/LGC-welding/image/20200529-2nd/Cu/focus/200524_F5_7_8.csv")
NG_4 = pd.read_csv("D:/data/LGC-welding/image/20200529-2nd/Cu/focus/200524_F5_9_10.csv")

NGs_5 = pd.concat([NG_1, NG_2, NG_3, NG_4])
NGs_5["class"] = 1

# NG 6
#NG_1 = pd.read_csv("D:/data/LGC-welding/image/20200529-2nd/Cu/focus/200524_P6_1_2.csv")
#NG_2 = pd.read_csv("D:/data/LGC-welding/image/20200529-2nd/Cu/focus/200524_P6_3_4.csv")
#NG_3 = pd.read_csv("D:/data/LGC-welding/image/20200529-2nd/Cu/focus/200524_P6_5_6.csv")

#NGs_6 = pd.concat([NG_1, NG_2, NG_3])
#NGs_6["class"] = 1

# NG 7
#NG_1 = pd.read_csv("D:/data/LGC-welding/image/20200529-2nd/Cu/focus/200524_P7_5_6.csv")
#NG_2 = pd.read_csv("D:/data/LGC-welding/image/20200529-2nd/Cu/focus/200524_P7_7_8.csv")
#NG_3 = pd.read_csv("D:/data/LGC-welding/image/20200529-2nd/Cu/focus/200524_P7_9_10.csv")
#NG_4 = pd.read_csv("D:/data/LGC-welding/image/20200529-2nd/Cu/focus/191115_161143_(No code)_NG.csv")
#NG_5 = pd.read_csv("D:/data/LGC-welding/image/20200529-2nd/Cu/focus/191120_153128_(No code)_NG.csv")

#NGs_7 = pd.concat([NG_1, NG_2, NG_3])
#NGs_7["class"] = 1

# NG 8
#NG_1 = pd.read_csv("D:/data/LGC-welding/image/20200529-2nd/Cu/focus/200524_P8_1_2.csv")
#NG_2 = pd.read_csv("D:/data/LGC-welding/image/20200529-2nd/Cu/focus/200524_P8_3_4.csv")
#NG_3 = pd.read_csv("D:/data/LGC-welding/image/20200529-2nd/Cu/focus/200524_P8_5_6.csv")
#NG_4 = pd.read_csv("D:/data/LGC-welding/image/20200529-2nd/Cu/focus/200524_P8_7_8.csv")
#NG_5 = pd.read_csv("D:/data/LGC-welding/image/20200529-2nd/Cu/focus/200524_P8_9_10.csv")

#NGs_8 = pd.concat([NG_1, NG_2, NG_3, NG_4, NG_5])
#NGs_8["class"] = 1

# mearge
All = pd.concat( [OKs, NGs_1, NGs_2, NGs_3, NGs_4, NGs_5] )


temp = np.zeros((27,20),dtype=int)
precision = pd.DataFrame( temp )
recall = pd.DataFrame( temp )

#
#
#
xIndex = "Vluv_mean"   # "Xxyz_mean"    # "Hhsv_mean"   #"Vluv_mean"
yIndex = "Uluv_mean"   # "Yxyz_mean"    # "Shsv_mean"   #"Uluv_mean"
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
    imageId = 1
    #
    # train
    #
    X = All.loc[ (All["index(image)"]==imageId) , [xIndex, yIndex]]
    y = All.loc[ (All["index(image)"]==imageId) , "class"]

    rnd_clf = RandomForestClassifier(n_estimators=100, max_leaf_nodes=4, n_jobs=-1 )

    rnd_clf.fit( X, y )

    # estimate
    y_pred = rnd_clf.predict(X)

    # get score
    # level 별 score
    for level in range(9) :
        y_train_each = (y==level)
        y_pred_each = (y_pred==level)
        precision.iloc[imageId, level] = precision_score( y_train_each, y_pred_each )
        recall.iloc[imageId, level] = recall_score( y_train_each, y_pred_each)

    # block data
    #y_train_last = ( (y==8) | (y==7) | (y==6) )
    #y_pred_last  = ((y_pred==8) | (y_pred==7) | (y_pred==6))
    #precision.iloc[imageId, 10] = precision_score(y_train_last, y_pred_last)
    #recall.iloc[imageId, 10] = recall_score(y_train_last, y_pred_last)

    #
    # show result
    #
    if imageId == 1 :
        #plot_decision_boundary( rnd_clf, X, y, axes=[0.3, 0.5, 0.3, 0.5] )  # xyz
        plot_decision_boundary( rnd_clf, X, y, axes=[0.0, 1.0, 0.0, 1.0] )  # hsv
        #plot_decision_boundary( rnd_clf, X, y, axes=[3000, 9000, 0, 9] )  # xyz

        plt.plot(
            All.loc[ (All["index(image)"]==imageId) & (All["class"]==0), xIndex ],
            All.loc[ (All["index(image)"]==imageId) & (All["class"]==0), yIndex ],
            "bo", label="OK(3,4)", markersize=markSize
        )

        plt.plot(
            All.loc[ (All["index(image)"]==imageId) & (All["class"]==1), xIndex ],
            All.loc[ (All["index(image)"]==imageId) & (All["class"]==1), yIndex ],
            "ro", label="NG(1,2,5,6)", markersize=markSize
        )

        #plt.xlabel("x @ xyy")
        #plt.ylabel("y @ xyy")
        #plt.xlabel("u @ hsv")
        #plt.ylabel("s @ hsv")
        plt.xlabel("u @ luv")
        plt.ylabel("v @ luv")

        plt.legend()

        plt.show()


precision.to_csv("d:/test/precision.csv")
recall.to_csv("d:/test/recall.csv")

