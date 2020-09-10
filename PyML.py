
#%%
import os
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

import joblib as jl



#%%
g_value = 10

def test():
    a = 123 * 100
    print( "python embedding")
    return a
    

def multiply(a,b):
    c = 0
    c = a*b
    return c
    

def add(a,b):
    g_value = a + b
    return g_value    


def get():
    return g_value    



#%%
class MyClass:

    m_data = 0
    m_model = 0

    #
    m_nRows = 0 # 학습 데이터의 갯수
    m_nCols = 0 # 학습 데이터의 dimesion
    m_A = 0 # 학습용 데이터
    m_B = 0 # 학습용 데이터(분류)
    m_X = 0 # 테스트용 데이터
    m_clf_rnd = 0 # RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1 )
    m_clf_svc = 0
    m_clf_softmax = 0
    m_clf_lda = 0
    m_clf_qda = 0

    def __init__(self):
        self.m_data = 0
        self.m_model = np.random.rand(2, 3) * 100

        self.m_nRows = 0
        self.m_nCols = 0
        self.m_A = 0
        self.m_B = 0
        self.m_X = 0
        self.m_clf_rnd = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1 )
        self.m_clf_softmax = LogisticRegression( multi_class="multinomial", solver="lbfgs", C=10 )
        self.m_cls_svc = SVC(kernel="poly", degree=3, coef0=1, C=5)
        self.m_clf_lda = LDA( n_components=2 )
        self.m_clf_qda = QDA()


    def pushData( self, row, data ):
        self.m_model[ row, 0 ] = data[0]
        self.m_model[ row, 1 ] = data[1]
        self.m_model[ row, 2 ] = data[2]

        res = self.m_model.ravel()

        return res


    def getModel(self):
        data = self.m_model.ravel()
        return data


    def add( self, data ) :
        self.m_data = self.m_data + data
        return self.m_data


    def set( self, iCnt, data ) :
        sum = 0

        for i in range(3) :
            sum = sum + data[i]
        
        return sum


    # ML의 위한 data 영역 생성
    def ML_setDim( self, nRows, nCols ) :
        self.m_nRows = nRows # convert to array (input)
        self.m_nCols = nCols
        self.m_A = np.zeros( (nRows, nCols) ) # 학습 데이터
        self.m_B = np.zeros( (nRows, 1) )     # 학습 데이터
        self.m_X = np.zeros( (1, nCols) )     # 테스트 데이터
       

    def ML_setA(self, nRow, nCol, data):
        self.m_A[nRow, nCol] = data


    def ML_setB(self, nRow, nCol, data):
        self.m_B[nRow, nCol] = data

    
    def ML_setX(self, nCol, data):
        self.m_X[0, nCol] = data

    
    def ML_learn(self):
        self.m_clf_rnd.fit(self.m_A, self.m_B)

    
    def ML_learn_svc(self):
        self.m_cls_svc.fit(self.m_A, self.m_B)

    
    def ML_learn_soft(self):
        self.m_clf_softmax.fit(self.m_A, self.m_B)

    
    def ML_learn_lda(self):
        self.m_clf_lda.fit(self.m_A, self.m_B)

    
    def ML_learn_qda(self):
        self.m_clf_qda.fit(self.m_A, self.m_B)

    
    def ML_predict(self):
        y_pred = self.m_clf_rnd.predict(self.m_X)

        data = y_pred.ravel()

        return data

    
    def ML_predict_svc(self):
        y_pred = self.m_cls_svc.predict(self.m_X)

        data = y_pred.ravel()

        return data

    
    def ML_predict_soft(self):
        y_pred = self.m_clf_softmax.predict(self.m_X)

        data = y_pred.ravel()

        return data

    
    def ML_predict_lda(self):
        y_pred = self.m_clf_lda.predict(self.m_X)

        data = y_pred.ravel()

        return data

    
    def ML_predict_qda(self):
        y_pred = self.m_clf_qda.predict(self.m_X)

        data = y_pred.ravel()

        return data

    
    def fit( self, nCols, nRows, data_a, data_b ) :      
        self.m_nRows = nRows # convert to array (input)
        self.m_nCols = nCols
        self.m_A = np.asarray( data_a ).reshape(nRows, nCols)
        self.m_B = np.asarray( data_b ).reshape(nRows, 1)

        # fit
        self.m_clf_rnd.fit(self.m_A, self.m_B)
        
        return 0


    def predict( self, nCols, nRows, data ) :
        # convert to array(input)
        A = np.asarray( data ).reshape(nRows, nCols)

        # predict
        y_pred = self.m_clf_rnd.predict(A)

        # convert (output)
        data = y_pred.ravel()

        return data


    def save( self, name ) :
        jl.dump( self.m_clf_rnd, name )

        return 0


    def load( self, name ) :
        self.m_clf_rnd = jl.load( name )

        return 0






#%%


instance = MyClass()

instance.getModel()



#%%

instance.ML_setDim( 4, 2 )

instance.ML_setA( 0, 0, 0 )
instance.ML_setA( 0, 1, 0 )
instance.ML_setB( 0, 0, 0 )

instance.ML_setA( 1, 0, 0.5 )
instance.ML_setA( 1, 1, 0 )
instance.ML_setB( 1, 0, 0 )

instance.ML_setA( 2, 0, 1 )
instance.ML_setA( 2, 1, 0 )
instance.ML_setB( 2, 0, 1 )

instance.ML_setA( 3, 0, 1.5 )
instance.ML_setA( 3, 1, 0 )
instance.ML_setB( 3, 0, 1 )

instance.ML_learn()
instance.ML_learn_svc()
instance.ML_learn_soft()
#instance.ML_learn_lda()
#instance.ML_learn_qda()


#%%

instance.ML_setX(0, 1.0)
instance.ML_setX(1, 0)


#%%
print( "random forest = " )
print( instance.ML_predict() )

print( "SVC = " )
print( instance.ML_predict_svc() )

print( "soft max = " )
print( instance.ML_predict_soft() )

#print( "lda = " )
#print( instance.ML_predict_lda() )

#print( "qda = " )
#print( instance.ML_predict_qda() )


#%%
#instance.load( 'd:/pyModel.pkl')
instance.save( 'd:/mlModel.bhc')


#%%
#arr = (1,2,3)
#instance.pushData(0, arr)

#%%
#res = instance.getModel()
#print(res)


#%%
#instance.add(1)


# %%
print("hello")



# %%
#array = np.ones((3,2))

#array2 = np.asarray((1,2,3,4,5,6)).reshape(3,2)

#print(array2)


# %%
#array

#A = [1, 0, 0, 1, 0.5, 0, 0, 0]
#B = [0, 1, 0, 1]

#instance.fit( 2, 4, A, B)



# %%
#A = [1, 0]
#res = instance.predict( 2, 1, A)
#print(res)

# %%
#A = [0.2, 0]
#res = instance.predict( 2, 1, A)
#print(res)

# %%
