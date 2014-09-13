#/usr/bin python

import rbmPredict
import rbmVtoH

def DBNPredict(m, X, isSingleDBN = True) :
    """implement DBN predict
    m : models trained(save in np.array)
    X : data to predict"""

    H = len(m)

    if isSingleDBN :
        for index in range(H-1) :
            X = rbmVtoH.rbmVtoH(m[index], X)

        [prediction, F] = rbmPredict.rbmPredict(m[H-1], X)

        return [prediction, F]
    else :
        for index in range(H) :
            X = rbmVtoH.rbmVtoH(m[index], X)

        return X

if __name__ == "__main__" :
    pass