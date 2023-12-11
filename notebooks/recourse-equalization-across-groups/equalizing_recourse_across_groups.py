# source: https://github.com/vgupta123/recourse-equalization/tree/master

import numpy as np
from sklearn.utils import check_random_state
from sklearn.base import clone
from blackbox_agnostic import LimeTabularClassification

def prepare_data(df, y):
    """Prepare data to feed into the implementation of "equalizing recourse across groups"."""
    dfs_dict = {
        "trainx": df.drop(columns="groups").copy().values,
        "testx":None,
        "traingrp": df.groups.values,
        "testgrp": None,
        "trainy": y.values,
        "testy": None
    }
    num_train_samples, num_features = dfs_dict["trainx"].shape
    num_test_samples = 0
    
    # Output:
    #     - dictionary of data with keys (trainx, testx, traingrp, testgrp, trainy, testy)
    #     - number of features
    #     - number of training samples
    #     - number of test samples
    return dfs_dict, num_features, num_train_samples, num_test_samples


def run_agnostic(X, y, classifier, sed=1):
    best_acc = 0
    chosen_samples = []
    chosen_weights = []
    test_samples = []
    test_weights = []

    # Fix randomness
    np.random.seed(sed) 
    data, d, ntr, ntst = prepare_data(X, y)

    results = []

    ##############################################################################################################
    # run loop 5 times - This section generates synthetic data and sample weights to train a regularized classifier
    ##############################################################################################################
    ntrial = 0
    while ntrial < 5:
        ntrial += 1

        samples = []
        weights = []

        rst = check_random_state(np.random.randint(100))

        # retrieve a classifier and train it
        clf = clone(classifier)
        clf.fit(data['trainx'], data['trainy'], sample_weight=np.ones(ntr))
        # print("Coefs first:", clf.coef_, clf.intercept_)
        ypred = clf.predict(data['trainx'])
 
        lf = LimeTabularClassification(data['trainx'], sample_around_instance=False, random_state=rst)

        runs = 2
        probs = np.zeros((runs, ntr))

        for i in range(runs):
            samp, ws, distsnpreds = lf.cal_distance(data['trainx'], clf.predict_proba)
            samples.append(samp)
            weights.append(ws)
            probs[i] += distsnpreds[:,2]
        
        avgProbs = np.average(probs, axis=0)
        # avgProbs[avgProbs < 0.5] = -1  # original
        avgProbs[avgProbs < 0.5] = 0
        avgProbs[avgProbs >= 0.5] = 1

        acc = float(np.sum(avgProbs * ypred > 0))/len(avgProbs)
        if acc > best_acc:
            best_acc = acc
            chosen_samples = samples.copy()
            chosen_weights = weights.copy()
        # print(ntrial)
        # print(acc)
    ##############################################################################################################
    # End of section
    ##############################################################################################################
    
    # print(best_acc)

    # I don't understand why this while True is here
    while True:
        rst = check_random_state(np.random.randint(100))

        # BEFORE 
        clf = clone(classifier)
        clf.fit(data['trainx'], data['trainy'], sample_weight=np.ones(ntr))
        # print("Coefs before:", clf.coef_, clf.intercept_)
        
        yall = clf.predict(data['trainx'])
        ypred = yall # [:ntr]
        # ytest = yall[ntr:]

        ltrain = LimeTabularClassification(data['trainx'], sample_around_instance=False, random_state=rst)
        # ltest = LimeTabularClassification(data['testx'], sample_around_instance=False, random_state=rst)

        runs = 2
        # tdists = np.zeros((runs, ntst))
        dists = np.zeros((runs, ntr))

        for i in range(runs):
            _, _, distsnpreds = ltrain.cal_distance(data['trainx'], clf.predict_proba, neighbors=(chosen_samples[i], chosen_weights[i]))
            # s, w, distsnpreds = ltrain.cal_distance(data['trainx'], clf.predict_proba)
            # chosen_samples.append(s)
            # chosen_weights.append(w)

            # ts, tw, tdistsnpreds = ltest.cal_distance(data['testx'], clf.predict_proba)
            # test_samples.append(ts)
            # test_weights.append(tw)

            dists[i] = dists[i] + distsnpreds[:,0]
        #     tdists[i] += tdistsnpreds[:,0]
        # 
        avgTraindist = np.average(dists, 0)
        # avgTstdist = np.average(tdists, 0)
        # 
        # minTrainD, maxTrainD = np.min(avgTraindist[ypred==-1]), np.max(avgTraindist[ypred==-1])

        # negavgtraindist = avgTraindist[ypred==-1] # original
        negavgtraindist = avgTraindist[ypred==0]


        # negtrainpredgrps = data['traingrp'][ypred==-1]
        # traingrpcnt = Counter(negtrainpredgrps)
        # gposTrainAvg = (np.sum(negavgtraindist[negtrainpredgrps==1])/traingrpcnt[1]) if traingrpcnt[1] != 0 else 0
        # gnegTrainAvg = (np.sum(negavgtraindist[negtrainpredgrps==-1])/traingrpcnt[-1]) if traingrpcnt[-1] != 0 else 0
 
        # recourse_diff_train = (gposTrainAvg - gnegTrainAvg)/(maxTrainD - minTrainD)
 
 
        # minTestD, maxTestD = np.min(avgTstdist[ytest==-1]), np.max(avgTstdist[ytest==-1])
        # negavgtestdist = avgTstdist[ytest==-1]
 
        # negtestpredgrps = data['testgrp'][ytest==-1]
        # testgrpcnt = Counter(negtestpredgrps)
        # gposTestAvg = (np.sum(negavgtestdist[negtestpredgrps==1])/testgrpcnt[1]) if testgrpcnt[1] != 0 else 0
        # gnegTestAvg = (np.sum(negavgtestdist[negtestpredgrps==-1])/testgrpcnt[-1]) if testgrpcnt[-1] != 0 else 0
 
        # recourse_diff_test = (gposTestAvg - gnegTestAvg)/(maxTestD - minTestD)
 
        # acc_train = float(np.sum(ypred * data['trainy'] > 0))/len(ypred)     
        # acc_test = float(np.sum(ytest * data['testy'] > 0))/len(ytest)
 
        # results.extend([abs(recourse_diff_train), abs(recourse_diff_test), acc_train, acc_test])

        
        
        # AFTER
        # training with weights inversely proportional to approx distance
        countneg = len(negavgtraindist)
        countneg_posdist = len(negavgtraindist[negavgtraindist > 0])
        m = np.min(negavgtraindist[negavgtraindist > 0]) if (2*countneg_posdist > countneg) else np.min(negavgtraindist)

        new_weights = np.ones(ntr)
        # new_weights[ypred==-1] = m/negavgtraindist  # original 
        new_weights[ypred==0] = m/negavgtraindist
        new_weights[(new_weights < 0)] = 1

        clf = clone(classifier)
        clf.fit(data['trainx'], data['trainy'], sample_weight=new_weights)
        # print("Coefs after:", clf.coef_, clf.intercept_)
        return clf
        # yall = clf.predict(data['both'])
        # ypred = yall[:ntr]
        # ytest = yall[ntr:]
        # 
        # tdists = np.zeros((runs, ntst))
        # dists = np.zeros((runs, ntr))
        # probs = np.zeros((runs, ntr))
        # 
        # for i in range(runs):
        #     distsnpreds = ltrain.cal_distance(data['trainx'], clf.predict_proba, neighbors=(chosen_samples[i], chosen_weights[i]))[2]
        #     tdistsnpreds = ltest.cal_distance(data['testx'], clf.predict_proba, neighbors=(test_samples[i], test_weights[i]))[2]
        # 
        #     dists[i] = dists[i] + distsnpreds[:,0]
        #     tdists[i] += tdistsnpreds[:,0]
        # 
        # avgTraindist = np.average(dists, 0)
        # avgTstdist = np.average(tdists, 0)
        # 
        # minTrainD, maxTrainD = np.min(avgTraindist), np.max(avgTraindist)
        # negavgtraindist = avgTraindist[ypred==-1]
        # 
        # negtrainpredgrps = data['traingrp'][ypred==-1]
        # traingrpcnt = Counter(negtrainpredgrps)
        # gposTrainAvg = (np.sum(negavgtraindist[negtrainpredgrps==1])/traingrpcnt[1]) if traingrpcnt[1] != 0 else 0
        # gnegTrainAvg = (np.sum(negavgtraindist[negtrainpredgrps==-1])/traingrpcnt[-1]) if traingrpcnt[-1] != 0 else 0
        # 
        # recourse_diff_train = (gposTrainAvg - gnegTrainAvg)/(maxTrainD - minTrainD)
        # 
        # minTestD, maxTestD = np.min(avgTstdist), np.max(avgTstdist)
        # negavgtestdist = avgTstdist[ytest==-1]
        # 
        # negtestpredgrps = data['testgrp'][ytest==-1]
        # testgrpcnt = Counter(negtestpredgrps)
        # gposTestAvg = (np.sum(negavgtestdist[negtestpredgrps==1])/testgrpcnt[1]) if testgrpcnt[1] != 0 else 0
        # gnegTestAvg = (np.sum(negavgtestdist[negtestpredgrps==-1])/testgrpcnt[-1]) if testgrpcnt[-1] != 0 else 0
        # 
        # recourse_diff_test = (gposTestAvg - gnegTestAvg)/(maxTestD - minTestD)
        # 
        # acc_train = float(np.sum(ypred * data['trainy'] > 0))/len(ypred)
        # acc_test = float(np.sum(ytest * data['testy'] > 0))/len(ytest)
        # 
        # results.extend([abs(recourse_diff_train), abs(recourse_diff_test), acc_train, acc_test])
        # 
        # return results
