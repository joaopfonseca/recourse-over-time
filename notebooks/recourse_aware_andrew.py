import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from scipy.optimize import basinhopping


class RecourseAwareClassifer(BaseEstimator, ClassifierMixin):
    def __init__(self, init_model, l = 100, niter=1, group_feature="groups", threshold=0.5):
        self.init_model = init_model
        self.l = l
        self.niter = niter
        self.group_feature = group_feature
        self.threshold = threshold

    def _sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def _getx0(self, X, y):
        self.init_model.fit(X, y)
        
        x0 = []
        x0.append(self.init_model.intercept_[0])
        x0.extend(list(self.init_model.coef_[0]))

        return x0

    def _getLoss(self, x, X, y, g):
        # Add column for intercept
        X = np.hstack((np.ones((X.shape[0],1)),X))
        
        linear_pred = np.dot(X,x)
        predictions = self._sigmoid(linear_pred)

        log_loss = np.sum(np.abs(y - predictions))

        # Group recourse difference

        # To get the distance from P3 perpendicular to a line drawn between P1 and P2
        c = -x[0]/x[2]
        m = -x[1]/x[2]

        p1=np.array([0,m*0 + c])
        p2=np.array([1,m*1 + c])

        avg_distances = []

        for group in np.unique(g):
            group_mask = g == group
            negative_class_mask = y == 0
            masks = group_mask & negative_class_mask

            total_distance = 0
            for x in (X[masks]):
                p3=np.array([x[1],x[2]])
                d=np.cross(p2-p1,p3-p1)/np.linalg.norm(p2-p1)
                total_distance += d

            n = len(X[masks])
            avg_distances.append(total_distance / n)

        group_recourse_loss = abs(avg_distances[0] - avg_distances[1])
        # print(log_loss, group_recourse_loss)
        loss = log_loss + self.l*group_recourse_loss
        
        return loss

    def fit(self, X, y):
        g = X[self.group_feature]
        X = X.drop(columns=self.group_feature)
        X, y = check_X_y(X, y)

        if X.shape[1] != 2:
            raise Exception("Only X matrices with 2 features are supported")

        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y

        f = lambda x: self._getLoss([x[0],x[1],x[2]],X,y,g)
        minimizer_kwargs = {"method":"BFGS", "jac":False}
        x0 = self._getx0(self.X_,self.y_)
        ret = basinhopping(f, x0, minimizer_kwargs=minimizer_kwargs, niter=2)

        self.w_ = ret.x
        
        # TEST
        ########################################
        self.w_ = self.w_ / self.w_.sum()
        ########################################
        
        self.intercept_ = np.array([list(self.w_)[0]])
        self.coef_ = np.array([list(self.w_)[1:]])

        return self

    # def predict_proba(self):
    #     check_is_fitted(self)
    # 
    #     X = np.hstack((np.ones((self.X_.shape[0],1)),self.X_))
    #     linear_pred = np.dot(X,self.w_)
    #     
    #     return linear_pred
    
    def predict_proba(self, X):
        check_is_fitted(self)

        X = X.drop(columns=self.group_feature)
        X = np.hstack((np.ones((X.shape[0],1)),X))
        linear_pred = np.dot(X,self.w_)
        linear_pred = np.expand_dims(linear_pred, 1)
        
        # Pass scores over a sigmoid
        pred = 1/(1+np.exp(-linear_pred))

        return np.hstack([pred, 1-pred])
    
    def predict(self, X):
        return (self.predict_proba(X)[:, -1] > self.threshold).astype(int).squeeze()
    