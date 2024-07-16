from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn import svm
import xgboost as xgb


def Baselines(method,trainX,trainY,testX):
    if method == 'LR':  # 线性回归
        LR = LinearRegression()
        Norm_pred = LR.fit(trainX, trainY).predict(testX)
    elif method == 'MLP':  # MPL回归
        MLP = MLPRegressor(hidden_layer_sizes=(50,), activation='relu',
                           solver='adam', alpha=1e-04,
                           learning_rate_init=0.001, max_iter=1000)
        Norm_pred = MLP.fit(trainX, trainY).predict(testX)
    elif method == 'DT':  # 决策树回归
        DT = DecisionTreeRegressor()
        Norm_pred = DT.fit(trainX, trainY).predict(testX)
    elif method == 'SVR':  # 支持向量机回归
        SVR_MODEL = svm.SVR()
        Norm_pred = SVR_MODEL.fit(trainX, trainY).predict(testX)
    elif method == 'KNN':  # K近邻回归
        KNN = KNeighborsRegressor()
        Norm_pred = KNN.fit(trainX, trainY).predict(testX)
    elif method == 'RF':  # 随机森林回归
        RF = RandomForestRegressor(n_estimators=1000)
        Norm_pred = RF.fit(trainX, trainY).predict(testX)
    elif method == 'AdaBoost':  # Adaboost回归
        AdaBoost = AdaBoostRegressor(n_estimators=50)
        Norm_pred = AdaBoost.fit(trainX, trainY).predict(testX)
    elif method == 'XGB':  # XGB回归
        XGB_params = {'learning_rate': 0.1, 'n_estimators': 50,
                      'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
                      'objective': 'reg:squarederror', 'subsample': 0.8,
                      'colsample_bytree': 0.8, 'gamma': 0,
                      'reg_alpha': 0.1, 'reg_lambda': 0.1}
        XGB = xgb.XGBRegressor(**XGB_params)
        Norm_pred = XGB.fit(trainX, trainY).predict(testX)
    elif method == 'GBRT':  # GBRT回归
        GBRT = GradientBoostingRegressor(n_estimators=100)
        Norm_pred = GBRT.fit(trainX, trainY).predict(testX)
    elif method == 'BR':  # Bagging回归
        BR = BaggingRegressor()
        Norm_pred = BR.fit(trainX, trainY).predict(testX)
    elif method == 'ETR':  # ExtraTree极端随机树回归
        ETR = ExtraTreeRegressor()
        Norm_pred = ETR.fit(trainX, trainY).predict(testX)

    return Norm_pred


