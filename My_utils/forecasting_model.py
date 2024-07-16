import torch


#%%           #forecasting-multi-outputs method-all sectiom#####################

def multi_outputs_forecasting(model,testX,testY,Normalization):

    all_simu = []
    all_real = []
    model.eval()  # 转换成测试模式
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pred = model(testX.float().to(device))
    Norm_pred = pred.data.cpu().numpy()
    for i in range(len(testX)):
        simu = Normalization.inverse_transform(Norm_pred[i, :, :])
        all_simu.append(simu)
        real = Normalization.inverse_transform(testY[i, :, :].data.numpy())
        all_real.append(real)

    return all_simu,all_real



