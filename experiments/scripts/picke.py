import pickle
import torch
import joblib

filename= 'C:/Users/EloiMartins/PycharmProjects/tracking_wo_bnw/output/output/faster_rcnn_fpn_training_mot_17/model_final_b275ba.pkl'
filename2= 'C:/Users/EloiMartins/PycharmProjects/tracking_wo_bnw/output/output/faster_rcnn_fpn_training_mot_17/model_epoch_.model'
filename3= 'C:/Users/EloiMartins/PycharmProjects/tracking_wo_bnw/output/output/faster_rcnn_fpn_training_mot_17/R-50.pkl'
path= 'C:/Users/EloiMartins/PycharmProjects/tracking_wo_bnw/output/output/faster_rcnn_fpn_training_mot_17/model.model'
with open(filename,'rb')as handle:
    model_full=pickle.load(handle)
model=model_full['model']
torch.save(model,path)


with open(filename2,'rb')as file:
    model_EPOCH=torch.load(file)

with open(path,'rb')as file1:
    model_full2=torch.load(file1)


with open(filename3,'rb')as handle2:
    model_R50=pickle.load(handle2)
model2=model_full['model']
print('ei')








