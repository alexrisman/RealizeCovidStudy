from data import dcm_to_jpg, load_jpg, get_dicom_fields
from models import get_covidnet, get_chexnet, get_covidnet_s
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

def predict(jpeg_path):
    pred_dict = {}
    for model_name in model_names:
        curr_pred_dict = {}
        img = load_jpg(jpeg_path, model_name=model_name)
        if model_name == 'covidnet':
            pred = covidnet_sess.run(covidnet_pred_tensor, feed_dict={covidnet_image_tensor: np.expand_dims(img, axis=0)})
            curr_pred_dict['pred_class'] = inv_mapping[pred.argmax(axis=1)[0]]
            curr_pred_dict['normal_prob'] = pred[0][0]
            curr_pred_dict['non_covid_pneum_prob'] = pred[0][1]
            curr_pred_dict['covid_prob'] = pred[0][2]
        elif model_name == 'covidnet_s_geo':
            pred = covidnet_s_geo_model.infer(img)
            curr_pred_dict['covid_prob'] = pred[0]
            curr_pred_dict['pred_class'] = curr_pred_dict['covid_prob'] >= .5
        elif model_name == 'covidnet_s_opc':
            pred = covidnet_s_opc_model.infer(img)
            curr_pred_dict['covid_prob'] = pred[0]
            curr_pred_dict['pred_class'] = curr_pred_dict['covid_prob'] >= .5            
        else:
            batch_x = img.reshape(1, 224, 224, 3)
            pred = chexnet_model.predict(batch_x)[0]
            curr_pred_dict['covid_prob'] = pred[chexnet_pneumonia_index]
            curr_pred_dict['pred_class'] = curr_pred_dict['covid_prob'] >= .5
        pred_dict[model_name] = curr_pred_dict
    if logging_ind:
        print(jpeg_path)
        print(pred_dict)
    return pred_dict


logging_ind = True
mapping = {'normal': 0, 'pneumonia': 1, 'COVID-19': 2}
inv_mapping = {0: 'normal', 1: 'pneumonia', 2: 'COVID-19'}
dcm_dir = 'lima_129_manual_download_from_cimar/'
model_names = ['covidnet', 'chexnet', 'covidnet_s_geo', 'covidnet_s_opc']
labels_df = pd.read_csv('lima_129.csv')
covidnet_sess, covidnet_image_tensor, covidnet_pred_tensor = get_covidnet()
covidnet_s_geo_model = get_covidnet_s()
covidnet_s_opc_model = get_covidnet_s(geo_ind=False)
chexnet_model, chexnet_pneumonia_index = get_chexnet()
accession_numbers = labels_df.accession_number.values
covid_pcr_inds = labels_df.covid_pcr_ind.values
case_dicts = []
for i in range(labels_df.shape[0]):
    case_dict = {}
    accession_number = str(accession_numbers[i])
    covid_pcr_ind = covid_pcr_inds[i]
    dcm_path = dcm_dir + accession_number + '.dcm'
    jpeg_path = dcm_to_jpg(accession_number, dcm_path)
    pred_dict = predict(jpeg_path)
    for model_name in model_names:
        model_pred_dict = pred_dict[model_name]
        for key in model_pred_dict.keys():
            val = model_pred_dict[key]
            field_name = "{0}_{1}".format(model_name, key)
            case_dict[field_name] = val 
    dcm_dict = get_dicom_fields(dcm_path)
    case_dict['accession_number'] = accession_number
    case_dict['manufacturer'] = dcm_dict['manufacturer']
    case_dict['covid_pcr_ind'] = covid_pcr_ind
    case_dicts.append(case_dict)
df = pd.DataFrame(case_dicts)
df.to_csv("model_preds.csv", index=False)
