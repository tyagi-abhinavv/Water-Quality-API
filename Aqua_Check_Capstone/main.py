from fastapi import FastAPI
from device_data_input import Device_Input
import pickle
import numpy as np
import pdb
from pycaret.regression import load_model, predict_model
import pandas as pd

app = FastAPI()

chloramines_model = load_model('Chloramines_pipeline')
sulfates_model = load_model('Sulfate_pipeline')
organic_carbon_model = load_model('Organic_Carbon_pipeline')
trihalomethanes_model = load_model('Trihalomethanes_pipeline')

@app.post('/evaluate')
def evaluate_chemical_parameters(input: Device_Input):
    
    #input_list = []
    #input_list.append(input.ph)
    #input_list.append(input.Hardness)
    #input_list.append(input.Solids)
    #input_list.append(input.Conductivity)
    #input_list.append(input.Turbidity)    
    #input_array = np.array(input_list).reshape(1,-1)
    input_df = pd.DataFrame([[input.ph, input.Hardness, input.Solids, input.Conductivity, input.Turbidity]])
    input_df.columns = ['ph', 'Hardness', 'Solids', 'Conductivity', 'Turbidity']

    output_1 = predict_model(chloramines_model, data=input_df) 
    output_2 = predict_model(sulfates_model, data=input_df) 
    output_3 = predict_model(organic_carbon_model, data=input_df) 
    output_4 = predict_model(trihalomethanes_model, data=input_df) 

    return{
        "Chloaramines Concentration": int(output_1['prediction_label'][0]),
        "Sulphates Concentration": int(output_2['prediction_label'][0]),
        "Organic Carbon Concentration": int(output_3['prediction_label'][0]),
        "Trihalomethanes Concentration": int(output_4['prediction_label'][0])
    }