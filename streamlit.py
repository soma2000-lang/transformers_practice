import requests
import json

class Generator:
    def __init__(self,nutrition_input:list,ingredients:list=[],params:dict={'n_neighbors':5,'return_distance':False}):
       """"""

    def set_request(self,nutrition_input:list,ingredients:list,params:dict):
       """"""

    def generate(self,):
        request={
          
        }
        response=requests.post(url='http://backend:8080/predict/',data=json.dumps(request))
        return response
