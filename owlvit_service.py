import requests
from service_interface import ServiceInterface
from PIL import Image
from typing import List
import numpy as np
# headers = {
#     'accept': '*/*',
#     # requests won't add a boundary if this header is set when you pass files=
#     # 'Content-Type': 'multipart/form-data',
# }

# files = {
#     'image': (None, 'http://74.82.29.209:9000//datasets/media/movies/1388863_fullriverred_499613.jpeg'),
#     'question': (None, 'what is the left person in the image holding in his hand?'),
# }

# try:
#     response = requests.post('http://124.70.217.159:8086/infer', headers=headers, files=files)
#     print(response.json())
# except Exception as e: 
#     a=0

class OwlVitService(ServiceInterface):
    def __init__(self):
        self.owlvit_service_ip = 'http://209.51.170.37:8086/infer'#'http://127.0.0.1:8086/infer'#'http://184.105.3.17:8086/infer'
        self.headers = {
            'accept': '*/*',
            # requests won't add a boundary if this header is set when you pass files=
            # 'Content-Type': 'multipart/form-data',
        }
    
    def get_url_response(self, image_url, question, score_threshold):
        print(score_threshold)
        files = {
            'image': (None, image_url),
            'question': (None, question),
            'score': (None, score_threshold)
        }
        try:
            response = requests.post(self.owlvit_service_ip, headers=self.headers, files=files)
            return [response.json()['answer']]
        except Exception as e:
            print("Error: {}".format(e))
            return None
    
    def get_image_response(self, image : Image, text : List[str]) -> List[str]:
        raise Exception("Use URLs only.")

def main():

    owl_service = OwlVitService()

    # # Inputs
    url = "http://74.82.29.209:9000//datasets/media/movies/1388863_fullriverred_499613.jpeg"
    texts = [["A photo of a person", "A photo of sword"]]
    import json
    texts = json.dumps(texts)
    # texts = np.asarray(texts, dtype=np.object_)
    score_threshold = json.dumps(0.1)
    outputs = owl_service.get_url_response(url, texts, score_threshold)

    print("Outputs: {}".format(outputs))

if __name__ == "__main__":
    main()