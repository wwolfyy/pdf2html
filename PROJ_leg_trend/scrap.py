# %% API call from Python
# import requests

# url = "http://localhost:8000/convert_pdf"
# files = {"file": open("example.pdf", "rb")}
# response = requests.post(url, files=files)

# if response.status_code == 200:
#     result = response.json()
#     print(result)
# else:
#     print("Error:", response.status_code, response.text)

# %% handle output of API call from client
# import requests
# import base64

# url = "http://localhost:8000/convert_pdf"
# files = {"file": open("DATA/training_data/leg_trend/Issue_pdf/201711_phy_평석_02-형사-주거침입절도의 경우에는 통신사실확인자료를 요청할 수 없다.pdf", "rb")}
# response = requests.post(url, files=files)

# if response.status_code == 200:
#     result = response.json()
#     result = eval(result['result'])
# else:
#     print("Error:", response.status_code, response.text)

# # convert image strings to bytstring
# for k, v in result.items():
#     for idx, el in enumerate(v):
#         if v[idx]['tag'] == '<img>':
#             v[idx]['line'] = base64.b64decode(v[idx]['line'])


# %% using curl, from within Python script
# import subprocess

# subprocess.run(
#     [
#         "curl",
#         "-F",
#         "file=@../DATA/training_data/Issue_pdf/201711_phy_평석_02-형사-주거침입절도의 경우에는 통신사실확인자료를 요청할 수 없다.pdf",
#         "http://localhost:8000/convert_pdf",
#     ]
# )
