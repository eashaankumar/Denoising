import gdown

url = "https://drive.google.com/uc?id=1-gygvIph7lvl7m4DnBfLA6kdDBYq6Tj2"
output = "/workspace/dataset.zip"
gdown.download(url, output, quiet=False)