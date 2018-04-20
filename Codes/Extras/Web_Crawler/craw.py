import requests
from bs4 import BeautifulSoup

print("Word:")
w = str(raw_input())
res = requests.get("http://www.dictionary.com/browse/" + w)
soup = BeautifulSoup(res.content,"lxml")

print(soup.find_all("div",{"class":"def-content"})[0].text)