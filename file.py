from googlesearch import search 
import requests
from bs4 import BeautifulSoup 


res = search("allergy", num_results=10 , lang="en")

result = list(res)
response = requests.get(result[0])
soup = BeautifulSoup(response.content , "html.parser")
print(soup.find_all("p")[0].text)
