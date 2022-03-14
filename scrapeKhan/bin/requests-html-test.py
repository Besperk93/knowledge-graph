from bs4 import BeautifulSoup
from requests_html import HTMLSession

session = HTMLSession()
url = "https://www.youtube.com/youtubei/v1/get_transcript?key=AIzaSyAO_FJ2SlqU8Q4STEHLGCilw_Y9_11qcW8"
response = session.post(url)
print(response.content)
