from selenium import webdriver
import pandas as pd
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

#NOTE: There is a chance that youtube will block or restrict the amount of data I can scrape

playlist_link = "https://www.youtube.com/c/khanacademy/playlists"
videos_link = "https://www.youtube.com/c/khanacademy/videos?view=0&sort=dd&shelf_id=0"

driver = webdriver.Chrome('/Users/barnabyperkins/Code/knowledge-graph/Vault/chromedriver')
driver.get("https://www.youtube.com/c/khanacademy/videos?view=0&sort=dd&shelf_id=0")

# Agree to Youtube terms and conditions
agree_button = driver.find_element(By.XPATH, '//*[@id="yDmH0d"]/c-wiz/div/div/div/div[2]/div[1]/div[4]/form/div/div/button')
agree_button.click()

# Find all videos on page
# NOTE: This is only loading the first 30 videos in the scroll, will have to trigger the rests
data = driver.find_elements(By.XPATH, '//*[@id="video-title"]')
print(len(data))

# Get Links to each video
links = []
for i in data:
    links.append(i.get_attribute('href'))

with open("video_links.txt", "w") as file:
    for link in links:
        file.write(link + "\n")
