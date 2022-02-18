from selenium import webdriver
import pandas as pd
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


class khan_scraper():

    SCROLL_PAUSE = 0.5

    def get_transcript(self):
        pass

    def open_transcript(self):
        pass

    def open_video(self):
        pass

    def get_links(self):
        links = []
        link_data = self.driver.find_elements(By.XPATH, '//*[@id="video-title"]')
        for link in link_data:
            links.append(link.get_attribute('href'))
        return

    def scroll_down(self):
        height = self.driver.execute_script('return document.body.scrollHeight')
        while True:
            self.driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')
            time.sleep(self.SCROLL_PAUSE)
            new_height = self.driver.execute_script('return document.body.scrollHeight')
            if new_height == height:
                break
            height = new_height

    def agree_terms(self):
        agree_button = self.driver.find_element(By.XPATH, '//*[@id="yDmH0d"]/c-wiz/div/div/div/div[2]/div[1]/div[4]/form/div/div/button')
        agree_button.click()
        return get_links()

    def prepare_driver(self):
        self.driver = webdriver.Chrome('/Users/barnabyperkins/Code/knowledge-graph/Vault/chromedriver')
        self.driver.get(self.start_page)
        return agree_terms()

    def scrape(self):
        pass

    def __init__(self, start_page):
        self.start_page = start_page
