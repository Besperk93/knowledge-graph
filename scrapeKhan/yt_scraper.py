from selenium import webdriver
import pandas as pd
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


class yt_channel_scraper():

    SCROLL_PAUSE = 10.0

    def save_text(self, text):
        try:
            page_title = self.driver.title
            with open(f'transcripts/{page_title}.txt', 'w') as file:
                file.write(text)
        except Exception as e:
            print(f'Error saving text: {repr(e)}')
            self.driver.quit()

    def get_transcript(self):
        try:
            time.sleep(10)
            transcript = self.driver.find_element(By.CSS_SELECTOR, '#body.style-scope.ytd-transcript-renderer')
            text = transcript.text
            return self.save_text(text)
        except Exception as e:
            print(f'Error getting the transcript: {repr(e)}')
            self.driver.quit()


    def open_transcript(self):
        try:
            time.sleep(10)
            transcript_button = self.driver.find_element(By.XPATH, '//*[contains(text(),"Open transcript")]')
            transcript_button.click()
            return self.get_transcript()
        except Exception as e:
            print(f'Error opening transcript: {repr(e)}')
            self.driver.quit()


    def open_menu(self):
        try:
            time.sleep(10)
            menu_button = self.driver.find_element(By.CSS_SELECTOR, '#button.dropdown-trigger.style-scope.ytd-menu-renderer')
            menu_button.click()
            # menu_button = self.driver.find_element(By.CSS_SELECTOR, '#button.yt-icon-button')
            # menu_button.click()
            return self.open_transcript()
        except Exception as e:
            print(f'Error opening menu: {repr(e)}')
            self.driver.quit()


    def open_video(self, links):
        try:
            # TODO: This splitter is not woking as expected. look into how we can queue up each video
            for video_link in links:
                self.driver.get(video_link)
                return self.open_menu()
        except Exception as e:
            print(f'Error opening videos: {repr(e)}')
            self.driver.quit()

    def get_links(self):
        try:
            links = []
            link_data = self.driver.find_elements(By.XPATH, '//*[@id="video-title"]')
            for link in link_data:
                links.append(link.get_attribute('href'))
            print(len(links))
            with open("videoLinks/video_links.txt", "w") as file:
                for link in links:
                    file.write(link + "\n")
            return self.open_video(links)
        except Exception as e:
            print(f'Error getting links: {repr(e)}')
            self.driver.quit()


    def scroll_down(self):
        # TODO: This isn't working, the browser is waiting but not scrolling. check the javascript
        try:
            height = self.driver.execute_script(r'return document.body.scrollHeight')
            while True:
                self.driver.execute_script(r'window.scrollTo(0, document.body.scrollHeight);')
                time.sleep(self.SCROLL_PAUSE)
                new_height = self.driver.execute_script(r'return document.body.scrollHeight')
                if new_height == height:
                    break
                height = new_height
            return self.get_links()
        except Exception as e:
            print(f'Error scrolling page: {repr(e)}')
            self.driver.quit()


    def agree_terms(self):
        try:
            agree_button = self.driver.find_element(By.XPATH, '//*[@id="yDmH0d"]/c-wiz/div/div/div/div[2]/div[1]/div[4]/form/div/div/button')
            agree_button.click()
            return self.scroll_down()
        except Exception as e:
            print(f'Error agreeing to page terms: {repr(e)}')
            self.driver.quit()


    def prepare_driver(self):
        try:
            options = webdriver.ChromeOptions()
            options.add_argument("start-maximized");
            options.add_argument("disable-infobars")
            options.add_argument("--disable-extensions")
            self.driver = webdriver.Chrome(chrome_options=options, executable_path='/Users/barnabyperkins/Code/knowledge-graph/Vault/chromedriver')
            self.driver.get(self.start_page)
            return self.agree_terms()
        except Exception as e:
            print(f'Error preparing driver: {repr(e)}')
            self.driver.quit()


    def scrape(self):
        self.prepare_driver()

    def __init__(self, start_page):
        self.start_page = start_page

khan_scraper = yt_channel_scraper("https://www.youtube.com/c/khanacademy/videos")
khan_scraper.scrape()
