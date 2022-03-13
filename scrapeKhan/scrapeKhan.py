import transcriptScraper

scraper = transcriptScraper.transcriptScraper
khan_scraper = scraper("https://www.youtube.com/c/khanacademy/videos")
khan_scraper.scrape()
