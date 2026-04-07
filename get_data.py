import requests
from bs4 import BeautifulSoup
import time

from pathlib import Path as path
cwd = path.cwd()
DATA_DIR = rf"{cwd}\data"


SITEMAP_URL = "https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/sitemap.xml"

def get_urls_from_sitemap(url):
    res = requests.get(url)
    soup = BeautifulSoup(res.text, "xml")
    
    urls = [loc.text for loc in soup.find_all("loc") if r"-ami-" not in loc.text]
    return urls

def scrape_page(url):
    res = requests.get(url)
    soup = BeautifulSoup(res.text, "html.parser")
    
    text = soup.get_text(separator=" ", strip=True)
    return text

def scrape_all():
    urls = get_urls_from_sitemap(SITEMAP_URL)
    
    print(f"Found {len(urls)} pages")

    print(urls[0:25])

    # for i, url in enumerate(urls):
    #     print(f"[{i}] Scraping: {url}")
        
    #     content = scrape_page(url)
        
    #     with open("ec2_clean.txt", "a", encoding="utf-8") as f:
    #         f.write(f"\n\n--- {url} ---\n\n{content}")
        
    #     time.sleep(5)  # Respect robots.txt

scrape_all()