import requests
from bs4 import BeautifulSoup
import urllib.parse

class NewsCrawler:
    def __init__(self, url="https://www.bbc.co.uk/news"):
        self.url = url
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
        }
        self.seen_links = set()
    
    def extract_latest_hot_news(self, url="", limit=5):
        """
        extract latest hot news from BBC
        : return: news list, including title and link
        """
        loaded_url = url if url else self.url
        response = requests.get(loaded_url, headers=self.headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        # 找到所有带链接的新闻卡片, 必须是以 /news/articles/ 开头的链接
        articles = soup.select('a[href^="/news/articles/"]')
        results = []
        seen_links = set()

        for article in articles:
            link = article.get('href')
            if not link.startswith('http'):
                link = "https://www.bbc.co.uk" + link

            # 去重
            if link in seen_links:
                continue
            seen_links.add(link)

            # 获取标题
            title_tag = article.find(['h3', 'p'])
            title = title_tag.get_text(strip=True) if title_tag else "无标题"
            content = self.extractNewsTextByURL(link)["content"]

            results.append({
                'title': title,
                'url': link,
                'content': content
            })

            if len(results) >= limit:
                break
        return results
    
    def handle_search_input(self, search_input):
        encoded_query = urllib.parse.quote_plus(search_input)  # handle spaces and special characters
        return encoded_query



    def extract_search_news(self, quary, limit=5):
        """
        extract search news from BBC
        """

        base_url = "https://www.bbc.co.uk/search?q="
        encoded_query = self.handle_search_input(quary)
        loaded_url =  f"{base_url}{encoded_query}&d=NEWS_PS"
        response = requests.get(loaded_url, headers=self.headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        # 找到所有带链接的新闻卡片, 必须是以 /news/articles/ 开头的链接
        articles = soup.select('a[href^="https://www.bbc.co.uk/news/articles/"]')
        results = []
        seen_links = set()

        for article in articles:
            link = article.get('href')
            if not link.startswith('http'):
                link = "https://www.bbc.co.uk" + link

            # 去重
            if link in seen_links:
                continue
            seen_links.add(link)

            # 获取标题
            title_tag = article.find(['h3', 'p'])
            title = title_tag.get_text(strip=True) if title_tag else "无标题"
            content = self.extractNewsTextByURL(link)["content"]

            results.append({
                'title': title,
                'url': link,
                'content': content
            })

            if len(results) >= limit:
                break
        return results
    
    def extractNewsTextByURL(self, url):
        """
        提取新闻文本
        :param url: 新闻链接
        :return: 新闻文本
        """
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 提取新闻标题和正文
        title = soup.find('h1').get_text(strip=True) if soup.find('h1') else "无标题"

        mainText = ""
        paragraphs = soup.select('div[data-component^="text-block"] p')
        for i in paragraphs:
            sectionText = i.get_text(strip=True)
            if sectionText:
                mainText += sectionText + "\n"
        
        return {
            'title': title,
            'content': mainText.strip()
        }
