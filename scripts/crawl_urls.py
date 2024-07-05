import requests
from bs4 import BeautifulSoup


def crawl_urls(file_path):
    with open(file_path, 'r') as f:
        urls = f.read().splitlines()

    pages = []
    for url in urls:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                text = soup.get_text()
                pages.append((url, text))
        except Exception as e:
            print(f"Failed to retrieve {url}: {e}")
    return pages


if __name__ == "__main__":
    url_list_file = r"C:\Users\oleja\PycharmProjects\NLP test\data\url_list.txt"
    pages = crawl_urls(url_list_file)
    for url, text in pages:
        print(f"URL: {url}\nText: {text[:200]}...\n")
