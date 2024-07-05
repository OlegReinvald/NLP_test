import gdown

def download_file_from_google_drive(file_id, dest_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, dest_path, quiet=False)

if __name__ == "__main__":
    file_id = '1TnOCjWiYIcD6lUdozC5o5dMiRp3k7AdP'  # Replace with your file ID
    dest_path = r'C:\Users\oleja\PycharmProjects\NLP test\data\url_list.txt'
    download_file_from_google_drive(file_id, dest_path)