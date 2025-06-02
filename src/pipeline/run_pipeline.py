from data.load_data import IrisDataDownloader

def main():
    downloader = IrisDataDownloader()
    downloader.download()

if __name__ == "__main__":
    main()