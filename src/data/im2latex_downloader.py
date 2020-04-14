import requests
import argparse
import os

from concurrent.futures import ThreadPoolExecutor


def get_filename(url):
    return url.split('/')[-1].split('?')[0]


def download(url, save_loc='.'):
    filename = get_filename(url)
    write_loc = save_loc + '/' + filename
    print(f"Downloading from {url} to {save_loc}")
    response = requests.get(url, stream=True)
    with open(write_loc, 'wb') as file:
        for chunk in response.iter_content(chunk_size=512):
            if chunk:
                file.write(chunk)


def main():
    parser = argparse.ArgumentParser(description='Download im2latex dataset')

    parser.add_argument('--save_loc',
                        type=str,
                        default="../../data/raw/",
                        help="Path where the files will be saved.")

    parser.add_argument('--parallel_downloads',
                        type=int,
                        default=2,
                        help="Number of parallel downloads.")

    parser.add_argument('--urls_file',
                        help="""File with the URLs of the files which """
                        """will be downloaded.""",
                        type=str,
                        default="./dataset_files.txt")

    args = parser.parse_args()

    lines = []
    with open(args.urls_file, 'r') as file:
        lines = file.read().splitlines()

    if not os.path.exists(args.save_loc):
        raise ValueError("The save directory does not exist")

    print("Downloading files. This might take a while...")
    with ThreadPoolExecutor(max_workers=args.parallel_downloads) as executor:
        for l in lines:
            executor.submit(download, l, args.save_loc)


if __name__ == "__main__":
    main()
