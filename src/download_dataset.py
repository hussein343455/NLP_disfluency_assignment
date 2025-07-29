import requests
import os


def download_file(url, folder_name="../Data/raw"):
    """
    Downloads a file from a given URL and saves it to a specified folder.
    """
    # Create the target directory if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Created directory: '{folder_name}'")

    # Extract filename from the URL
    filename = url.split('/')[-1]
    filepath = os.path.join(folder_name, filename)

    print(f"Downloading {filename}...")

    try:
        # Send a GET request to the URL
        response = requests.get(url, stream=True)
        # Raise an HTTPError for bad responses (4xx or 5xx)
        response.raise_for_status()

        # Write the content to the file in chunks
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"✅ Successfully downloaded to '{filepath}'")

    except requests.exceptions.RequestException as e:
        print(f"❌ Error downloading {url}: {e}")


if __name__ == "__main__":
    # Note: GitHub page URLs were converted to raw content URLs for direct download.
    urls = [
        "https://raw.githubusercontent.com/google-research-datasets/Disfl-QA/main/train.json",
        "https://raw.githubusercontent.com/google-research-datasets/Disfl-QA/main/dev.json",
        "https://raw.githubusercontent.com/google-research-datasets/Disfl-QA/main/test.json"
    ]

    for url in urls:
        download_file(url)

    print("\nDataset download process complete. ✨")