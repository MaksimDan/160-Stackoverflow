import requests
import progressbar


def download_file(url):
    local_filename = url.split('/')[-1]
    r = requests.get(url, stream=True)
    with open(local_filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    return local_filename

if __name__ == "__main__":
	base = 'https://archive.org'
	bar = progressbar.ProgressBar()
	with open('urls', 'r') as f:
		contents = f.readlines()[0].split()
		for url in bar(contents):
			download_file(base + url)
