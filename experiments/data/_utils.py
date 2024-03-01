"""

    Utils

"""

import tarfile
from contextlib import closing
from pathlib import Path

import requests
from tqdm import tqdm


def download(
    url: str,
    dst: Path,
    chunk_size: int = 1024,
    verbose: bool = True,
) -> Path:
    """Download a file from a ``url`` to ``dst``.

    Args:
        url (str): URL of the file to download
        dst (Path): download destination. If a directory, the filename
            will be determined using the URL.
        chunk_size (int): size of "chunks" to use when streaming the data
        verbose (bool): if ``True`` display a progress bar

    Returns:
        dst (Path): the path to the downloaded file

    """
    if dst.is_dir():
        dst = dst.joinpath(Path(url).name)

    response = requests.get(url, stream=True)
    total = int(response.headers.get("content-length", 0)) or None
    with dst.open("wb") as file:
        with tqdm(
            desc=f"Downloading {Path(url).name}",
            total=total,
            unit="iB",
            unit_scale=True,
            unit_divisor=chunk_size,
            disable=total is None or not verbose,
        ) as pbar:
            for data in response.iter_content(chunk_size=chunk_size):
                size = file.write(data)
                pbar.update(size)
    return dst


def untar(
    src: Path,
    dst: Path,
    delete_src: bool = False,
    verbose: bool = False,
) -> Path:
    """Untar ``src``.

    Args:
        src (Path): source file to untar
        dst (Path): destination directory
        delete_src (bool): if ``True``, delete ``src`` when complete

    Returns:
        None

    """
    if not src.is_file():
        raise OSError(f"No file {str(dst)}")
    if not dst.is_dir():
        raise OSError(f"No directory {str(dst)}")

    if verbose:
        print(f"Untaring {str(src)}...")
    with closing(tarfile.open(src)) as f:
        f.extractall(dst)

    if delete_src:
        src.unlink()
