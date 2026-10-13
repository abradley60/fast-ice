import pooch
from pathlib import Path

def fetch_svm_trainingdata(output_dir):
    """
    Download and unzip the SVM training data from the Digital Earth Antarctica
    fast-ice public data repository, returning the path to the extracted directory.

    Parameters
    ----------
    output_dir : str or pathlib.Path — directory to download/unzip into

    Returns
    -------
    pathlib.Path to the extracted directory containing the training data
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    url = (
        "https://deant-data-public-dev.s3.ap-southeast-2.amazonaws.com/"
        "persistent/repositories/fast-ice/SVM_trainingdata/SVM_trainingdata.zip"
    )

    fnames = pooch.retrieve(
        url=url,
        known_hash="0e00e5732c7428641c1f8fa1957b3bad83aef5fb08a252c1af99b06072fa0d25",
        fname="SVM_trainingdata.zip",
        path=output_dir,
        processor=pooch.Unzip(extract_dir="."),
    )

    return Path(output_dir) / "SVM_trainingdata"