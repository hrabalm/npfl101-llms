from pathlib import Path
import datasets

username = "hrabalm"  # TODO: update username
my_datasets = Path(f"/storage/brno12-cerit/home/{username}/datasets")
my_datasets.mkdir(exist_ok=True, parents=True) # ensure directory exists

eng = Path("./data/eng")
ces = Path("./data/ces")

def create_dataset():
    data = [{
        "source_text": ces_text,
        "target_text": eng_text,
        "source_lang": "cs",
        "target_lang": "en",
    } for ces_text, eng_text in zip(ces.read_text().splitlines(), eng.read_text().splitlines())]

    print("First five examples:", data[0:5])

    # see also from_dict, from_generator, from_pandas, etc.
    # https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.Dataset.from_dict
    ds = datasets.Dataset.from_list(data)

    return ds

if __name__ == "__main__":
    ds = create_dataset()
    ds.save_to_disk(str(my_datasets / "npfl101_test_dataset"))
    # ds.push_to_hub("USERNAME/npfl101_test_dataset", private=True)  # TODO: fix HF username, check that HF_API_KEY is set

# Optional exercises:
# - use from_generator to load data from a generator so that the file does not have to be loaded into memory
