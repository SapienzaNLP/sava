import argparse
import csv
import os
from datasets import load_dataset

def write_files(flores_ds_it, flores_ds_en, path_it_en, path_en_it, split):
    with open(path_it_en, "w") as f_it_en, open(path_en_it, "w") as f_en_it:
        write_it_en = csv.writer(f_it_en)
        write_en_it = csv.writer(f_en_it)

        write_it_en.writerow(["src","ref","lp"])
        write_en_it.writerow(["src","ref","lp"])

        for sample_it, sample_en in zip(flores_ds_it[split], flores_ds_en[split]):
            text_it = sample_it["text"]
            text_en = sample_en["text"]

            write_it_en.writerow([text_it, text_en,"it-en"])
            write_en_it.writerow([text_en, text_it,"en-it"])


def main(args):
    ouput_folder = args.output_folder

    output_folder_it_en = os.path.join(ouput_folder, "flores.it-en")
    output_folder_en_it = os.path.join(ouput_folder, "flores.en-it")

    if not os.path.exists(output_folder_it_en):
        os.makedirs(output_folder_it_en)

    if not os.path.exists(output_folder_en_it):
        os.makedirs(output_folder_en_it)

    dev_path_it_en = os.path.join(output_folder_it_en, "dev.csv")
    dev_path_en_it = os.path.join(output_folder_en_it, "dev.csv")

    test_path_it_en = os.path.join(output_folder_it_en, "test.csv")
    test_path_en_it = os.path.join(output_folder_en_it, "test.csv")

    flores_ds_it = load_dataset("openlanguagedata/flores_plus", "ita_Latn")
    flores_ds_en = load_dataset("openlanguagedata/flores_plus", "eng_Latn")

    # populate dev
    write_files(flores_ds_it, flores_ds_en, dev_path_it_en, dev_path_en_it, "dev")

    # populate test
    write_files(flores_ds_it, flores_ds_en, test_path_it_en, test_path_en_it, "devtest")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='download and process FLORES',
                    description='Download and preprocess the FLORES dataset from openlanguagedata/flores_plus')
    parser.add_argument('-o', '--output_folder')      # option that takes a value

    args = parser.parse_args()
    main(args)
