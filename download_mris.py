import csv
import os
import pexpect
import sys


N_FILES = 200
CSV_PATH = "data/raw/csv/adni_aibl.csv"
REMOTE_PATH = "/local/ADNI_AIBL/ADNI_AIBL_T1_smoothed/all_images"
LOCAL_PATH = "brain_data/ADNI_AIBL/ADNI_AIBL_T1_smoothed/all_images/"


def main():
    pwd = sys.argv[1]
    with open(CSV_PATH) as csvfile:
        reader = csv.DictReader(csvfile)
        c = 0
        for row in reader:
            c += 1
            if c > N_FILES:
                break

            file_path = row["image_label"] + "_mni_aligned.nii.gz"
            remote = os.path.join(REMOTE_PATH, file_path)

            cmd = 'scp mhoerold@isegpu2.inf.ethz.ch:{}'.format(remote)
            cmd += ' {}'.format(LOCAL_PATH)
            print(cmd)
            child = pexpect.spawn(cmd)
            child.expect("mhoerold@isegpu2.inf.ethz.ch's password:")
            child.sendline(pwd)
            child.expect(pexpect.EOF)

if __name__ == "__main__":
    main()
