import os
import shutil
import re


def zip_folders(record_folder, reg):
    if not os.path.exists(record_folder):
        return
    names = os.listdir(record_folder)
    for name in names:
        p = os.path.join(record_folder, name)
        if not os.path.isdir(p):
            continue
        match = reg.match(p)
        if match is not None:
            out_name = p
            shutil.make_archive(out_name, 'zip', p)
            shutil.rmtree(p)


def remove_folders(record_folder, reg):
    if not os.path.exists(record_folder):
        return
    names = os.listdir(record_folder)
    for name in names:
        p = os.path.join(record_folder, name)
        if not os.path.isdir(p):
            continue
        match = reg.match(p)
        if match is not None:
            shutil.rmtree(p)


def compress_embedding_folders(record_folder, keep):
    if keep:
        pattern = record_folder + "/" + "train_[0-9]{1}"
        reg = re.compile(pattern)
        zip_folders(record_folder, reg)
        pattern = record_folder + "/" + "test_[0-9]{1}"
        reg = re.compile(pattern)
        zip_folders(record_folder, reg)
    else:
        pattern = record_folder + "/" + "train_[0-9]{1}"
        reg = re.compile(pattern)
        remove_folders(record_folder, reg)
        pattern = record_folder + "/" + "test_[0-9]{1}"
        reg = re.compile(pattern)
        remove_folders(record_folder, reg)
