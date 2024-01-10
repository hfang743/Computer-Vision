import numpy as np
import torch
import pandas as pd
import re
import shutil
import os


def CreateFolders(Dir):
    os.mkdir(Dir)
    return


def SeparateCD(Dir, DirCats, DirDogs):
    for f in os.listdir(Dir):
        catSearchObj = re.search("cat", f)
        dogSearchObj = re.search("dog", f)
        if catSearchObj:
            shutil.move(f'{Dir}/{f}', DirCats)
        elif dogSearchObj:
            shutil.move(f'{Dir}/{f}', DirDogs)

    print('All pictures now in subdirectories')

    return


def MoveFiles(toDir, fromDir):
    for f in os.listdir(fromDir):
        shutil.move(f'{fromDir}/{f}', toDir)

    print('All pictures restored in subdirectory')

    return


def DeleteFolder(Dir):
    os.rmdir(Dir)
    return


def SubDirectories(train_dir, subdir):
    check = 0
    train_dogs_dir = f'{train_dir}/dog'
    train_cats_dir = f'{train_dir}/cat'
    if os.path.isdir(train_dogs_dir):
        print('Dog Folder already Created')
        check += 2

    if os.path.isdir(train_cats_dir):
        print('Cat Folder already Created')
        check += 1

    if check == 3:
        if len(os.listdir(train_dogs_dir)) > 0 and len(os.listdir(train_cats_dir)) > 0:
            # Move Files
            # delete files
            if os.path.isdir(train_dir + subdir) == False:
                print('Subdirectories are already created')
                return

            else:
                DeleteFolder(train_dir + subdir)
        else:
            SeparateCD(train_dir + subdir, train_cats_dir, train_dogs_dir)
            DeleteFolder(train_dir + subdir)
    else:
        if check == 2:
            CreateFolders(train_cats_dir)

        elif check == 1:
            CreateFolders(train_dogs_dir)

        else:
            CreateFolders(train_cats_dir)
            CreateFolders(train_dogs_dir)

        SeparateCD(train_dir + subdir, train_cats_dir, train_dogs_dir)
        DeleteFolder(train_dir + subdir)


def restoreSubdirectories(train_dir, subdir):
    train_dogs_dir = f'{train_dir}/dog'
    train_cats_dir = f'{train_dir}/cat'

    CreateFolders(train_dir + subdir)
    MoveFiles(train_dir + subdir, train_cats_dir)
    MoveFiles(train_dir + subdir, train_dogs_dir)
    DeleteFolder(train_dogs_dir)
    DeleteFolder(train_cats_dir)

    print('\nAll files restored')
