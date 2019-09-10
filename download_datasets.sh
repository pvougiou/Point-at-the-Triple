#!/bin/bash

# Downloads and uncompresses all the files for the Full corpus.
wget -O Full.zip https://www.dropbox.com/s/w86awmtwfeg9rb2/Full.zip?dl=1
unzip -o Full.zip
rm Full.zip
echo All files for the Full corpus have been downloaded and un-compressed successfully.
