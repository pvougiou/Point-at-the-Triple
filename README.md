# Point at the Triple
This repository accompanies the submission: "Point at the Triple: Generation of Text Summaries from Knowledge Base Triples" to Journal of Artificial Intelligence Research. It contains instructions about how you can download the corpora that we used in our experiments.

## Datasets
We used two datasets of aligned knowledge base triples from DBpedia with snippets of text. 

 * Biographies: triples aligned with Wikipedia biographies
 * Full: triples aligned with open-domain Wikipedia summaries (incl. biographies)

The first is the D1 dataset, which has been provided by [https://www.sciencedirect.com/science/article/pii/S1570826818300313](https://www.sciencedirect.com/science/article/pii/S1570826818300313). It can be downloaded by following the instructions at: [github.com/pvougiou/Neural-Wikipedian](https://github.com/pvougiou/Neural-Wikipedian). 

In order to download and un-compress the Full dataset in its corresponding folder `Full`, in a Unix shell environment execute: `sh download_datasets.sh`. The dataset folder contains two sub-folders:

* `data` contains the aligned dataset in binary-encoded `pickle` files. Each file is a hash table. Each hash table is a Python dictionary of lists.
* `utils` contains the dataset's supporting files, such as hash tables of the instance types and the labels of the entities. All the files are binary-encoded in `pickle` files.

[`dataset.ipynb`](dataset.ipynb) is an iPython Notebook that allows easier inspection of the Full dataset. It provides also further details regarding the structure of the dataset and the functionality of its supporting files.

## License
This project is licensed under the terms of the Apache 2.0 License.
