# Stroke Width Transform

A test implementation of the Stroke Width Transform algorithm
described in the paper [Detecting Text in Natural Scenes with Stroke Width Transform](https://www.microsoft.com/en-us/research/publication/detecting-text-in-natural-scenes-with-stroke-width-transform/) (PDF [here](paper/201020CVPR20TextDetection.pdf)):

> We present a novel image operator that seeks to find the value of stroke width for each image pixel, and demonstrate its use on the task of text detection in natural images. The suggested operator is local and data dependent, which makes it fast and robust enough to eliminate the need for multi-scale computation or scanning windows. Extensive testing shows that the suggested scheme outperforms the latest published algorithms. Its simplicity allows the algorithm to detect texts in many fonts and languages.

## Example

To run SWT with connected components against the `text.jpg` example image, execute

```bash
python main.py images/text.jpg
```

Given the following image ...

![](images/text.jpg)

... it will find these connected components:

![](.readme/connected-components.png)

## Conda environment

A conda environment is available in `environment.yaml`. To create and activate it, run

```bash
conda env create -f environment.yaml
conda activate swt
```

## Original publication

```bibtex
@InProceedings{epshtein2010detecting,
    author = {Epshtein, Boris and Ofek, Eyal and Wexler, Yonatan},
    title = {Detecting Text in Natural Scenes with Stroke Width Transform},
    year = {2010},
    month = {June},
    abstract = {We present a novel image operator that seeks to find the value of stroke width for each image pixel, and demonstrate its use on the task of text detection in natural images. The suggested operator is local and data dependent, which makes it fast and robust enough to eliminate the need for multi-scale computation or scanning windows. Extensive testing shows that the suggested scheme outperforms the latest published algorithms. Its simplicity allows the algorithm to detect texts in many fonts and languages.},
    publisher = {IEEE - Institute of Electrical and Electronics Engineers},
    url = {https://www.microsoft.com/en-us/research/publication/detecting-text-in-natural-scenes-with-stroke-width-transform/},
}
```

## License

The code in this repository is made available under the MIT license (see [LICENSE.md](LICENSE.md)).
