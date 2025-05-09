# *shoulder*

[![PyPI Latest Release](https://img.shields.io/pypi/v/shoulder.svg)](https://pypi.org/project/shoulder/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This package uses a collection of machine learning models to detect anatomic landmarks on 3d models of shoulder bones and also generates patient specific coordinate systems. An stl of the shoulder bone of interest is all that is needed to get started. Currently only implemented for the humerus,  with expansion to the scapula in the future. Landmarks that *shoulder* can currently identify on the humerus are:

- canal 
- transepicondylar axis
- bicipital groove
- anatomic neck 


## Installation
```
pip install shoulder
```

## Example
Start by using the example bone stl's located in "tests/test_bones"

    # pass stl into Humerus
    hum = shoulder.Humerus("tests/test_bones/humerus_left.stl")

    # apply coordinate sysytem
    hum.apply_csys_canal_transepiconylar()
    
    # calculate landmarks
    hum.canal.axis()
    hum.trans_epiconylar.axis()
    hum.anatomic_neck.points()
    hum.bicipital_groove.axis()

    # calculate metrics
    hum.radius_curvature()
    hum.neckshaft()
    hum.retroversion()

    # construct plot from above humeral bone with landmarks and coordinate system
    plot = shoulder.Plot(hum)
    plot.figure.show()

The output of the plot will appear as shown below with landmarks included and transformed from the original CT coordinate system to a coordainte system defined by the canal and transepicondylar axis.

![Plot of Example code above](https://raw.githubusercontent.com/gregspangenberg/shoulder/main/images/plot.png)


## Contributing 
Clone the repo, open the cloned folder containing the poetry.lock file, then install the development dependencies using poetry. 
```
poetry install --with dev
```

## Citation
If you use the *shoulder* package in your research, please cite the following papers:

```
@article{spangenberg2024automatic,
  title={Automatic bicipital groove identification in arthritic humeri for preoperative planning: A Random Forest Classifier approach},
  author={Spangenberg, Gregory W and Uddin, Fares and Faber, Kenneth J and Langohr, G Daniel G},
  journal={Computers in Biology and Medicine},
  volume={178},
  pages={108653},
  year={2024},
  publisher={Elsevier},
  issn={0010-4825},
  doi={https://doi.org/10.1016/j.compbiomed.2024.108653}
}

@article{SPANGENBERG2025,
  title={Automatic Determination of the Resection Plane for Shoulder Arthroplasty in Arthritic Humeri : A Deep Learning Model},
  journal={Journal of Shoulder and Elbow Surgery},
  year={2025},
  issn={1058-2746},
  doi={https://doi.org/10.1016/j.jse.2025.03.010},
  author={Gregory W. Spangenberg and Fares Uddin and Ahmed A. Habis and Kenneth J. Faber and G. Daniel G. Langohr}
}
```
