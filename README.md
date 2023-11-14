# *shoulder*

[![PyPI Latest Release](https://img.shields.io/pypi/v/shoulder.svg)](https://pypi.org/project/shoulder/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This package uses a collection of machine learning models to detect anatomic landmarks on 3d models of shoulder bones and also generates patient specific coordinate systems. An stl of the shoulder bone of interest is all that is needed to get started. Currently only implemented for the humerus,  with expansion to the scapula in the future. Landmarks that *shoulder* can currently identify on the humerus are:

- canal 
- transepicondylar axis
- bicipital groove
- anatomic neck 


## Installation
compatible with python 3.10 and 3.11
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

