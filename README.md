# *shoulder*

[![PyPI Latest Release](https://img.shields.io/pypi/v/shoulder.svg)](https://pypi.org/project/shoulder/)
[![License](https://img.shields.io/pypi/l/shoulder.svg)](https://github.com/gspangenberg/shoulder/blob/main/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This package detects anatomic landmarks on stl files of shoulder bones and also generates patient specific coordinate systems. It is currently implemented for the humerus and will be extended to the glenoid in the future. Landmarks that *shoulder* can currently identify on the humerus are:

- bicipital groove
- canal 
- transepicondylar axis
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

    # calculate landmarks
    hum.canal.axis()
    hum.trans_epiconylar.axis()
    hum.anatomic_neck.plane()
    hum.bicipital_groove.axis()

    # apply coordinate sysytem
    hum.apply_csys_canal_transepiconylar()

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

