# GlobeNet-Keras
Neural Nets for understanding Weather Events from Satellite Observation

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Input
A complex images consisted of
* stacked up with multiple vision channels
* delivered from remote sensors

### Candidates
#### 4ch IR Image
* source: COMS-1 MI (https://www.wmo-sat.info/oscar/instruments/view/283)
* size (NHWC): (?, 1544, 1934, 4)  

## Execution (Windows 10 x64)
* Run Anaconda Prompt
```
# activate tensorflow 1.10.0
> activate tf110
> cd globenet-keras
> set PYTHONPATH=%CD%
> python runtime\regressor.py
(...)
```
