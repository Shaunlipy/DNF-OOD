# DNF-OOD

This repo is for Deep Neural Forest for OOD detection.

## 1. Source Datasets
### HAM10000 dataset [1]
this dataset contains 7 lesion classes: NV, MEL, DF, AK, SCC, BCC, VASC
### DermNet [2]
this dataset contains 23 lesion classes. [link to Drive](https://drive.google.com/file/d/1E7Z_ub-UErWDyN1fFWJjEzQjaKEZ0MZF/view?usp=sharing)
## 2. OOD sets
### Unseen disease detection
please refer to our paper for the leave-one-out experiment on HAM10000, and treating 4 diseases as OOD data on DermNet.
### HAM10000-crop
please download the data.
### Gradient-based heatmap cropping
[75-gradient-cropping](https://drive.google.com/file/d/1VSGqbCHp6RDjW-qQQp6Sa6xHYwytuP2Y/view?usp=sharing), [50-gradient-cropping](https://drive.google.com/file/d/1Edh98Uu_TlUmWktqfI9zIA4ziitIh3VP/view?usp=sharing), [25-gradient-cropping](https://drive.google.com/file/d/1tKLZ2f9Y0TQpzbDDVP6XHB8NCfX0K5TV/view?usp=sharing)

## 3. Code for OOD detection




## References
[1] Tschandl, Philipp, Cliff Rosendahl, and Harald Kittler. "The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions." Scientific data 5 (2018): 180161.

[2] Oakley, Amanda. "DermNet new zealand." (2016).
