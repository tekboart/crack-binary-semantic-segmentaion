<div align="center">
  <p>
    <a align="center" href="" target="_blank">
      <img
        width="75%"
        style="border-radius: 20px;"
        src="images/20160222_164141 - visualization high quality.jpg"
      >
    </a>
  </p>
  <br>

  <div align="center">
      <a href="https://github.com/tekboart/">
          <img
            src="images/logos/github-gray.svg"
            width="3%"
          />
      </a>&nbsp;&nbsp;&nbsp;
      <a href="https://www.linkedin.com/in/kyan-bhr/">
          <img
            src="images/logos/linkedin-gray.svg"
            width="3%"
            style="border-radius: 5px !important; filter: invert(40%;"
          />
      </a>&nbsp;&nbsp;&nbsp;
      <a href="https://scholar.google.com/citations?user=r3xmjQUAAAAJ&hl=en">
          <img
            src="images/logos/googlescholar-gray.svg"
            width="3%"
          />
      </a>&nbsp;&nbsp;&nbsp;
      <a href="https://www.kaggle.com/tekboart">
          <img
            src="images/logos/kaggle-gray.svg"
            width="3%"
          />
      </a>&nbsp;&nbsp;&nbsp;
  </div>
</div>

<hr height="10">

# Crack Binary Segmentation Using Deep Learning Computer Vision
![Python](https://badges.aleen42.com/src/python.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-svg?style=flat&color=EE4C2C&logo=pytorch&logoColor=white&labelColor=gray)
![NumPy](https://img.shields.io/badge/NumPy-svg?style=flat&color=013243&logo=numpy&logoColor=white&labelColor=gray)
![Pandas](https://img.shields.io/badge/pandas-svg?style=flat&color=150458&logo=pandas&logoColor=white&labelColor=gray)
![Matplotlib](https://img.shields.io/badge/Matplotlib-svg?style=flat&color=65BAEA&label=&logoColor=white&labelColor=gray)
![PIL](https://img.shields.io/badge/Pillow-svg?style=flat&color=yellow&label=PIL&logoColor=white&labelColor=gray)
![Ray_Tune](https://img.shields.io/badge/Ray_Tune-svg?style=flat&color=028CF0&logo=ray&logoColor=white&labelColor=gray)


## Description
Used several semantic segmentation models (i.e., UNet++, FPN, DeepLabV3+) with different CNN encoders, pre-trained with 12M ImageNet dataset, to detect cracks in built environment images (e.g., bridges, infrastructures, pavement, etc.) with quite favorable results (See Figure [[1]](#1)).

<p align="center">
    <img width="90%" src="images/inference_testset_3.png">
    <p align="center"><b>Fig <a id="1">[1]</a> :</b> A few sample inference results of the Test set images.</p>
</p>




## Requirements
![Python](https://img.shields.io/badge/Python-%3D%3D_3.11.3-396D99.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%3D%3D_2.0.1+cu118-FF6F00.svg)
![Pandas](https://img.shields.io/badge/pandas-%3D%3D_2.0.3-150458.svg)
![NumPy](https://img.shields.io/badge/NumPy-%3D%3D_1.23.5-013243.svg)

- Please refer to the file `requirements.txt` for a comprehensive list of packages and their corresponding version.

## Project Dir Structure (only 2 level)
```bash
.
.
├── data
│   ├── testcrop
│   ├── testdata
│   └── traincrop
├── images
├── logs
├── models
├── outputs
│   ├── history
│   ├── hyperparams
│   ├── hyperparams_search
│   ├── Inferences
│   └── plots
├── reports
├── runs
├── temp
└── utils
    └── models


58 directories
```

## Contact
<!-- Unfortunately this repo is no longer actively maintained.  -->
Should you have any questions, feel free to contact TekBoArt @tekboart.


## License
<!-- Creative Common Licenses -->
<!-- "Creative Commons Attribution-NonCommercial-ShareAlike (CC-BY-NC-SA)" -->
Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

<!-- MIT License (can be used commercially) -->
<!-- Shield: [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) -->

- Refer to the file `LICENSE` for more information regarding the license of this repository.

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
