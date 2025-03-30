# fan-inverse-forward-methods

This repository contains Python implementations of the forward and inverse methods for analyzing debris and alluvial fan morphology. The **forward method** is based on the work of [Chen and Capart (2022)](https://doi.org/10.1016/j.cageo.2022.105228) and focuses on generating fan surfaces based on assumed elevation-distance relationships. The **inverse method**, which is a complement to the forward approach, extracts key fan elevation-distance profile from observed topography.

## Focus: Inverse Method

While the **forward method** generates fan surfaces using an assumed elevation-distance relationship, the **inverse method** focuses on reconstructing fan parameters from observed topography. This method is a new contribution, aimed at deducing fan characteristics by analyzing real-world data, making it a valuable tool for fan simulation.

## Key Algorithms Used

1. **Visibility Polygon Algorithm**: Used to obtain visibility polygon and compute shortest path distances from the apex. We implemented the [Python binding](https://github.com/tsaoyu/PyVisiLibity) of [VisiLibity 1](http://www.VisiLibity.org), originally developed by [Yu Cao](https://www.tsaoyu.com/). The implementation was modified in [Commit 0873a76](https://github.com/damiel-hub/fan-inverse-forward-methods/commit/0873a7673f9fa7b588234915ffc3a4eabcbf8882) to record the semi-apexes of the visibility polygon.

2. **Contouring Algorithm**: For extracting fan boundaries from topographic data we using the [scikit-image](https://github.com/scikit-image/scikit-image) module in Python.

3. **Point-in-Polygon Algorithm**: Determines which points lie inside a given polygon. We use the fast point-in-polygon algorithm developed by [Engwirda (2020)](https://github.com/dengwirda/inpoly-python) and described in [Kepner et al. (2020)](https://arxiv.org/abs/2005.03156). For easier installation, we use the version by [Hans Alemão](https://github.com/hansalemaos), which adds automatic compilation of the fast kernels from the original [inpoly-python](https://github.com/dengwirda/inpoly-python#fast-kernels) by Engwirda (2020).


## Repository Implementation

This repository contains Python code that translates the original MATLAB implementation of the **forward method** and develops the **inverse method** for reconstructing fan parameters from observed topography.


# Installation Instructions

## Prerequisite Software
1. Anaconda: [Install tutorial](https://www.jcchouinard.com/install-python-with-anaconda-on-windows/), 
[Download link](https://www.anaconda.com/download?utm_source=anacondadocs&utm_medium=documentation&utm_campaign=download&utm_content=installwindows).

2. Microsoft C++ Build Tools: [Download link](https://visualstudio.microsoft.com/visual-cpp-build-tools/)


3. SWIG: [Download link](https://www.swig.org/download.html)
    
    - **Note:** Make sure to add the SWIG path to your system's environment variables.

3. Visual Studio Code (VS Code): [Downlaod link](https://code.visualstudio.com/).

## Setup Guide (Step-by-Step)


1. **Download and extract required files**  
   - Download the file **`fan-inverse-forward-methods.zip`** from the website or directly using this [link](https://github.com/damiel-hub/fan-inverse-forward-methods/archive/refs/heads/main.zip).  
   - Extract the contents of the ZIP file to your preferred directory.

2. **Install dependencies**  
   - Open the **Anaconda Prompt** on your computer.  
   - Navigate to the directory **`fan-inverse-forward-methods\setup`** using the `cd` command. Replace `to_your_path` with the actual path to the extracted folder:  
     ```bash
     cd to_your_path\fan-inverse-forward-methods\setup
     ```  
   - **Note:** If the folder is on a different drive (ex, D:), switch to that drive first by typing the drive letter followed by a colon (`:`):  
     ```bash
     D:
     ```  
   - Run the following commands to create and set up the environment:  
     ```bash
     conda create --name fan python=3.11
     conda activate fan
     python -m pip install GDAL-3.10.1-cp311-cp311-win_amd64.whl
     python -m pip install -r requirements.txt
     python -m pip install PyVisiLibity-master\. 
     ```


5. **Run demo code**  
   - Open the VS Code.  
   - File > Open Folder > Select the folder **`fan-inverse-forward-methods`**.
    - Open the **`demo_fan_inverse_forward_methods.ipynb`** file from the left panel.
    - If you are prompted to install the Jupyter extension, click on the **Install** button.
    - If you are prompted to install the Python extension, click on the **Install** button.
    - Select the Python interpreter by clicking on the **Python** version in the upper right corner of VS Code, then choose the **`fan`** environment from the list.
    - If you are prompted to install the ipykernel, click on the **Install** button.
 
   - Run the `demo_fan_inverse_forward_methods.ipynb`

By following these steps, you can successfully set up and run inverse and forward analyses.

## How to Cite

If you use this project in your research or work, please give us credit by citing it with the following BibTeX entries:

### Inverse method: extracting the distance-elevation relationship
```
Chiu, Y.H. & Capart, H. (In process). *Inverse Computational Morphology of Debris and Alluvial Fans*. Retrieved from [https://github.com/damiel-hub/geometric-alluvial-fan].
```
### Forward method: simulating fan topography

```bibtex
@article{chenComputationalMorphologyDebris2022,
title = {Computational morphology of debris and alluvial fans on irregular terrain using the visibility polygon},
journal = {Computers & Geosciences},
volume = {169},
pages = {105228},
year = {2022},
issn = {0098-3004},
doi = {https://doi.org/10.1016/j.cageo.2022.105228},
url = {https://www.sciencedirect.com/science/article/pii/S0098300422001777},
author = {Tzu-Yin Kasha Chen and Hervé Capart},
keywords = {Morphology, Debris flow, Alluvial fan, Surface of revolution, Eikonal equation, Visibility polygon}
}
```


## Reference

- Chen, T.Y.K. & Capart, H. (2022). [Computational morphology of debris and alluvial fans on irregular terrain using the visibility polygon](https://doi.org/10.1016/j.cageo.2022.105228). *Computers & Geosciences*, *169*, 105228. 

- Kepner, J., Engwirda, D., Gadepally, V., Hill, C., Kraska, T., Jones, M., Kipf, A., Milechin, L., & Vembar, N. (2020). [Fast Mapping onto Census Blocks](https://arxiv.org/abs/2005.03156). *IEEE HPEC*. 

- Obermeyer, K.J., & Contributors. (2008). [VisiLibity: A C++ Library for Visibility Computations in Planar Polygonal Environments](http://www.VisiLibity.org). *v1*.

- van der Walt, S., Schönberger, J.L., Nunez-Iglesias, J., Boulogne, F., Warner, J.D., Yager, N., Gouillart, E., Yu, T., & the scikit-image contributors. (2014). [scikit-image: Image processing in Python](https://doi.org/10.7717/peerj.453). *PeerJ*, *2*, e453.