## Expressive Body Capture: 3D Hands, Face, and Body from a Single Image

[[Paper Page](https://smpl-x.is.tue.mpg.de/)] [[Paper](https://ps.is.tuebingen.mpg.de/uploads_file/attachment/attachment/497/SMPL-X.pdf)]
[[Supp. Mat.](https://ps.is.tuebingen.mpg.de/uploads_file/attachment/attachment/498/SMPL-X-supp.pdf)]

![SMPL-X Examples](./images/teaser_fig.png)


## Table of Contents
  * [License](#license)
  * [Description](#description)
  * [Dependencies](#dependencies)
  * [Example](#example)
  * [Citation](#citation)
  * [Contact](#contact)


## License

Software Copyright License for **non-commercial scientific research purposes**.
Please read carefully the [terms and conditions](https://github.com/vchoutas/smplx/blob/master/LICENSE) and any accompanying documentation before you download and/or use the SMPL-X/SMPLify-X model, data and software, (the "Model & Software"), including 3D meshes, blend weights, blend shapes, textures, software, scripts, and animations. By downloading and/or using the Model & Software (including downloading, cloning, installing, and any other use of this github repository), you acknowledge that you have read these terms and conditions, understand them, and agree to be bound by them. If you do not agree with these terms and conditions, you must not download and/or use the Model & Software. Any infringement of the terms of this agreement will automatically terminate your rights under this [License](./LICENSE).

## Description

This repository contains the fitting code used for the experiments in [Expressive Body Capture: 3D Hands, Face, and Body from a Single Image](https://smpl-x.is.tue.mpg.de/).
A sample command for executing the code is:
```Shell
python smplifyx/main.py --config cfg_files/fit_smplx.yaml 
    --data_folder DATA_FOLDER 
    --output_folder OUTPUT_FOLDER 
    --visualize="True/False"
    --model_folder MODEL_FOLDER
    --vposer_ckpt VPOSER_FOLDER
```
where the `DATA_FOLDER` should contain two subfolders, *images*, where the
images are located, and *keypoints*, where the OpenPose output should be
stored.

 
## Dependencies

Follow the installation instructions for each of the following before using the
fitting code.

1. [PyTorch](https://pytorch.org/)
2. [SMPL-X](https://github.com/MPI-IS/smplx)
3. [V-Poser](https://github.com/nghorbani/HumanBodyPrior)

### Optional Dependencies

1. [PyTorch Mesh self-intersection](https://github.com/MPI-IS/torch-mesh-isect) for interpenetration penalty
1. [Trimesh](https://trimsh.org/) for loading triangular meshes
1. [Pyrender](https://pyrender.readthedocs.io/) for visualization

## Citation

If you find this Model & Software useful in your research we would kindly ask you to cite:

```
@inproceedings{SMPL-X:2019,
  title = {Expressive Body Capture: 3D Hands, Face, and Body from a Single Image},
  author = {Pavlakos, Georgios and Choutas, Vasileios and Ghorbani, Nima and Bolkart, Timo and Osman, Ahmed A. A. and Tzionas, Dimitrios and Black, Michael J.},
  booktitle = {Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
  year = {2019}
}
```

## Contact

For questions about our paper or code, please contact [Vassilis Choutas](vassilis.choutas@tuebingen.mpg.de).
