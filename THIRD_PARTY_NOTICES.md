# Third-party notices

Original work in this repository (code, tests, the explorer page and the written results) is
(c) Jane Rose Ferdinant and licensed under CC BY-NC-ND 4.0 (see `LICENSE`). The components
below are not part of that licence and keep their own terms. None of them is committed to this
repository: the images and the pretrained weights are downloaded at run time (archives are checked
against the SHA-256 digests below before use).

## TF Flowers photographs (data, downloaded at run time)

- Source: https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
  (archive SHA-256 `4c54ace7911aaffe13a365c34f650e71dd5bf1be0a58b464e5a7183e3e595d9c`,
  228,813,984 bytes, checked on the download used for the committed run).
- Licence: every photograph is licensed under Creative Commons Attribution 2.0 (CC BY 2.0),
  https://creativecommons.org/licenses/by/2.0/ . The archive ships a `LICENSE.txt` that lists the
  photographer and the Flickr URL of each of the 3,670 photos. Anyone who redistributes the
  images must keep that attribution; this repository does not redistribute them.
- What this repository publishes derived from them: only file names (`results/split_manifest.json`,
  which lists which photo went to which split) and aggregate training curves. No image pixels.
- Attribution: "Thanks to all the photographers listed in the archive's `LICENSE.txt`."

## MobileNetV2 architecture and ImageNet-pretrained weights (downloaded at run time)

- Architecture: Sandler, Howard, Zhu, Zhmoginov, Chen, "MobileNetV2: Inverted Residuals and Linear
  Bottlenecks", CVPR 2018, https://arxiv.org/abs/1801.04381
- Implementation and weights: `tf.keras.applications.MobileNetV2`, which downloads
  `mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_96_no_top.h5` from
  https://storage.googleapis.com/tensorflow/keras-applications/ at first use. The Keras
  Applications code is released under the Apache License 2.0
  (https://www.apache.org/licenses/LICENSE-2.0).
- The weights were trained on ImageNet (ILSVRC-2012, https://www.image-net.org/). ImageNet's terms
  restrict use of the image data to non-commercial research and educational purposes. This
  repository is itself non-commercial (CC BY-NC-ND 4.0) and only uses the weights for research
  measurement. Check the current Keras Applications and ImageNet terms before any commercial use.

## EuroSAT RGB satellite images (data, downloaded at run time)

- Source: https://zenodo.org/records/7711810 (file `EuroSAT_RGB.zip`, 94,658,721 bytes, SHA-256
  `b4f5b234ecb7d7ff9c6cddb046543b4717c53fd6e9815be6c0e80cc614f51b90`; MD5 `f46e308c4d50d4bf32fedad2d3d62f3b`
  matches the digest published on the Zenodo record). Project page: https://github.com/phelber/EuroSAT
- Licence: MIT (stated on the Zenodo record and in the project repository). The full MIT text of the
  project repository is reproduced below, as that licence requires.
- Underlying imagery: Sentinel-2 satellite data of the Copernicus programme, which is free and open.
- Citation requested by the authors: P. Helber, B. Bischke, A. Dengel, D. Borth, "EuroSAT: A Novel
  Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification", IEEE Journal of
  Selected Topics in Applied Earth Observations and Remote Sensing, 2019; and "Introducing EuroSAT:
  A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification", IGARSS 2018.
- What this repository publishes derived from it: file names (`results/split_manifest_eurosat.json`)
  and aggregate training curves. No image pixels.

```
MIT License

Copyright (c) 2023 Patrick Helber

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Additional backbones and ImageNet-pretrained weights (downloaded at run time)

All are loaded through `keras.applications` (Keras Applications code: Apache License 2.0), which
downloads the weights from https://storage.googleapis.com/tensorflow/keras-applications/ (or the
`keras-applications` bucket) at first use. All weights were trained on ImageNet (ILSVRC-2012); the
ImageNet non-commercial terms described above apply to them as well.

| Backbone | Reference | Weight file used |
|---|---|---|
| ResNet50 | He, Zhang, Ren, Sun, "Deep Residual Learning for Image Recognition", CVPR 2016, https://arxiv.org/abs/1512.03385 | `resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5` |
| EfficientNetB0 | Tan, Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks", ICML 2019, https://arxiv.org/abs/1905.11946 | `efficientnetb0_notop.h5` |
| MobileNetV3Small | Howard et al., "Searching for MobileNetV3", ICCV 2019, https://arxiv.org/abs/1905.02244 | `weights_mobilenet_v3_small_224_1.0_float_no_top_v2.h5` |

## Software dependencies (installed by pip, not redistributed)

| Package | Licence |
|---|---|
| TensorFlow / Keras | Apache License 2.0 |
| NumPy | BSD 3-Clause |
| pandas | BSD 3-Clause |
| Pillow | HPND (MIT-CMU style) |
| pytest | MIT |
| matplotlib | Matplotlib licence (PSF-based, BSD-compatible) |
| ruff, mypy (development only) | MIT |

Each is used as an unmodified library; their licence texts are available from their package
distributions and project pages.
