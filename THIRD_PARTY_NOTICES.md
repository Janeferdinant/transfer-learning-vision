# Third-party notices

Original work in this repository (code, tests, the explorer page and the written results) is
(c) Jane Rose Ferdinant and licensed under CC BY-NC-ND 4.0 (see `LICENSE`). The components
below are not part of that licence and keep their own terms. None of them is committed to this
repository: the images and the pretrained weights are downloaded at run time.

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

## Software dependencies (installed by pip, not redistributed)

| Package | Licence |
|---|---|
| TensorFlow / Keras | Apache License 2.0 |
| NumPy | BSD 3-Clause |
| pandas | BSD 3-Clause |
| Pillow | HPND (MIT-CMU style) |
| pytest | MIT |

Each is used as an unmodified library; their licence texts are available from their package
distributions and project pages.
