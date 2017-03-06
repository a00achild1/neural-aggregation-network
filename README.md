# Neural Aggregation Network for Video Face Recogtion

Not stable! Caution to use.

Paper:
J. Yang, P. Ren, D. Chen, et al. Neural aggregation network for video face recognition. arXiv preprint arXiv:1603.05474, 2016.

Usage:
    layer {
        name: "attention_block0"
        type: "AttentionBlock"
        bottom: "feature"
        top: "embedded_feature0"
        param {
            lr_mult: 1
        }
    }

    layer {
        name: "embedding_kernel1"
        type: "InnerProduct"
        bottom: "embedding_feature0"
        top: "embedding_kernel1"
        param {
            lr_mult: 1
        }
        inner_product_param {
            num_output: 1
            axis: 0
            weight_filler {
                type: "xavier"
            }
            bias_filler {
                type: "constant"
            }
        }
    }

    layer {
        name: "embedding_tanh"
        type: "Tanh"
        bottom: "embedding_kernel1"
        top: "embedding_kernel1"
    }

    layer {
        name: "attention_block1"
        type: "AttentionBlock"
        bottom: "feature"
        bottom: "embedding_kernel1"
        top: "embedded_feature1"
    }

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }
