#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/attention_block_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void AttentionBlockLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* attention_data = attention_.mutable_gpu_data();
  Dtype* prob_data = prob_.mutable_gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  if (bottom.size() < 2)
  {
    const Dtype* kernel = this->blobs_[0]->gpu_data();
  }
  else
  {
    const Dtype* kernel = bottom[1]->gpu_data();
  }
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_, (Dtype)1., 
    bottom_data, kernel, (Dtype)0., attention_data);
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1., 
    prob_data, bottom_data, (Dtype)0., top_data);
}

template <typename Dtype>
void AttentionBlockLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* prob_diff = prob_->mutable_gpu_diff();
  const Dtype* attention_diff = attention_->gpu_diff();
  const Dtype* bottom_data = bottom[0]->gpu_data();

  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_, (Dtype)1., 
    bottom_data, top_diff, (Dtype)0., prob_diff);
  softmax_layer_->Backward(softmax_top_vec_, propagate_down, softmax_bottom_vec_);
  if (this->param_propagate_down_) {
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_, (Dtype)1., 
      bottom_data, attention_diff, (Dtype)0., this->blobs_[0]->mutable_gpu_diff());
  }
  if (propagate_down[1]) {
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_, (Dtype)1., 
      bottom_data, attention_diff, (Dtype)0., bottom[1]->mutable_gpu_diff());
  }
  if (propagate_down[0]) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1., 
      attention_diff, kernel, (Dtype)0., bottom[0]->mutable_gpu_diff());
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(AttentionBlockLayer)

}  // namespace caffe
