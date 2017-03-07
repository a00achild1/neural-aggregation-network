#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/attention_block_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void AttentionBlockLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  N_ = 1;
  K_ = bottom[0]->count(1);
  if (bottom.size() < 2) {
    if (this->blobs_.size() > 0) {
      LOG(INFO) << "Skipping parameter initialization";
    } else {
      this->blobs_.resize(1);
      // Initialize the weights
      vector<int> weight_shape(2);
      weight_shape[0] = K_;
      weight_shape[1] = N_;
      this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
      // fill the weights
      shared_ptr<Filler<Dtype> > filler(GetFiller<Dtype>(
          this->layer_param_.attention_block_param().filler()));
      filler->Fill(this->blobs_[0].get());
    }
    this->param_propagate_down_.resize(this->blobs_.size(), true);
  }

  LayerParameter softmax_layer_param_(this->layer_param_);
  softmax_layer_param_.set_type("Softmax");
  SoftmaxParameter* softmax_param_ = softmax_layer_param_.mutable_softmax_param();
  softmax_param_->set_axis(0);

  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_layer_param_);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(&attention_);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);
}

template <typename Dtype>
void AttentionBlockLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int new_K = bottom[0]->count(1);
  CHECK_EQ(K_, new_K)
      << "Input size incompatible with inner product parameters.";
  // The first "axis" dimensions are independent inner products; the total
  // number of these is M_, the product over these dimensions.
  M_ = bottom[0]->count(0, 1);
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension N_.
  if (bottom.size() == 2) 
  {
    vector<int> kernel_shape;
    kernel_shape[0] = K_;
    kernel_shape[1] = N_;
    bottom[0]->Reshape(kernel_shape);
  }

  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);

  vector<int> top_shape = bottom[0]->shape();
  top_shape.resize(2);
  top_shape[1] = N_;
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void AttentionBlockLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* attention_data = attention_.mutable_cpu_data();
  Dtype* prob_data = prob_.mutable_cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* kernel = (bottom.size() < 2) ? this->blobs_[0]->cpu_data() : bottom[1]->cpu_data();
  
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_, (Dtype)1., 
    bottom_data, kernel, (Dtype)0., attention_data);
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1., 
    prob_data, bottom_data, (Dtype)0., top_data);
}

template <typename Dtype>
void AttentionBlockLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* prob_diff = prob_.mutable_cpu_diff();
  const Dtype* attention_diff = attention_.cpu_diff();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* kernel = (bottom.size() < 2) ? this->blobs_[0]->cpu_data() : bottom[1]->cpu_data();

  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_, (Dtype)1., 
    bottom_data, top_diff, (Dtype)0., prob_diff);
  softmax_layer_->Backward(softmax_top_vec_, propagate_down, softmax_bottom_vec_);
  if (this->param_propagate_down_[0]) {
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_, (Dtype)1., 
      bottom_data, attention_diff, (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
  }
  if (propagate_down[1]) {
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_, (Dtype)1., 
      bottom_data, attention_diff, (Dtype)0., bottom[1]->mutable_cpu_diff());
  }
  if (propagate_down[0]) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1., 
      attention_diff, kernel, (Dtype)0., bottom[0]->mutable_cpu_diff());
  }
}

#ifdef CPU_ONLY
STUB_GPU(AttentionBlockLayer);
#endif

INSTANTIATE_CLASS(AttentionBlockLayer);
REGISTER_LAYER_CLASS(AttentionBlock);

}  // namespace caffe
