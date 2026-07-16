Technical Challenges:

Dataset challenges:
- in the process of downloading the dataset and caching their features, some of the cached features ended up being corrupted, leading to spikes in the loss and derailing training





Baseline:
- training a trasnformer model can be dangerous to train, since choosing the right learning rate can lead to spikes in training as well 
- changed the model to be pre layer norm instrad of post layer norm due to training instability issues
 
results:
- the model looks like it is on the verge overfitting the data, as the training loss begins to diverge from the validation , but the validation and training accuracy are about the same
- the baseline took aroun 10 minutes to train for ten epochs across the whole dataset
- model achieves around 46%, which is impressive, since we are training a realtively small model (10 MB) and getting half the question right on I think relatively hard tasks of visual question answering.

Primary:
- the primary model requires a custom CUDA kernel in order to parallelize the rwkv "attention" calculation, and requires some extra debugging
- primary model is around 11 mb and can be scaled further

resutls
- the model
- the model overfits a lot on the data, and the validation error starts increasing while the training loss keeps decreasing, the vaidatoin and training accuracy are pretty much the same. this would lead to me look 
- it took around 10 minutes, (the same amount of time as the baseline) to train per epoch, which makes some sense, since the gain in attention calculation compared to the transformer is offset by the extra computation used for the spatial bridge compared to the MLP used on the baseline
- the model achieves an accuracy of 47%, almost identical to the baseline, which is somewhat surprising, considering our spatial bridge should outperform the MLP, but this can be further explained by the overfitting cutting back on the perfromance gain, and for future testing, it would be interesting to look into regularizatoin techniques and train for longer epochs.

Feasability/future steps:
- the project seems more than feasibile as we are able to see pretty impressvie results from relatively small models (46% on models the size of 10mb)
    - further experiments will be hyperparameter tune on model complexity and size, since the models are very much within the 100 mb limit
- further experiments should be to incorporate teacher forcing (autoregressive decoding) rather then just classification on the top 1000 answers, since it would further demonstrate time efficiency of inference and training using the rwkv model compared to the transformer model