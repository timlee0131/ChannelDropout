# ChannelDropout Experimentation

experimenting how dropping entire channels as a regularization technique might influence model performance in multivariate time series forecasting

## Motivation and methodology
ChannelDrop is a regularization technique designed for multivariate time series whereby channels containing their entire look-back window are dropped at random at fixed probabilities. It is analogous to regular dropout except for the fact that ChannelDrop drops entire channels whereas regular dropout (Hinton et al. 2014) performs element-wise drops to regularize the embedding space. As such, ChannelDrop is a much more aggressive regularization technique compared to regular dropout. Due to this, proper scaling of retained data to match the expected values is crucial to ensure stable gradient flow. The aim with ChannelDrop was to create a channel-wise regularizer that can add robustness and reliability to channel-dependent (CD) models and discourage them from creating an over-reliance on specific channel interactions during training to raise model performance and generalizability. 

## Observations from stuff I tried
1. standard channel drop
    - fixed dropout rates (every layer vs. only during intermediate layers): rates between 0.05 and 0.10 worked best. Higher rates resulted in poor performance.
    - dropout rate scheduler: generally better than fixed dropout rates.

    overall, standard ChannelDrop resulted in noticeably better performance on Weather but could only match the performance of the control model in every other benchmark dataset (ECL, Solar, Traffic, etc.)

2. adaptive channel isolation (ACI): Isolated channels were prevented from interacting with other channels in a channel-dependent manner. Instead, isolated channels were only allowed to perform self-attention (CI) while the other channels were allowed to interact with each other (CD). This way, ACI was deployed to act as a bridge between CI and CI methods where channels that need CD could continue to perform channel-wise interactions while channels that may benefit from CI would learn to be isolated. Overall, did not see any improvements from the control model, if anything ACI resulted in training instability and slightly worse performance.

3. standard channel drop as a way to make the model more resilient against missing data: inconclusive results, basically did not see any improvements from the control model
    - train on complete data, test on missing data
    - train on missing data, test on missing data

Overall, despite pretty extensive experimentation and hyperparameter tuning, ChannelDrop in all its forms did not resulted in convincing performance gain in both standard forecasting benchmarks or missing data resilience across widely acknowledged benchmarks. I think dropping entire channels may be an overly aggressive form of regularization. While ChannelDrop often resulted in comparable performance as the control model (which is curious on its own as to how a model that drops entire channels can match the performance of a model that retains all its data), it almost never resulted in convincing performance gain. The fact that ChannelDrop does not result in noticeable performance degradation makes me think this is not an entirely useless technique, however. So there could be something here, I just can't quite find it.