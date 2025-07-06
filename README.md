# My Implementation of [AlexNet](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)

## Key Insights
- ### Speeding up with ReLU
During it's era, the standard way to model a neuron's output was to use the `tanh(x)` function which squishes the real numbers to [-1, 1]. However, this paper discusses that a CNN using the **non-saturating** linearity `f(x) = max(0, x)` - a.k.a ReLU - reaches 25% training error on the CIFAR-10 dataset six times faster than one trained using `tanh(x)`.
- ### Parallel GPU Training
This was one of the seminal papers when it comes to using GPUs to train large neural networks. In particular, the highly optimized convolution operations used in compute graphics proved to be a powerful tool in training large CNNs. However, the paper also discussed a **potential** drawback when it comes to training the network in parallel using multiple GPUs. The authors had to choose a pattern of connectivity between the GPUs that'll balance training time and overall model performance. The one trick they used is to only have the GPUs share parameters between the 2nd and 3rd layers but other than that, the rest of the layers can only see the other parameters that reside on the same GPU. They compared the performance of a half-sized network on a single GPU and a half-sized network on two GPUs and found that the two GPU network actually performs better than the single GPU network (which technically had more parameters) with faster training time.
- ### Local Response Normalization (LRN)
The authors borrowed a concept from neuroscience called "lateral inhibition", which is a phenomenon in a real brain where stimulated neurons suppress the activity of nearby neurons. In the case of LRN, it ensures that a neuron only produces a very high output if it stands out significantly from its local channel neighbors. The hyperparameter they chose for the number of neighboring kernels to normalize across was 5, which reduced their top-1 and top-5 error rates by 1.4% and 1.2%, respectively.
- ### Overlapping Pooling
Instead of using standard pooling methods where the stride and size of the pooling kernels are equal, they found that using **overlapping** pooling, where the stride < size of pooling kernel makes it more difficult for the model to overfit. They chose a stride of 2 and a pooling kernel size of 3x3 at the end, which reduced the top-1 and top-5 error rates by 0.4% and 0.3%, respectively.
- ### PCA Data Augmentation
The authors used an unconventional method of data augmentation where they identified the principal components of the RGB pixel values. Intuitively, this allows us to encode the primary axes of variation in colors across images in the data. The main idea was: "object identity is invariant to changes in the intensity and color of the illumination".

## Todo
- Add Local Response Normalization (LRN)