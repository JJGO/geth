
Survey of Parallel SGD Work
=====================

(Based on a previous survey done with J. Frankle)

Current state of the art of parallel SGD to look for opportunities to use instability analysis to improve time to train on models.


## Paper Idea

The general idea is to examine how early we can start doing _post-local SGD_ as opposed to standard large-batch training. The belief is that we can start from the iteration of _stability_.

## Papers that Propose Averaging as a Practical Technique


### As a Replacement for Parallel SGD

**Don't Use Large Mini-Batches, Use Local SGD** - Lin, Stich, Patel, Jaggi (EPFL) - ICLR 2020

- Shows that _Local SGD_ (train in parallel for K iterations and average) outperforms synchronous SGD for an equivalent number of iterations of processing the same sized mini-batches on each GPU.
- Proposes _Post-Local SGD_ in which you use large-batch training until the first learning rate drop and then switch to Local SGD.
- The most up-to-date empirical analysis of the kind of SGD with averaging that we're contemplating.
- Paper isn't as empirically rigorous as I'd like, and we can probably start running local SGD much earlier than they propose.

### For Federated Learning

**Communication-Efficient Learning of Deep Networks from Decentralized Data** -  McMahan et al (https://arxiv.org
/abs/1602.05629) - 2017

- The original Federated Learning paper.
- Each client has a set of data that is never uploaded to a server. Each client computes an update to the global model and only sends this update. A central server then updates the model.
- The paper suggests performing it in a synchronous parallel SGD fashion (just average the gradients from many workers into a single batch).
- They consider a hyperparameter where each client takes many steps and then the resulting models are averaged.
- They try our stability experiment with disjoint data for MNIST and find that it works well. We should cite this.

**Towards Federated Learning at Scale: System Design** -  Bonawitz et al. (Google) (https://arxiv.org
/pdf/1902.01046.pdf) - 2019

- Describes Google's system design around federated learning. Appears to still average all of the updates from the devices after each step of SGD to update the model.

### Pre Deep Learning Papers that Use Averaging

**Parallelized Stochastic Gradient Descent** -  Zinkevich et al. (Smola) (https://papers.nips.cc/paper/4006-parallelized-stochastic-gradient-descent) - 2010

- Proposes training in parallel using SGD and averaging at the end.

**Distributed Training Strategies for the Structured Perceptron** -  McDonald, Hall, and Mann (Google) (https://www.aclweb.org/anthology/N10-1069.pdf) - 2010

- For NLP: train many perceptrons on different subsets of the data and then use a weighted average of the
parameters at the end of training.

**Efficient Large-Scale Distributed Training of conditoinal Maximum Entropy Models** - McDonald, Mohri, et al. - (https://papers.nips.cc/paper/3881-efficient-large-scale-distributed-training-of-conditional-maximum-entropy- models) - 2009
- They show that "parameter mixtures over independent models use fewer resources and achieve comparable loss as compared to standard approaches."

### Modern Papers that Use Averaging

**On the Convergence Properties of a K-Step Averaging SGD Algorithm for Nonconvex Optimization** - Zhou and Cong (https://arxiv.org/abs/1708.01012) - 2017

- Looks at what we would call _local SGD_.
- Mainly examines convergence properties.
- Some analysis of the algorithm on VGG for CIFAR-10, although only compares to asynchronous SGD.
- At 32 steps in parallel, accuracy begins to drop off. This does not seem to support the conclusion that averaging replaces large batch training.

**Experiments on Parallel Training of Deep Neural Network Using Model Averaging** - Su, Chen, Xu - (https://arxiv.org/abs/1507.01239) - 2015

- Look into natural SGD and restricted boltzmann machines.
- No warmup on a single thread before parallelizing (as in some cited work - post-local SGD may not be especially novel)
- They use a telephone speech dataset.
- They get reasonable speedup, but there are no baselines.

**Efficient Decentraliezd Deep Learning by Dynamic Model Averaging** -  Kamp et al (https://arxiv.org/abs/1807.03210) - 2018

- They aim to solve the problem of training on distributed data sources in communication-limited environments. (How realistic is this problem? They claim this is useful for self-driving cars that learn online. Sounds safe.)
- Uses a gossip-style algorithm where models are compared local and averaged only when they drift apart. This avoids global communication and reduces the number of synchronization operations.
- Experiments are on a CNN for MNIST.
- Empirical evaluation is generally lacking.

**Parallel Training of DNNs with Natural Gradient and Parameter Averaging** - Povey et al (https://arxiv.org/abs/1410.7455) - ICLR 2015

- `Kaldi` speech framework for training neural networks.
- Average every minute or two.
- Doesn't work well with SGD; needs an approximate NGD optimizer.
- As they add workers, they increase the learning rate linearly. Their justification is rather vague, and it's unclear whether this is actually a good choice. Perhaps it can correct for instability from large-batch training?
- Includes a maximum parameter change per iteration to avoid explosion. In general, seems very hacky.
- Not much in the way of baselines, and not really comparable to our benchmarks.


### Papers that Show Convergence Rates for Parallel SGD

**Parallel SGD: When Does Averaging Help?** - Zhang, De Sa, Mitliagkas, Re (Stanford) - Arxiv 2016

- Proves that local SGD converges in convex settings, and it leads to faster convergence in the face of gradient variance.
- Proves that frequent averaging is important in non-convex settings.
- Runs a few experiments on Lenet-5 of averaging every ten iterations and averaging all at once at the end of training.

**Local SGD Converges Fast and Communicates Little** - Stich (EPFL) - ICLR 2019
- Proves that local SGD converges as fast as mini-batch SGD on convex problems.
- No experiments.

**Parallel Restarted SGD with Faster Convergence and Less Communication: Demystifying Why Model Averaging Works for Deep Learning** - Yu, Yang, Zhu (Alibaba) - AAAI 2019

- Mainly about convergence rates for local SGD.
- Includes one plot that shows that local SGD outperforms parallel SGD in wall-clock time using Horovod.

**NOTE: All work in this area consistently claims that local SGD is a popular technique. However, the only citations to that effect are _When Does Averaging Help_ and a couple of papers from 2010 that are pre-neural net.**


### First Pass through the Literature

**Parallel SGD: When Does Averaging Help?** Zhang et al. (https://arxiv.org/abs/1606.07365) - 2016

- They articulate a framework where independent workers train a model and average after N iterations. If N is 1, then it is equivalent to having a larger mini-batch (batchnorm notwithstanding). If N is larger, you may get better hardware efficiency but worse statistical efficiency.
- Their results are theoretical in nature.
- For convex problems, convergence rates depend on gradient variance. Higher variance requires more frequent averaging.
- For non-convex problems, more frequent averaging leads to better results.
- Experiments are on least squares and logistic regression.

**Distributed Training Across the World** Zhu et al. (Song Han) (https://openreview.net/forum?id=SJeuueSYDH) - 2019

- Recently rejected from ICLR, but a good place to start on looking at modern techniques in this area.
- Good related work section with a lot of recent and less recent work in this area.
- They apply two techniques:
	- i. Time-delayed updates. Only synchronize the gradients every `n` steps. When you do so, retroactively go back `n` steps and correct the gradients you applied to include all of the gradients. This still leaves some error, but it intuitively reduces the error you would otherwise accumulate as each worker diverges. Since you now have `n` steps of computation before each sync, you more effectively pipeline communication and computation.
	- ii. Temporally sparse updates. Use the gradients accumulated across multiple steps to reduce the amount of communication necessary.
- Results: They can tolerate 300ms latencies while losing only 0.8 percentage points of accuracy on ResNet-50/ImageNet.
- Paper was rejected from ICLR for not being especially novel and for relying heavily on gradient sparsification (they sparsify to 1%).

**Hovorod: Fast and Easy Distributed Deep Learning in TensorFlow** Sergeev and Del Balso (Uber) (https://arxiv.org/abs/1802.05799) - 2018

- An addon to TensorFlow that makes distributed training easier. (TensorFlow support is otherwise pretty terrible)
- Problem: TensorFlow's standard distribued training API was scaling at about 0.5x for large numbers of GPUs
(e.g., 128).
- Standard (synchronous) data parallel approach to training (FB's approach):
	- Each GPU reads a copy of the data, runs it through the model, and computes the gradients.
	- The GPUs exchange the gradients and average them.
	- Each GPU updates its copy of the model with the averaged gradients.
- Parameter server approach to training (TF's approach):
	- Separate "parameter server" role that averages the gradients.
	- Can have multiple parameter servers that average some of the gradients.
	- This avoids the all-to-all communication of the distributed approach, but introduces additional latency.
	- It's tricky to find the right number of parameter servers - too many increases the amount of communication; too few becomes a computational bottleneck (when averaging).
- The ring-allreduce approach (http://andrew.gibiansky.com/) allows you to eliminate a parameter server when averaging the gradients.
- These changes make it possble to reach 80-90% of linear scaling.
- *Importantly, this paper still uses a synchronous SGD approach. In other words, this approach is still exact. Our remaining opportunity to improve performance is in this final 10% by eliminating unnecessary communication.*

**Democratizing Production-Scale Distributed Deep Learning** Ma et al. (Apple) (https://arxiv.org/abs/1811.00143) - 2018

- Basically builds off of Horovod. Nothing algorithmically new there.

**A Hitchhiker's Guide On distribured Training of Deep Neural Networks** Chahal et al. (https://arxiv.org/abs/1810.11787) - 2018

- A survey of distribued deep learning. Nothing notable, but it's quite comprehensive.

**Parallel Restarted SGD with Faster Convergence and Less Communication: Demystifying Why Model Averaging Works for Deep Learning** Yu et al. (Alibaba) (https://arxiv.org/abs/1807.06629) - 2018

- Parallel SGD with averaging is apparently a common technique according to this paper. With that said, I haven't seen it mentioned in the distributed SGD literature to this point. The paper claims that it is a mystery as to why this method works, and purports to provide a theoretical explanation. Perhaps we have an explanation via instability.
- This work is theoretical in nature, proving a convergence bound for parallel SGD with averaging in nonconvex settings.
- They include one experiment on Resnet-20 that shows that averaging works reasonably well. They don't seem to have a warmup period as we would suggest.

**Cooperative SGD: A Unified Framework for the Design and Analysis of Communication-Efficient SGD Algorithms** Wang and Joshi (CMU) (https://arxiv.org/abs/1808.07576) - 2018

- They put together the many distributed frameworks (model averaging, elastic averaging, data parallel SGD with local updates, etc.) into a single theoretical framework and analyze convergence rates.
- Section 2 has a nice summary of various ways of doing imprecise distributed training.

**Parallelized Stochastic Gradient Descent** Zinkevich et al. (Smola) (https://papers.nips.cc/paper/4006-parallelized-stochastic-gradient-descent) - 2010

- Proposes training in parallel using SGD and averaging at the end.
- Pre deep learning.

**Distributed Training Strategies for the Structured Perceptron** McDonald, Hall, and Mann (Google) (https://www.aclweb.org/anthology/N10-1069.pdf) - 2010

- For NLP: train many perceptrons on different subsets of the data and then use a weighted average of the parameters at the end of training.
- Pre neural net.

**Efficient Large-Scale Distributed Training of conditoinal Maximum Entropy Models** McDonald, Mohri, et al. (https://papers.nips.cc/paper/3881-efficient-large-scale-distributed-training-of-conditional-maximum-entropy- models) - 2009

- They show that "parameter mixtures over independent models use fewer resources and achieve comparable loss as compared to standard approaches."
- Pre neural net.

**Web-Scale named Entity Recognition** Whitelaw, et al. (https://dl.acm.org/doi/10.1145/1458082.1458102) - 2008

- Another pre-neural net paper. Can't access ACM DL at FB right now.

**How to Scale Distributed Deep Learning?** Jin, Yuan, Iandola, Keutzer (https://arxiv.org/abs/1611.04581) - 2016

- Frames distributed SGD as either synchronous (allreduce) or asynchronous with a parameter server.
	- Synchronous SGD problem: stragglers, underutilization, failures.
	- Asynchronous: communication bottlenecks with the parameter server.
- Benchmark a Resnet on ImageNet with different settings of synchornous and asynchronous SGD.
- Also propose their own "gossiping asynchronous SGD"
- Findings: Async converges faster for larger step sizes and on smaller numbers of nodes while sync does better at larger scales.

**CoCoA: A General Framework for Communication-Efficient Distributed Optimization** Smith et al. (https://arxiv.org/abs/1611.02189) - 2016-18

- Appears to be a non-neural net distributed optimization system.

**Experiments on Parallel Training of Deep Neural Network Using Model Averaging** Su et al. (https://arxiv.org/abs/1507.01239) - 2015

- Train the networks in parallel, average every few mini-batches.
- They use NGD and RBM pretraining. Seems like an old paper.
- They use the 300 hour switchboard telephone task (?)

**Efficient Decentralized Deep Learning by Dynamic Model Averaging** Kamp et al. (https://arxiv.org/abs/1807.03210) - 2018

- Goal is to use this for privacy preservation in a proto-federated learning way. (They mention federated learning explicitly.)
- They propose a dynamic averaging scheme to only communicate when models get out of sync.
- Nothing other than MNIST.

**SWAP: Stochastic Weight Averaging in Parallel - Large Batch Training that Generalizes Well** Gupta et al. (Apple) (http://people.eecs.berkeley.edu/~vipul_gupta/swap_large_batch_training.pdf) - 2020

**NEED TO TRACE CITES THIS PAPER MAKES**

- Problem: large batch training gets worse performance.
- Solution: do large batch training and then train in parallel from there on smaller batch sizes at lower learning rates, then average the resulting models.
- Slight improvements in performance from doing so (tenths of percentage points)
- Unimpressive - very similar to Stochastic Weight Averaging (from Andrew Wilson's group)

**Deep learning with Elastic Averaging SGD** Zhang et al (LeCun) (https://arxiv.org/abs/1412.6651) - 2014

**Parle: parallelizing stochastic gradient descent** Chaudhari et al. (https://arxiv.org/abs/1707.00424) - 2017

**Collaborative Deep Learning Across Multiple Data Centers** Xu et al. (https://arxiv.org/abs/1810.06877) - 2018

**Local SGD with Periodic Averaging: Tighter Analysis and Adaptive Synchronization** Haddadpour et al. (http://papers.nips.cc/paper/9288-local-sgd-with-periodic-averaging-tighter-analysis-and-adaptive-synchronization) - 2019

**Collaborative Deep Learning in Fixed Topology Networks** Jiang et al. (http://papers.nips.cc/paper/7172-collaborative-deep-learning-in-fixed-topology-networks) - 2017

**Don't Use Large Mini-Batches, Use Local SGD** Lin et al. (https://arxiv.org/abs/1808.07217) - 2019

- Recently accepted into ICLR
- Does what we would propose: explores local SGD (train for k iterations and then average) and shows that it outperforms large-batch training.
- Results aren't particularly convincing - they didn't do appropriate hyperparameter search for large-batch training on CIFAR, so results are questionable. They just linearly scale the learning rate (from a relatively high starting point) and choose five epochs for warmup (which is what Goyal et al. do for ImageNet). In general, baselines are unfair.
- They propose post-local sgd in which they run sync-SGD with large batches for the first part of training and then switch to local SGD after the first learning rate drop. We believe we can make this switch earlier.
- Results are a mess. Poorly presented, confusing, hard to make sense of. I think we could write this paper better.

**Parallel training of DNNs with Natural Gradient and Parameter Averaging** Povey et al. (https://arxiv.org/abs/1410.7455) - 2015

**Local SGD Converges Fast and Communicates Little** Stich (https://arxiv.org/abs/1805.09767) - 2019

**On the convergence properties of a K-step averaging stochastic gradient descent algorithm for nonconvex optimization** Zhou and Cong (https://arxiv.org/abs/1708.01012) - 2018

**Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour** Goyal et al. (https://arxiv.org/abs/1706.02677) - 2018

**Communication Compression for Decentralized Training** Tang et al. (https://papers.nips.cc/paper/7992-communication-compression-for-decentralized-training.pdf) - 2018

**Bringing HPC Techniques to Deep Learning** Gibiansky (http://andrew.gibiansky.com/) - 2017

- Proposes the Ring-AllReduce method for synchronizing gradients. Everyone now uses this method.

**ChainerMN: Scalable Distributed Deep Learning Framework** Akiba et al. (https://arxiv.org/abs/1710.11351) - 2017

**SparkNet: Training Deep Networks in Spark** Moritz et al. - 2015

**Pentuum: A New Platform for Distributed Machine Learning on Big Data** Xing et al. (http://www.cs.cmu.edu/~seunghak/petuum_kdd15.pdf) - 2015

**AdaComp : Adaptive Residual Gradient Compression for Data-Parallel Distributed Training** Chen et al. (https://arxiv.org/abs/1712.02679) - 2017

**Highly Scalable Deep Learning Training System with Mixed-Precision: Training ImageNet in Four Minutes** Jia et al. (https://arxiv.org/abs/1807.11205) - 2018

**Hogwild: A lock-free approach to parallelizing SGD** Niu, Recht, Re (https://papers.nips.cc/paper/4390-hogwild-a-lock-free-approach-to-parallelizing-stochastic-gradient-descent.pdf) - 2012

**Large Scale Distributed Deep Networks** Dean et al. (https://papers.nips.cc/paper/4687-large-scale-distributed-deep-networks) - 2012

**Scaling Distributed Machine Learning with the Parameter Server** Li et al. (https://www.usenix.org/conference/osdi14/technical-sessions/presentation/li_mu) - 2014

**Taming the Wild: A Unified Analysis of Hogwild!-Style Algorithms** De Sa, Zhang, Olkotun, Re (https://arxiv.org/abs/1506.06438) - 2015

**Optimizing Network Performance for Distributed DNN Training on GPU Clusters: ImageNet/AlexNet Training in 1.5 Minutes** Sun et al. (https://arxiv.org/abs/1902.06855) - 2019

**Federated Learning: Challenges, Methods, and Future Directions** Li et al. (CMU) - 2019

- A great overview of the current SOTA methods for doing federated learning, including model averaging every few steps.
