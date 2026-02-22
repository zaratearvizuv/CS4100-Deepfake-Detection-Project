# Comparing Deepfake Detection Methods

## Overall Goal
Overall goal: Detect deepfakes

## Research Question
Which deepfake detection method generalizes best across different GANs?

How will it be done: 
1) Download datasets that generated images from different GANs.
2) Build multiple deepfake detectors using different methods:
- CNN (ResNet)
- CNN (No pre-training)
- DCT + Gradient Boosting
- Noise Pattern Analysis
- DIRE
3) Train each detector on the same dataset
4) Test each detector on the same GAN it was trained on. Then test each detector on different GANs it has not seen to see if it will generalize
5) Analyze results and see which method was more accurate and if we can combine methods to better detect deepfakes.


## Helpful Vocab
- GAN: Generative Adversarial Network. Two models are fighting to fool each other. One model (discriminator) will act as a judge, looking at images and thinking "is this real? or is it fake?" while the other model (generator) generates fake images from random noise that may look like tv static. The generator will continue to produce fake images until it fools the discriminator into believing it is a real image. The end result is the generator itself that has been trained to fool the discriminator. We now have something to generate realistic fake image.

- ResNet: Residual Network. CNN model that's already been trained on 14 million images to recognize lots of different things. ResNet(Number Here) is how many layers the CNN model has. More layers mean it can learn more complex patterns