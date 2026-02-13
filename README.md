# uhhhhh

Goal: We train separate deepfake detectors on individual GAN architecture and measure how well each detector generalizes on unseen GANS, ooking for patterns in cross-architecture transferability? I think?

## Helpful Vocab
- GAN: Generative Adversarial Network. Two models are fighting to fool each other. One model (discriminator) will act as a judge, looking at images and thinking "is this real? or is it fake?" while the other model (generator) generates fake images from random noise that may look like tv static. The generator will continue to produce fake images until it fools the discriminator into believing it is a real image. The end result is the generator itself that has been trained to fool the discriminator. We now have something to generate realistic fake image.