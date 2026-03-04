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

- DCT: Discrete Cosine Transform. This is a method used to compress images and is used in jpeg files.  
  - Context: An entire image is divided into multiple 8x8 pixel blocks. If the resolution of an image is not a multiple of 8, extra padding is added that usually extends the edges of the image's pixels up to the multiple of 8. Each pixel within the 8x8 block (one of the 64 pixels) is assigned a value representative of its brightness value where 0 is black and 255 is white in cases where the image is converted to greyscale. With color, it's simplly (0, 0, 0) for black and (255, 255, 255) values which are shown at an individual pixel within the 8x8 block. Next, we begin to work one 8x8 block at a time until we go through all 8x8 blocks.
  - Aiding Frequency Template: There is a separate template of 64 predetermined cosine wave patterns used to aid in this process. In this template, the top left is flat/uniform, indicative of a slow changing colors, a low frequency. The bottom right is complex, indicative of multiple harsh changing colors, a high frequency. Frequencies can be thought of like trying to name how rapidly the colors are changing when throwing confetti+glitter in a room, it's too often, there's a high frequency. If one color of confetti with no glitter is thrown in a room, the color changes slowly, a low frequency. On the template, each frequency is shown as a pattern of light and dark blocks rather than an explicit wave shape. The light blocks are where the wave peaks and dark is where it's at the lowest.
  - Applying DCT: 