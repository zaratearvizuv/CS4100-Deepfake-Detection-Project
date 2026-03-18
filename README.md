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
  - Context: An entire image is divided into multiple 8x8 pixel blocks. If the resolution of an image is not a multiple of 8, extra padding is added that usually extends the edges of the image's pixels up to the multiple of 8. Each pixel within the 8x8 block (one of the 64 pixels) is assigned a value representative of its brightness value where 0 is black and 255 is white in cases where the image is converted to greyscale. With color, it's simply (0, 0, 0) for black and (255, 255, 255) values which are shown at an individual pixel within the 8x8 block. Next, we begin to work one 8x8 block at a time until we go through all 8x8 blocks.
  - Aiding Frequency Template: There is a separate template of 64 predetermined cosine wave patterns used to aid in this process. In this template, the top left is flat/uniform, indicative of a slow changing colors, a low frequency. The bottom right is complex, indicative of multiple harsh changing colors, a high frequency. Frequencies can be thought of like trying to name how rapidly the colors are changing when throwing confetti+glitter in a room, it's too often, there's a high frequency. If one color of confetti with no glitter is thrown in a room, the color changes slowly, a low frequency. On the template, each frequency is shown as a pattern of light and dark blocks rather than an explicit wave shape. The light blocks are where the wave peaks and dark is where it's at the lowest.
  - Applying DCT: An 8x8 block from the image is grabbed. We also create a new empty 8x8 block to hold results. We'll go through each pattern from frequency template, multiplying each pixel by the value at the same cell location in the frequency template, then add up all the values from the multiplication to end up with a single result called a coefficient. We can move this coefficient to a new 8x8 block that will hold all our coefficient values as we move from pattern 1 to pattern 64, repeating the same process. After this point, we've created a new 8x8 block full of coefficients. The top-left coefficient represents the lowest frequency and the bottom-right coefficient represents the highest frequency. The order the coefficient block was filled in follows a diagonal zigzag pattern, where the path goes like:
    ```
    0  1  5  6  14 15 27 28
    2  4  7  13 16 26 29 42
    3  8  12 17 25 30 41 43
    9  11 18 24 31 40 44 53
    10 19 23 32 39 45 52 54
    20 22 33 38 46 51 55 60
    21 34 37 47 50 56 59 61
    35 36 48 49 57 58 62 63
    ```
    The pattern is set in a zigzag because it ensures the low frequency coefficients are visited first and high frequency last.  
    Quick recap: 
    - For each of the 64 patterns
      - Take all 64 pixel brightness values individually
      - Multiply each by the corresponding template cell value
      - Sum all 64 results together
      - That sum is your coefficient for that pattern
    
    This process is applied to every 8x8 block in the image.
  - Reading The Coefficients Block: The cell at position 0 is called the DC coefficient while the other 63 coefficients are called AC. DC stands for direct current while AC stands for alternating current. DC represents the average brightness of the image while AC holds frequency details throughout the image.

- Gradient Boosting: A collection of decision trees, similar to random forest. Unlike random forest which builds trees independently and averages their results, gradient boosting is additive where each new tree learns from the previous tree, gradually improving the overall prediction. It does this by minimizing the gradient of the loss function, a measure of how wrong the prediction is.