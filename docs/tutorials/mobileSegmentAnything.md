# Mobile Segment anything

The Mobile Segment Anything Model is constructed upon the foundation of the Facebook SAM model. It serves as a lightweight iteration of SAM, designed to empower edge devices to seamlessly execute this model on their CPUs. The model's publication is rooted in the research paper titled *[Mobile Segment Anything](https://arxiv.org/pdf/2306.14289.pdf)*, authored by Chaoning Zhang, Dongshen Han, Yu Qiao, Jung Uk Kim, Sung-Ho Bae, Seungkyu Lee, and Choong Seon Hong.

*Tutorial Created by [Subin Erattakulangara](www.subinek.com)*

## Overview
The MobileSAM model empowers users to effortlessly segment entities within images through a simple click. This lightweight variant of the SAM, aptly named MobileSAM, boasts a remarkable reduction in size, being more than 60 times smaller while maintaining performance levels equivalent to the original SAM.

In terms of inference speed, MobileSAM demonstrates remarkable efficiency. On a single GPU, it processes images in approximately 10 milliseconds per image, with 8 milliseconds allocated to the image encoder and 4 milliseconds to the mask decoder.

### Using the Segment anything with HIstomicsTK

1. Navigate to the DSA module `runCustomAIModel`.

![Navigate to DSA adapter](../media/show-histomicstk.gif)
&nbsp;

2. Select **"Mobile Segment Anything"** from the dropdown menu for AI models.
&nbsp;

3. Determine the nuclei you wan't to select.

![Select segment anything]()