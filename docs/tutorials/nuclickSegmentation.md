# Nuclick nuclei segmentation

The AI model is built based on the research paper titled *["NuClick: A Deep Learning Framework for Interactive Segmentation of Microscopy Images"](https://arxiv.org/abs/2005.14511)* authored by Navid Alemi Koohbanani, Mostafa Jahanifar, Neda Zamani Tajadin, and Nasir Rajpoot.

*Tutorial Created by [Subin Erattakulangara](www.subinek.com)*

## Overview
Segments individual nuclei provided the user give input for the center of the nuclei. Users can provide the nuclei center as a list by either manually entering the locations in the format [[x1,y1],[x2,y2],[xn,yn] or by adding the locations visually by using the polygon tool. One of the requirements while using the polygon tool is that there should be atleast 4 points and the polygon should be closed by double clicking at the last point.

### Using the Segment anything with DSA

1. Navigate to the DSA module `runCustomAIModel`.
![Navigate to DSA adapter](../media/show-histomicstk.gif)
&nbsp;

2. Select **"Segment Anything"** from the dropdown menu for AI models.
&nbsp;

3. Determine the Region of Interest (ROI) and submit the request.
   <!-- ![Select nuclick segmentation](../media/segment-anything.gif) -->