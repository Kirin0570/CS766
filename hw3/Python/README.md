# CS766: Computer Vision, Spring 2024
## Homework 3

### Challenge 1b - `generateHoughAccumulator()`:
**Finetuning Hough Accumulator resolution:** 
To perform Hough Transform in theta-rho space, the number of bins used in each axis are as follows:

&theta;: 360
&rho;: 1800

The intuition is that each bin within the theta space represents 0.5 degrees, while each bin in the rho space equates to roughly 0.5 pixels in distance. Consequently, the theta space encompasses all potential orientations ranging from 1 degree to 180 degrees. In a similar vein, the maximum conceivable distance for a line corresponds to the length of the diagonal, which, for all these three images, is no more than 800 pixels. Therefore by leaving some margin for error, all possible distances from -800 pixels to +800 pixels will be captured at a level of approximately 1 pixel distance per bin.

**Voting Scheme:**
In the voting process, the votes are simply counted for each single bin. The result is good enough, so there is no need to vote for a small patch of bins.


### Challenge 1c - `lineFinder()`:
Given the Hough accumulator returned by `generateHoughAccumulator()` in Challenge 1b, a standard threshold is used to identify the peaks. The value of thresholds used for `hough_1.png`, `hough_2.png`, and `hough_3.png` are `120`, `68`, and `52` respectively.

### Challenge 1d - `lineSegmentFinder()`:


Utilize two pointers to navigate through each line in search of line segments. A point is considered part of a line segment if any of its neighboring points are marked on the edge image. The neighborhood is defined as an  `R * R` square centered around the point in question. For this procedure, we have selected `R` to be 3 pixels.