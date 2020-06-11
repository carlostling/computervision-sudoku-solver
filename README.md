# computervision-sudoku-solver
A Python program that uses computer vision and machine learning to solve any Sudoku puzzle from an image.
Digit detection is done with a CNN trained on the MNIST dataset - the saved model and the script to train it is located in this repository.


# Demo
This demo showcases an solution of the program on this initial image:

![Alt text](demo_images/initial.jpg?raw=true "Initial image")

The solution:

![Alt text](demo_images/solve.png?raw=true "Initial image")



### Steps for finding solution
* After turning the image grayscale, we blur the it to smooth out noise. Makes extracting of lines easier.
* Threshold the image leaving only blacks and whites.
* The image is inverted to help identify black borders.
* Dilation is applied to fill in gaps in lines and digits
* The largest contour (border of puzzle) is found in order to find the Sudoku in the image
* The image is warped and a grid is inferred
* Each digit is extraced from the grid and identified using a CNN trained on the MNIST dataset
* The digits get passed so a backtracking algorithm solving the puzzle which is the presented on the warped version of the original image




![Alt text](demo_images/demo.gif?raw=true "Initial image")

