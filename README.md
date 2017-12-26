# Digit_recognition
Given an image of a simple math expression, recognize all the digits and operations, and solve the equation.

This works, aside from the following problems:

1. All 6's are missclassified as 5's

2. Images that have larger digits causes a stack overflow. 
I should use an iterative connected components algorithm with a queue to fix this.

Needed enhancements:

1. Use a differenct method of finding bounding boxes on non-fully connected digits.
This will allow for recognition of more sloppy handwriting.


