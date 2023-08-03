# Traveling-Salesman-solver
This repository contains a program created to solve the traveling salesman problem using a genetic algorithm
The program aims to solve a classic optimization problem known as the Traveling Salesman Problem (TSP). In the TSP, a salesman needs to visit a set of cities and return to the starting city while minimizing the total distance traveled.

The program uses a Genetic Algorithm (GA) to find an optimal solution to the TSP. A Genetic Algorithm is inspired by the process of natural selection and evolution. It starts with a random population of solutions and iteratively improves them by applying genetic operations like crossover and mutation.

The program provides a graphical user interface (GUI) to make it easier for users to interact with the optimization process. The GUI allows users to set parameters for the GA and visualize the progress of the algorithm.

When the program starts, it displays a window with input fields to provide a data file containing the city coordinates and various parameters for the GA, such as population size, number of generations, tournament size, and mutation rate.

Once the user clicks the "Run Optimization" button, the program starts the GA process to find the best solution to the TSP problem. It shows the progress of the algorithm, including the best solution found and the fitness value (total distance) of that solution.

The program uses matplotlib, a Python library for plotting, to visualize the best solution found so far and the fitness history over generations.

Users can also choose to display or hide the plots during the optimization process by selecting the "Show Plots" checkbox.

The program includes a button labeled "Parameter Information." When users click this button, a new window appears displaying information about each parameter used in the GA. This information helps users understand the role of each parameter in the optimization process.

The program runs in a loop, waiting for user interactions. Users can explore different parameter settings, try different data files, and observe how the GA performs in finding better solutions.

If users close the window or click the "Exit" button, the program terminates.

In summary, the program provides an easy-to-use interface for users to apply a Genetic Algorithm to solve the Traveling Salesman Problem. Users can experiment with different parameters and visualize the optimization process to better understand how the algorithm works and find near-optimal solutions to the TS
