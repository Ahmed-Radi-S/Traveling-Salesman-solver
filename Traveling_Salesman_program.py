import random
import numpy as np
import matplotlib.pyplot as plt
import time
import PySimpleGUI as sg
import io
from PIL import Image
import threading


# Set the PySimpleGUI backend for matplotlib
sg.theme('LightGrey2')


def show_parameter_info(parameter_name):
    if parameter_name == "Population Size":
        info = "The number of individuals (solutions) in each generation of the Genetic Algorithm.\n" \
               "A larger population size can increase the exploration ability of the algorithm but also requires more computational resources.\n" \
               "A smaller population size may lead to faster convergence but risks getting stuck in non optimal solutions."
    elif parameter_name == "Number of Generations":
        info = "The total number of generations the Genetic Algorithm will run for.\n" \
               "Increasing the number of generations allows the algorithm more time to search for better solutions.\n" \
               "However, too many generations may lead to longer execution times without significant improvements."
    elif parameter_name == "Tournament Size":
        info = "The number of individuals randomly selected from the population for the tournament selection process.\n" \
               "A larger tournament size provides more diversity in parent selection and may lead to better solutions.\n" \
               "A smaller tournament size may cause the algorithm to converge quickly but risks premature convergence."
    elif parameter_name == "Mutation Rate":
        info = "The probability that an individual's genes will be mutated during the evolution process.\n" \
               "Mutation introduces random changes to the solution and helps explore new regions in the solution space.\n" \
               "Too high mutation rate may hinder convergence, while too low mutation rate may result in slow exploration."
    else:
        info = "No information available for this parameter."

    return info




# The section includes the ability to read the data file.
def read_data_file(filename):
    cities = []
    with open(filename, 'r') as file:
        found_section = False
        for line in file:
            line = line.strip()
            if line == "NODE_COORD_SECTION":
                found_section = True
                continue
            if found_section and line != "EOF":
                _, x, y = line.split()
                cities.append((float(x), float(y)))
    return cities

# Calculate the Euclidean distance between two cities
def distance(city1, city2):
    x1, y1 = city1
    x2, y2 = city2
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Generate an initial population of solutions
def generate_initial_population(cities, population_size):
    population = []
    for _ in range(population_size):
        solution = list(range(len(cities)))
        random.shuffle(solution)
        population.append(solution)
    return population

# Evaluate the fitness of a solution (total distance)
def evaluate_fitness(solution, cities):
    total_distance = 0
    for i in range(len(solution)):
        city1 = cities[solution[i]]
        city2 = cities[solution[(i + 1) % len(solution)]]
        total_distance += distance(city1, city2)
    return total_distance

# Tournament selection to choose parents
def tournament_selection(population, cities, tournament_size):
    tournament = random.sample(population, tournament_size)
    best_solution = min(tournament, key=lambda x: evaluate_fitness(x, cities))
    return best_solution

# Crossover to create offspring for the next generation
def crossover(parent1, parent2):
    child = [-1] * len(parent1)
    start, end = sorted(random.sample(range(len(parent1)), 2))
    child[start:end + 1] = parent1[start:end + 1]
    for i in range(len(parent2)):
        if parent2[i] not in child:
            j = (i + end + 1) % len(parent2)
            while child[j] != -1:
                j = (j + 1) % len(parent2)
            child[j] = parent2[i]
    return child

# Perform mutation to introduce genetic diversity
def mutate(solution):
    index1, index2 = random.sample(range(len(solution)), 2)
    solution[index1], solution[index2] = solution[index2], solution[index1]
    return solution

# Solve the TSP using a Genetic Algorithm
def solve_tsp(window, cities, population_size, generations, tournament_size, mutation_rate):
    population = generate_initial_population(cities, population_size)
    best_fitness = float('inf')
    best_solution = None
    fitness_history = []

    # Add a timer
    start_time = time.time()

    # Create the figure and subplots with one row and two columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Initialize the plot for best solution visualization
    ax1.set_title("Best Solution")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 100)

    # Initialize the plot for fitness history visualization
    ax2.set_title("Fitness Convergence Curve")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Total Distance")
    ax2.set_xlim(0, generations)
    ax2.set_ylim(0, evaluate_fitness(range(len(cities)), cities))
    ax2.grid(True)

    for generation in range(generations):
        new_population = []
        for _ in range(population_size // 2):
            parent1 = tournament_selection(population, cities, tournament_size)
            parent2 = tournament_selection(population, cities, tournament_size)
            offspring1 = crossover(parent1, parent2)
            offspring2 = crossover(parent2, parent1)
            if random.random() < mutation_rate:
                offspring1 = mutate(offspring1)
            if random.random() < mutation_rate:
                offspring2 = mutate(offspring2)
            new_population.extend([offspring1, offspring2])
        population = new_population

        best_current_fitness = min(evaluate_fitness(solution, cities) for solution in population)
        if best_current_fitness < best_fitness:
            best_fitness = best_current_fitness
            best_solution = population[np.argmin([evaluate_fitness(solution, cities) for solution in population])]

        fitness_history.append(best_fitness)

        # Update the best solution plot
        ax1.cla()
        visualize_solution(cities, best_solution, ax1)

        # Update the fitness history plot
        ax2.plot(range(generation + 1), fitness_history, 'b-')

        # Redraw the figure
        fig.canvas.draw()
        fig.canvas.flush_events()

        # Update the plot in the PySimpleGUI window
        plt_image = fig.canvas.tostring_rgb()
        img_width, img_height = fig.canvas.get_width_height()
        img = Image.frombytes("RGB", (img_width, img_height), plt_image)
        bio = io.BytesIO()
        img.save(bio, format="PNG")

    # Display the final best solution and its fitness in the interface
    window['-RESULTS-'].update(f"Best Solution: {best_solution}\nBest Fitness: {best_fitness}\n")

    return best_solution, best_fitness, fitness_history

# This section includes the simple visualization for the bonus using matplotlib
def visualize_solution(cities, solution, ax1):
    ax1.scatter([city[0] for city in cities], [city[1] for city in cities], color='blue')
    for i in range(len(solution)):
        city1 = cities[solution[i]]
        city2 = cities[solution[(i + 1) % len(solution)]]
        ax1.plot([city1[0], city2[0]], [city1[1], city2[1]], color='red')

# Main function
def main():
    layout = [
        [sg.Text("TSP Genetic Algorithm", font=("Helvetica", 18))],
        [sg.Text("Data File:", size=(10, 1)), sg.Input(default_text="Browse for Data File", key="-DATAFILE1-"), sg.FileBrowse()],
        [sg.Text("Population Size:", size=(15, 1)), sg.InputText("300", key="-POPULATIONSIZE-")],
        [sg.Text("Number of Generations:", size=(15, 1)), sg.InputText("400", key="-GENERATIONS-")],
        [sg.Text("Tournament Size:", size=(15, 1)), sg.InputText("13", key="-TOURNAMENTSIZE-")],
        [sg.Text("Mutation Rate:", size=(15, 1)), sg.InputText("0.09", key="-MUTATIONRATE-")],
        [sg.Checkbox("Show Plots", default=True, key="-SHOWPLOTS-")],
        [sg.Button("Run Optimization"), sg.Button("Exit"), sg.Button("Parameter Information")],
        [sg.Multiline(size=(60, 10), key='-RESULTS-', autoscroll=True)],
        [sg.Multiline(size=(60, 10), key='-PARAMETER_INFO-', disabled=True, autoscroll=True)]
    ]

    window = sg.Window('TSP Genetic Algorithm', layout)

    while True:
        event, values = window.read()

        if event == sg.WIN_CLOSED or event == 'Exit':
            break

        if event == "Run Optimization":
            data_file = values['-DATAFILE1-']
            population_size = int(values['-POPULATIONSIZE-'])
            generations = int(values['-GENERATIONS-'])
            tournament_size = int(values['-TOURNAMENTSIZE-'])
            mutation_rate = float(values['-MUTATIONRATE-'])

            cities = read_data_file(data_file)  # Read the cities from the data file
            window['-RESULTS-'].update("Running optimization...\n")
            plt.ion()
            optimization_thread = threading.Thread(target=solve_tsp,
                                                   args=(window, cities, population_size, generations, tournament_size, mutation_rate))
            optimization_thread.start()
      
        elif event == "Parameter Information":
            all_info = ""
            for parameter_name in ["Population Size", "Number of Generations", "Tournament Size", "Mutation Rate"]:
                info = show_parameter_info(parameter_name)
                all_info += f"{parameter_name}:\n{info}\n\n"
            window['-PARAMETER_INFO-'].update(all_info)

    window.close()


if __name__ == '__main__':
    main()
