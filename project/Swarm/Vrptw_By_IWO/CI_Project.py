import math
import random
import functools
from matplotlib import pyplot as plt
import csv
import copy
import numpy as np

# Invasive Weed Optimization
# Vehicle Routing Problem with Time Windows

class IWO_VRPTW:
    # Number of trucks
    trunk_amount = None 
    # Truck capacities 
    trunk_volumes = None  
    # Number of destinations (customers)
    target_amount = None 
    # Coordinates of destinations 
    target_sites = None 
    # Time windows for each destination [begin, end]
    target_time_limits = None  
    # Volume of goods for each destination
    target_volumes = None  
    # Service times at each destination
    target_service_times = None  

    # Distance matrix
    dist = None  

    losses = []

    # IWO parameters

    # Initial population size
    ipop = 20  
    # Maximum population size
    mpop = 100 
    # Number of iterations 
    iter = 5000
    # Minimum number of seeds  
    smin = 0 
    # Maximum number of seeds 
    smax = 5 
    # Initial standard deviation 
    isigma = 1  
    # Final standard deviation
    fsigma = 1e-6  
    

    def __init__(self, trunk_volumes: list, target_sites: list, target_time_limits: list, target_volumes: list, target_service_times):
        self.trunk_volumes = trunk_volumes
        self.trunk_amount = len(trunk_volumes)

        self.target_sites = target_sites
        self.target_time_limits = target_time_limits
        self.target_volumes = target_volumes
        self.target_service_times = target_service_times
        
        self.target_amount = len(target_volumes)

        self.dist = np.zeros((self.target_amount, self.target_amount))

        for i in range(self.target_amount):
            for j in range(self.target_amount):
                self.dist[i][j] = math.sqrt((target_sites[0][i] - target_sites[0][j])**2 + (target_sites[1][i] - target_sites[1][j])**2)

    def run(self):
        pos = self.init_solution()

        score = np.array([self.cost(p) for p in pos])

        # Global best score
        gbest = np.min(score)
        # Global best solution  
        gbest_pos = pos[np.argmin(score)]  

        for t in range(self.iter):
            if t % 10 == 0:
                print(t, 'Best cost:', gbest)

            # Update standard deviation  

            sigma = ((self.iter - t - 1) / (self.iter - 1)) ** 2 * (self.isigma - self.fsigma) + self.fsigma
            new_pos, new_score = self.reproduce(pos, score, sigma)
            pos, score = self.exclusion(new_pos, new_score)

            if np.min(score) < gbest:
                gbest = np.min(score)
                gbest_pos = pos[np.argmin(score)]
            self.losses.append(gbest)

        self.best_solution = gbest_pos
        print("Best solution:\n", self.best_solution[0], '\n', self.best_solution[1])

        self.draw_pic()
        self.plot_loss()

    def plot_loss(self):

        plt.plot(range(len(self.losses)), self.losses)
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.title('Loss Over Time')
        plt.show()

    def init_solution(self):
        # Initialize the weed population

        pos = []
        for _ in range(self.ipop):
            # Initialize with the depot
            solution = [[0], [0]]  
            volumes = np.zeros(self.trunk_amount)

            for j in range(1, self.target_amount):
                No_ = random.randint(0, self.trunk_amount - 1)
                counter = 0

                while volumes[No_] + self.target_volumes[j] > self.trunk_volumes[No_]:
                    No_ = random.randint(0, self.trunk_amount - 1)

                    counter += 1
                    if counter > 0xffff:
                        raise ValueError('Vehicle capacity is insufficient to load all goods. No solution possible.')
                    
                solution[0].append(No_)
                solution[1].append(len([i for i in solution[0] if i == No_]))
                volumes[No_] += self.target_volumes[j]

            pos.append(solution)
        return pos

    def cost(self, solution):

        # Calculate the cost of a solution 
        
        # early and late penalty
        PE, PL = 10, 100 
        max_time = 0
        sum_dist = 0
        penalty = 0

        for k in range(self.trunk_amount):
            pre_pos, _ = 0, 0
            k_time = 0
            counter = 0

            for i in range(1, len(solution[0])):
                if solution[0][i] == k:
                    k_time += self.dist[pre_pos][i]
                    sum_dist += self.dist[pre_pos][i]
                    counter += 1

                    if self.target_time_limits[0][i] > k_time and counter > 1:
                        # early penalty
                        penalty += PE * (self.target_time_limits[0][i] - k_time)
                        k_time = self.target_time_limits[0][i]

                    elif self.target_time_limits[1][i] < k_time:
                        # late penalty
                        penalty += PL * (k_time - self.target_time_limits[1][i])

                    k_time += self.target_service_times[i]
                    pre_pos = i

            k_time += self.dist[pre_pos][0]
            sum_dist += self.dist[pre_pos][0]
            max_time = max(max_time, k_time)

        return sum_dist + penalty

    def reproduce(self, pos, score, sigma):
        new_pos = []
        new_score = []
        min_score = np.min(score)
        max_score = np.max(score)

        for i in range(len(pos)):
            ratio = 0.5 if min_score == max_score else (score[i] - max_score) / (min_score - max_score)
            # the number of seeds
            snum = math.floor(self.smin + (self.smax - self.smin) * ratio) 

            for _ in range(snum):
                temp_pos = copy.deepcopy(pos[i])
                for j in range(1, len(temp_pos[0])):
                    if random.random() < sigma:
                        temp_pos[0][j] = random.randint(0, self.trunk_amount - 1)
                    temp_pos[1][j] = len([idx for idx in temp_pos[0] if idx == temp_pos[0][j]])

                new_pos.append(temp_pos)
                new_score.append(self.cost(temp_pos))
        return new_pos, new_score

    def exclusion(self, new_pos, new_score):
        # Competitive exclusion

        if len(new_pos) > self.mpop:
            sorted_idx = np.argsort(new_score)
            return [new_pos[i] for i in sorted_idx[:self.mpop]], [new_score[i] for i in sorted_idx[:self.mpop]]
        return new_pos, new_score

    def draw_pic(self):
        color = ['#00FFFF', '#7FFFD4', '#0000FF', '#8A2BE2', '#A52A2A', '#DEB887',
                 '#FFFF00', '#9ACD32', '#008000', '#5F9EA0', '#7FFF00', '#D2691E',
                 '#FF7F50', '#6495ED', '#DC143C', '#00FFFF', '#008B8B', '#B8860B',
                 '#A9A9A9', '#006400', '#BDB76B', '#8B008B', '#556B2F', '#FF8C00',
                 '#9932CC', '#8B0000', '#E9967A', '#8FBC8F', '#483D8B', '#2F4F4F',
                 '#00CED1', '#9400D3', '#FF1493', '#00BFFF', '#696969', '#1E90FF',
                 '#B22222', '#FFFAF0', '#228B22', '#FF00FF', '#DCDCDC', '#F8F8FF',
                 '#FFD700', '#DAA520', '#808080', '#FF69B4', '#CD5C5C', '#4B0082',
                 '#FFFFF0', '#F0E68C', '#E6E6FA', '#FFF0F5', '#7CFC00', '#FFFACD',
                 '#ADD8E6', '#F08080', '#E0FFFF', '#FAFAD2', '#90EE90', '#D3D3D3',
                 '#FFB6C1', '#FFA07A', '#20B2AA', '#87CEFA', '#778899', '#B0C4DE',
                 '#FFFFE0', '#00FF00', '#32CD32', '#FAF0E6', '#FF00FF', '#800000',
                 '#66CDAA', '#0000CD', '#BA55D3', '#9370DB', '#3CB371', '#7B68EE',
                 '#00FA9A', '#48D1CC', '#C71585', '#191970', '#F5FFFA', '#FFE4E1',
                 '#FFE4B5', '#FFDEAD', '#000080', '#FDF5E6', '#808000', '#6B8E23',
                 '#FFA500', '#FF4500', '#DA70D6', '#EEE8AA', '#98FB98', '#AFEEEE',
                 '#DB7093', '#FFEFD5', '#FFDAB9', '#CD853F', '#FFC0CB', '#DDA0DD',
                 '#B0E0E6', '#800080', '#FF0000', '#BC8F8F', '#4169E1', '#8B4513',
                 '#FA8072', '#FAA460', '#2E8B57', '#FFF5EE', '#A0522D', '#C0C0C0',
                 '#87CEEB', '#6A5ACD', '#708090', '#FFFAFA', '#00FF7F', '#4682B4',
                 '#D2B48C', '#008080', '#D8BFD8', '#FF6347', '#40E0D0', '#EE82EE',
                 '#F5DEB3', '#FFFFFF', '#F5F5F5']

        for i in range(self.trunk_amount):
            x, y = [self.target_sites[0][0]], [self.target_sites[1][0]]

            for j in range(1, len(self.best_solution[0])):
                if self.best_solution[0][j] == i:
                    x.append(self.target_sites[0][j])
                    y.append(self.target_sites[1][j])

            x.append(self.target_sites[0][0])
            y.append(self.target_sites[1][0])
            plt.plot(x, y, color=color[i % len(color)], marker='o', label=f'Truck {i + 1}')

        plt.scatter(self.target_sites[0], self.target_sites[1], color='red', zorder=5)
        plt.legend()
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('VRPTW Solution')
        plt.show()


def read_in_data(path: str, row_num):
    def func_cmp(a, b):
        if a[3] == b[3]:
            return a[4] - b[4]
        return a[3] - b[3]
    
    # Coordinates, volumes, time windows, service times

    data = [[[], []], [], [[], []], []]  
    data_in = []

    with open(path) as file:
        csvf = csv.reader(file)
        next(csvf)
        for _ in range(row_num):
            line = next(csvf)
            ele = [int(line[1]), int(line[2]), int(line[3]), int(line[4]), int(line[5]), int(line[6])]
            data_in.append(ele)

    data_in = sorted(data_in, key=functools.cmp_to_key(func_cmp))

    for i in data_in:
        data[0][0].append(i[0])
        data[0][1].append(i[1])
        data[1].append(i[2])
        data[2][0].append(i[3])
        data[2][1].append(i[4])
        data[3].append(i[5])
    return data

if __name__ == "__main__":
    data = read_in_data('./input/rc101.csv', 51)

    trunk_volume = [200 for _ in range(13)]
    target_site = data[0]
    target_volume = data[1]
    target_time_limit = data[2]
    target_service_time = data[3]

    iwo = IWO_VRPTW(trunk_volume,target_site,target_time_limit,target_volume,target_service_time)
    iwo.run()