import numpy as np
import pprint as pp
import random as rand
from operator import attrgetter
from main import rastrigin

from numpy import ones,vstack
from numpy.linalg import lstsq

class Abc:
    def __init__(self, func, population = 50, trials=10,
            employed_bees_percentage=0.5, epochs=50,
            borders_x=[-5, -5],
            borders_y=[5, 5]):

        self.population = population
        self.epochs = epochs
        self.func = func
        self._trials_ = 0
        self.trials = trials
        self.borders_x = np.array(borders_x)
        self.borders_y = np.array(borders_y)

        self.nmb_of_employed_bees = round(population * employed_bees_percentage)
        self.nmb_of_onlooker_bees = population - self.nmb_of_employed_bees

        self.food_sources = [self.create_random_food(borders_x, borders_y) for x in range(self.nmb_of_employed_bees)]
        self.bst_sol = [[0,0],0]

    def create_random_food(self, borders_x, borders_y):
        sol = [rand.uniform(borders_x[0],borders_y[0]), rand.uniform(borders_x[1],borders_y[1])]
        return sol, self.fitnesse(sol)

    def fitnesse(self,sol):
        res = self.func(sol)
        return res

    def find_optim(self):
        for epoch in range(self.epochs):
            self.employed_bees_stage()
            self.onlooker_bees_stage()
            self.scout_bees_stage()

    def employed_bees_stage(self):
        for i in range(self.nmb_of_employed_bees):
            food_source = self.food_sources[i]
            new_solution = self.generate_solution(i)
            if(self.fitnesse(new_solution)> food_source[1]):
                self.food_sources[i] = [new_solution, self.fitnesse(new_solution)]
            ft = food_source[1]
            if(ft>self.bst_sol[1]):
                self.bst_sol = food_source

    def onlooker_bees_stage(self):
        for i in range(self.nmb_of_onlooker_bees):

            sm = sum(x[1] for x in self.food_sources)
            probabilities = [x[1] / sm for x in self.food_sources]

            selected_index = rand.choices(np.arange(0, len(probabilities)), probabilities)[0]
            selected_source = self.food_sources[selected_index]
            new_solution = self.generate_solution(selected_index)
            if (self.fitnesse(new_solution) > self.food_sources[i][1]):
                self.food_sources[i] = [new_solution, self.fitnesse(new_solution)]

    def scout_bees_stage(self):
        for i in range(self.nmb_of_employed_bees):
            food_source = self.food_sources[i]
            self._trials_+=1
            #if food_source.trials > self.trials_limit:
            if(self._trials_ == self.trials):
                self.food_sources[i] = self.create_random_food(self.borders_x, self.borders_y)

    def generate_solution(self, working_bee):
        solution = self.food_sources[working_bee][0]
        another_source_index = self.another_solution([working_bee])
        another_solution = self.food_sources[another_source_index][0]

        if (all(solution) == all(another_solution)):
            points = [solution, self.create_random_food(self.borders_x,self.borders_y)[0]]
        else:
            points = [solution,another_solution]
        x_coords, y_coords = zip(*points)
        A = vstack([x_coords, ones(len(x_coords))]).T
        m, c = lstsq(A, y_coords, rcond=None)[0]
        val = rand.uniform(0, 1 / 2)
        new_solution = np.copy(solution)
        if(x_coords[0]<x_coords[1]):
            new_solution[0] += val
        else:
            new_solution[0] -= val
        new_solution[1] = m*new_solution[0] + c
        return new_solution

    def another_solution(self,working_bee):
        srted = sorted(self.food_sources, key = lambda x: -x[1])
        sm = sum(x[1] for x in srted)
        prob = [x[1] / sm for x in srted]
        ex = rand.choices(np.arange(0, len(srted)), prob)[0]
        return ex


k = Abc(rastrigin)
k.find_optim()
print(k.bst_sol)