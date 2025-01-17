import numpy as np
import random

class cheetah_optimizer():
    def __init__(self, n_cheetahs, n_iterations = None) -> None:
        self.n_cheetahs = n_cheetahs
        self.n_iterations = n_iterations

    def set_obj_f(self, obj_f) -> None:
        self.obj_f = obj_f
    
    def set_bounds(self, bounds) -> None:
        self.bounds = bounds
    
    def gen_cheetah(self):
        cheetahs = []
        for lower, upper in self.bounds:
            cheetahs.append(np.random.randint(lower * 100, upper * 100, self.n_cheetahs)/100)
        return np.array(cheetahs).T
    
    def clip(self, arr):
        clipped = np.array(arr)
        for i, (lower, upper) in enumerate(self.bounds):
            if clipped[i] < lower:
                clipped[i] = lower + np.random.rand()*(upper - lower)
            elif clipped[i] > upper:
                clipped[i] = upper - np.random.rand()*(upper - lower)
            else:
                clipped[i] = clipped[i]
        return np.round(clipped, 2)
    
    def random_select(self, low, high, size):
        arr = list(range(low, high, 1))
        selected = []
        for _ in range(size):
            gen = arr[np.random.randint(0, len(arr))]
            selected.append(gen)
            q = arr.index(gen)
            arr.pop(q)
        return np.array(selected)

    def optimize(self):
        history = {"iterations": [], "Score": [], "Cheetah" : []}
    
        # Define the initial population size and dimensions
        n = self.n_cheetahs
        D = len(self.bounds)

        # Generate the initial population of the cheetahs and evaluate the fitness of each cheetah
        cheetahs = self.gen_cheetah()
        fitness = []
        for cheetah in cheetahs:
            fitness.append(self.obj_f(cheetah))
        fitness = np.array(fitness)
        
        #Initialize home, leader and prey
        home = cheetahs.copy()
        leader_index =  np.argmax(fitness)
        leader = cheetahs[leader_index].copy()
        prey_index = np.argmax(fitness)
        prey = cheetahs[prey_index].copy()

        # Initialize the hunting time and iteration counters, and determine max hunting time and iteration number
        t = 0
        leaders = [leader.copy()]

        it = 0

        if self.n_iterations == None:
            maxit = D*2000
        else:
            maxit = self.n_iterations

        T = 60 * int(np.ceil(D/10))
        
        
        # Lopp while max iteration number is not reached
        while it <= maxit:

            # Select m (1 <= m <= n) members of cheetahs randomly
            m = np.random.randint(2, n)
            members = self.random_select(0, n, m)
            

            # loop over each member
            for k, member in enumerate(members):

                # define the neighbour member
                if k != len(members) - 1:
                    neighbour = k + 1
                else:
                    neighbour = k - 1

                X = cheetahs[members[k]]
                X1 = cheetahs[members[neighbour]]
                Xbest = prey.copy()
                Z = X.copy()

                # loop ever each dimension in a random order
                DD = self.random_select(0, D, D)
                for j in DD:

                    # Calculate r_hat, r_check, alpha, beta and H
                    r_hat = np.random.rand()
                    
                    r = np.random.randn()
                    r_check = abs(r) ** np.exp(r / 2) * np.sin(2 * np.pi * r)

                    if member == leader_index:
                        alpha = 0.0001 * (t/T) * (self.bounds[j][1] - self.bounds[j][0])
                    else:
                        alpha = 0.0001 * (t/T) * (self.bounds[j][1] - self.bounds[j][0]) + (0.0001 * (np.random.rand() > 0.9))

                    beta = X1[j] - X[j]

                    h0 = np.exp(2 * (1 - t / T))
                    r1 = np.random.rand()
                    H = h0 * (2 * r1 - 1)

                    # Calculate r2 and r3
                    r2 = np.random.rand()
                    r3 = np.random.rand()

                    if r2 <= r3:
                        # Calculate r4
                        r4 = 3 * np.random.rand()
                        if H > r4: #Attack
                            Z[j] = Xbest[j] + r_check * beta
                            #print("Attack")

                        else: #Search
                            Z[j] = X[j] + (1 / r_hat) * alpha
                            #print("Search")

                    else: # Sit and Wait
                        Z[j] = X[j]
                        #print("Sit and Wait")

                # Update cheetahs[member] and leader and prey
                Z = self.clip(Z)
                cheetahs[member] = Z.copy()
                fitness[member] = self.obj_f(cheetahs[member])

                if member != leader_index:
                    if fitness[member] > fitness[leader_index]:
                        leader_index = member
                        leader = cheetahs[leader_index].copy()
                else:
                    leader_index = np.argmax(fitness)
                    leader = cheetahs[leader_index].copy()
                #print("member updated")

                if self.obj_f(cheetahs[member].copy()) > self.obj_f(prey):
                    prey = cheetahs[member].copy()
                    #print("prey updated from member assignment")
                            
            # update hunting time
            t += 1
            leaders.append(leader.copy())

            # Implement when to leave prey and go back home
            r = np.random.rand()
            if t > r * T and round(t - r * T) > 1 and t > 2:
                if (abs(leaders[t] - leaders[round(t - r * T)]) <= abs(0.01 * leaders[t])).all():
                    #print("left prey and went back home")
                    cheetahs = home.copy()
                    cheetahs[member] = prey.copy()
                    t = 0

                    fitness = []
                    for cheetah in cheetahs:
                        fitness.append(self.obj_f(cheetah))
                    leader_index = np.argmax(fitness)
                    leader = cheetahs[leader_index].copy()
                    leaders = [leader.copy()]

            # update iteration number
            a = int(it / maxit * 40)
            b = 40 - a
            print(f"\r{ a * '='}{b * '-'} {round(it / maxit * 100)}%", end = "", flush = True)
            it += 1

            # Update prey
            if self.obj_f(leader) > self.obj_f(prey):
                prey = leader.copy()
                #print("prey updated from leader assignmet")

            score = self.obj_f(prey)
            history["iterations"].append(it - 1)
            history["Score"].append(score)
            history["Cheetah"].append(prey)
        
        return history

class swarm_optimizer():
    def __init__(self, population_size, max_iter) -> None:
        """
        This function initializes a swarm optimizer using its
        population size and maximum generations for evolution
        """
        self.pop_size = population_size
        self.max_iter = max_iter

    def set_obj_f(self, obj_f) -> None:
        """
        This function sets the objective function of the swarm optimizer
        """
        self.obj_f = obj_f
    
    def set_bounds(self, bounds) -> None:
        """
        This function sets the bounds of the swarm optimizer for each
        dimension in the search space. 

        bounds is a list that is of the same length as the number of
        dimensions in the search space. Each element of bounds is a tuple 
        of two elements containing the values for the lower and upper 
        limits as the first and second elements.
        """
        self.bounds = bounds
    
    def evaluate(self, members):
        """
        This function returns the fitness of all members specified in an itertive
        """
        return np.apply_along_axis(self.obj, 1, members)

    def gen_population(self):
        """
        This function generates a random population of members 
        given the bounds specifiied for each dimension
        """
        population = []
        for lower, upper in self.bounds:
            population.append(np.random.randint(lower * 100, upper * 100, self.pop_size) / 100)
        return np.array(population).T
    
class elephant_herding_optimizer(swarm_optimizer):
    def __init__(self, population_size, max_gen, n_clans, output = True) -> None:
        # overide initiation from swarm_optimizer class to include maax_clan size
        if population_size >= 3: 
            self.pop_size = population_size 
        else:
            self.pop_size = 3
            print("Population size is too small, population size has been set to 3")

        self.max_gen = max_gen
        
        # ensure the maax clan size is valid and reassign it if it not valid
        max_clan_size = int(population_size / 3)
        if n_clans <= max_clan_size:
            self.n_clans = n_clans
        else:
            self.n_clans = max_clan_size
            print(f"Clan size is too large, clan size was reduced to {max_clan_size}")
        
        self.output = output

    def optimize(self):
        # initialize generation counter and output storage
        gen_it = 0
        if self.output == True:
            output = []
        
        # initialize population
        population = self.gen_population()

        # seperate population into clans
        random.shuffle(population)
        clan_members_map = {}
        clan_size = int(self.pop_size / self.n_clans)
        member_index = 0
        
        for clan in range(self.n_clans):
            clan_members_map[clan] = list(range(member_index, member_index + clan_size))
            member_index = member_index + clan_size

        n_unassigned_members = self.pop_size % self.n_clans

        current_clan = 0
        for _ in range(n_unassigned_members):
            clan_members_map[current_clan].append(member_index)
            member_index += 1
            current_clan += 1
        
        best_member, best_fitness = [None, np.inf]

        # peroform optimization steps
        while gen_it < self.max_gen:

            # for each clan
            for clan in clan_members_map.keys():
                # find best member
                clan_fitness = self.evaluate([population[member] for member in clan_members_map[clan]])
                best_clan_member_index = clan_members_map[clan][np.argmin(clan_fitness)]

                # update members in clan using best member
                alpha = 0.5 
                for member_index in clan_members_map[clan]:
                    updated_member = []
                    for dimension in range(len(population[member_index])):
                        updated_member.append(population[member_index][dimension] + alpha * (population[best_clan_member_index][dimension] - population[member_index][dimension]) * random.random())
                    population[member_index] = updated_member

                # find the center
                center = np.mean(np.array([population[center_member_index] for center_member_index in clan_members_map[clan]]), axis = 0)

                # update the best using the center
                beta = 0.1
                population[best_clan_member_index] = beta * center

                # find the worst
                clan_fitness = self.evaluate([population[member] for member in clan_members_map[clan]])
                worst_member_index = clan_members_map[clan][np.argmax(clan_fitness)]

                # update worst member using the bounds
                updated_member = []
                for lower, upper in self.bounds:
                    updated_member.append(lower + (upper - lower) * random.random())
                population[worst_member_index] = updated_member

            # find the best member in population and history and store it
            overall_fitness = self.evaluate(population)
            best_member_index = np.argmin(overall_fitness)

            if overall_fitness[best_member_index] < best_fitness:
                best_member = population[best_member_index]
                best_fitness = overall_fitness[best_member_index]

            if self.output == True:
                output.append([best_member, best_fitness])

            gen_it += 1
        
        # return best member and fitness
        if self.output == True: 
            return output
        else: 
            return (best_member, best_fitness)

class dwarf_mongoose_optimizer(swarm_optimizer):
    def __init__(self, population_size, max_iter, n_babysitters, history = True):
        super().__init__(population_size, max_iter)
        self.n_babysitters = n_babysitters
        self.history = history
    
    def random_select(self, population_ids, remove = False):
        """
        This function selects a random id from a population
        """
        id = random.choice(population_ids)

        if remove == True:
            population_ids.remove(id)
        
        return id

    def optimize(self):
        """
        This function returns an optimal position and fitness 
        given an objective function and a search space using 
        the dwarf mongoose optimization algorithm

        https://doi.org/10.1016/j.cma.2022.114570
        """

        # Initialize the algorithm parameters and mongoose population
        population = self.gen_population()

        # evaluate the fitness of each member and set the alpha as the best
        fitness = self.evaluate(population)
        alpha_id = np.argmin(fitness)
        best = [population[alpha_id], fitness[alpha_id]]
        history = [tuple(best)]

        # divide the population into babysitters and search agents and 
        # add the alpha as a search agent
        search_agents = []
        babysitters = []
        search_agents.append(alpha_id)

        mongeese = list(range(self.pop_size))
        mongeese.remove(alpha_id)

        babysitters_added = 0
        for _ in range(len(mongeese)):
            id = self.random_select(mongeese, remove = True)
            babysitters.append(id)
            babysitters_added += 1
            if babysitters_added == self.n_babysitters:
                search_agents = mongeese.copy()
                break

        # set the babysitter exchange parameter L and time counter
        L = np.round(0.6 * self.n_babysitters * len(population[0]), 0)
        C = 0

        # for each iteration perform the following:
        for i in range(self.max_iter):

            # Calculate the fitness of the population and find the alpha female
            fitness = self.evaluate(population)
            alpha_id = np.argmin(fitness)
            if fitness[alpha_id] <= best[1]:
                best = [population[alpha_id], fitness[alpha_id]]

            # update the time counter C
            C += 1

            # update food position
            phi = np.random.uniform(-1, 1)

            # evaluate new fitness of population

            # determine the movement vector

            # exchange the baby sitters

            # compute the scout group

            # update the history
            if history:
                history.append(tuple(best))

        # return the best solution
        if self.history:
            return history
        else:
            return best

