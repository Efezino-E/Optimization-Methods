import numpy as np
import random

# TODO implement an error function that checks for needed values before optimizing and call them out if mising
# TODO implete prgress bar as a genral function and let it print a done line after

class optimizer():
    def __init__(self, bounds, obj_f):
        """
        This function instantiates an optimizer class with the search space (bounds)
        and the objective function to maximize within that search space.

        *   bounds defines a search space. Each dimension in the search space, is specified by a tuple 
        (lower bound for dimesnion, upper bound for dimension) in bounds
        *   obj_f is the objective function specified is to be maximized. 
        If instead you have a cost function, multiply it by -1 or raise it to the power of -1
        """
        self.bounds = bounds
        self.lb, self.ub = zip(*bounds)
        self.obj_f = obj_f
        self.history = True
    
    def __name__():
        return "Optimizer"

    def set_history(self, value):
        """
        This function sets whether best values and parameters found for each iteration
        should be stored or not for the optimization process. Default value is True
        """
        self.history = value

class swarm_optimizer(optimizer):
    def __init__(self, bounds, obj_f, population_size = None, max_iter = None):
        """
        This functions instantiates a swarm optimizer class using its
        population size and maximum generations for evolution
        """
        super().__init__(bounds, obj_f)
        self.pop_size = population_size
        self.max_iter = max_iter

        # ensure population is valid
        if self.pop_size != None and self.pop_size < 3: 
            raise ValueError ("Population size is too small")
    
    def __name__():
        return "SWarm Optimizer"
    
    def set_population_size(self, pop_size):
        self.pop_size = pop_size
        if self.pop_size < 3: 
            raise ValueError ("Population size is too small")
    
    def set_max_iter(self, max_iter):
        self.max_iter = max_iter

    def evaluate(self, members):
        """
        This function returns the fitness of all members specified in an itertive
        """
        return np.apply_along_axis(self.obj_f, 1, members)

    def gen_population(self):
        """
        This function generates a random population of members 
        given the bounds specifiied for each dimension
        """
        population = []
        for lower, upper in self.bounds:
            population.append(np.random.randint(lower * 100, upper * 100, self.pop_size) / 100)
        return np.array(population).T

    def clip(self, arr):
        """
        This function clips an entire array within the bounds of the problem
        """
        return np.clip(arr, self.lb, self.ub)

class cheetah_optimizer(swarm_optimizer):
    def __init__(self, bounds, obj_f, population_size = None, max_iter = None):
        super().__init__(bounds, obj_f, population_size, max_iter)
    
    def __name__():
        return "Cheetah Optimizer"

    def random_select(self, low, high, size):
        """
        Returns N unique elements within the limits low nd high provided
        """
        if high - low < size:
            raise ValueError("Range is too small to generate required numbers")
        
        return np.random.choice(range(low, high), size = size, replace = False)

    def optimize(self):
        """
        This function implements the Cheetah Optimizer as defined by Mohammed et al (2022)
        https://doi.org/10.1038/s41598-022-14338-z
        """
        if self.history:
            history = {"iteration_no": [], "score": [], "parameters" : []}
    
        # Define the initial population size and dimensions
        n = self.pop_size
        D = self.dim

        # Generate the initial population of the cheetahs and evaluate the fitness of each cheetah
        cheetahs = self.gen_population()
        fitness = self.evaluate(cheetahs)
        
        #Initialize home, leaders and prey
        home = cheetahs.copy()
        prey_index = np.argmax(fitness)
        prey = cheetahs[prey_index].copy()
        score = fitness[prey_index]
        leaders = []

        # Initialize the hunting time (t) and iteration counter (it)
        t = 0
        it = 0

        # determine max hunting time (T) and iteration number (maxit)
        T = 60 * int(np.ceil(D/10))
        if self.max_iter == None:
            maxit = D*2000
        else:
            maxit = self.max_iter
        
        # Loop while max iteration number is not reached
        while it <= maxit:
            # Select m (1 <= m <= n) members of cheetahs randomly
            m = np.random.randint(2, n)
            members = self.random_select(0, n, m)
            iteration_fitness = [fitness[member] for member in members]
            leader_index = members[np.argmax(iteration_fitness)]
            leader = cheetahs[leader_index].copy()

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
                    # Calculate r_hat and r_check
                    r_hat = np.random.rand()
                    r = np.random.randn()
                    r_check = abs(r) ** np.exp(r / 2) * np.sin(2 * np.pi * r)

                    # calculate alpha
                    if member == leader_index:
                        alpha = 0.0001 * (t/T) * (self.bounds[j][1] - self.bounds[j][0])
                    else:
                        alpha = 0.0001 * (t/T) * (self.bounds[j][1] - self.bounds[j][0]) + (0.0001 * (np.random.rand() > 0.9))

                    # calculate beta
                    beta = X1[j] - X[j]

                    # calculate H
                    h0 = np.exp(2 * (1 - t / T))
                    r1 = np.random.rand()
                    H = h0 * (2 * r1 - 1)

                    # Calculate r2 and r3
                    r2 = np.random.rand()
                    r3 = np.random.rand()

                    if r2 <= r3:
                        # Calculate r4
                        r4 = 3 * np.random.rand()
                        if H > r4: # Attack
                            Z[j] = Xbest[j] + r_check * beta

                        else: # Search
                            Z[j] = X[j] + (1 / r_hat) * alpha

                    else: # Sit and Wait
                        Z[j] = X[j]

                # Update cheetahs[member] and leader and prey
                Z = self.clip(Z)
                cheetahs[member] = Z.copy()
                fitness[member] = self.obj_f(cheetahs[member])
                iteration_fitness[np.argmax(members == member)] = fitness[member]

                if member != leader_index:
                    if fitness[member] > fitness[leader_index]:
                        leader_index = member
                        leader = cheetahs[leader_index].copy()
                else:
                    leader_index = members[np.argmax(iteration_fitness)]
                    leader = cheetahs[leader_index].copy()

                if fitness[leader_index] > score:
                    score = fitness[leader_index]
                    prey_index = leader_index
                    prey = cheetahs[leader_index].copy()
                            
            # update hunting time
            t += 1
            leaders.append(leader.copy())

            # Implement when to leave prey and go back home
            r = np.random.rand()
            if t > r * T and round(t - r * T) > 1 and t > 2:
                if (abs(leaders[t - 1] - leaders[round(t - r * T - 1)]) <= abs(0.01 * leaders[t - 1])).all():
                    t = 0
                    cheetahs = home.copy()
                    cheetahs[member] = prey.copy()
                    fitness = self.evaluate(cheetahs)
                    leaders = []

            # update iteration number and progress bar
            a = int(it / maxit * 40)
            b = 40 - a
            print(f"\r{ a * '='}{b * '-'}{round(it / maxit * 100)}%{'' * 5}", end = "", flush = True)
            it += 1

            # Update prey 
            max_index = np.argmax(fitness)
            if fitness[max_index] > score:
                score = fitness[max_index]
                prey_index = max_index
                prey = cheetahs[prey_index]

            if self.history:
                history["iteration_no"].append(it - 1)
                history["score"].append(score)
                history["parameters"].append(prey)
        
        if self.history:
            return history
        else:
            return score , prey
  
class elephant_herding_optimizer(swarm_optimizer):
    def __init__(self, bounds, obj_f, population_size = None, max_iter = None, n_clans = None) -> None:
        super().__init__(bounds, obj_f, population_size, max_iter)
        self.n_clans = n_clans
    
    def __name__():
        return "Elephant Herding Optimizer"
    
    def set_population_size(self, pop_size):
        self.pop_size = pop_size
        # ensure clan size is valid
        if self.n_clans == None:
            self.n_clans = int(pop_size / 3)
        elif self.n_clans > int(pop_size / 3):
            raise ValueError (f"Clan size is too large")
    
    def set_n_clans(self, n_clan):
        self.n_clans = n_clan
        if self.pop_size == None:
            raise ValueError ("n_clans specified before pop size")
        if self.n_clans > int(self.pop_size / 3):
            raise ValueError ("Clan size is too large")

    def optimize(self):
        """
        This function implements the Elephant Herding Optimizer as define by Gai-Ge et al (2015)
        https://doi.org/10.1109/ISCBI.2015.8
        """
        # initialize generation counter and output storage
        gen_it = 0
        if self.history == True:
            history = {"iteration_no": [], "score": [], "parameters" : []}
        
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
        
        best_member, best_fitness = [None, -np.inf]

        # perform optimization steps
        while gen_it < self.max_iter:
            # for each clan
            for clan in clan_members_map.keys():
                # find best member
                clan_fitness = self.evaluate([population[member] for member in clan_members_map[clan]])
                best_clan_member_index = clan_members_map[clan][np.argmax(clan_fitness)]

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
            best_member_index = np.argmax(overall_fitness)

            if overall_fitness[best_member_index] > best_fitness:
                best_member = population[best_member_index]
                best_fitness = overall_fitness[best_member_index]

            # Update iteration number and progress bar
            gen_it += 1
            a = int(gen_it / self.max_iter * 40)
            b = 40 - a
            print(f"\r{ a * '='}{b * '-'}{round(gen_it / self.max_iter * 100)}%{'' * 5}", end = "", flush = True)

            if self.history == True:
                history["iteration_no"].append(gen_it)
                history["score"].append(best_fitness)
                history["parameters"].append(best_member)
        
        # return best member and fitness
        if self.history == True: 
            return history
        else: 
            return best_fitness, best_member

class dwarf_mongoose_optimizer(swarm_optimizer):
    def __init__(self, population_size, max_iter, n_babysitters, history = True):
        super().__init__(population_size, max_iter)
        self.n_babysitters = n_babysitters
        self.history = history
    
    def __name__():
        return "Dwarf Mongoose Optimizer"
    
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
            raise NotImplementedError

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

class chameleon_swarm_algorithm(swarm_optimizer):
    def __init__(self, population_size, max_iter):
        super().__init__(population_size, max_iter)
    
    def __name__():
        return "Chameleon Swarm Optimizer"

    def optimize(self):
        chameleon_positions = self.gen_population() # 7. Randomize the position of the chameleons

        fitness = self.evaluate(chameleon_positions) # 9. evaluate the position of the chameleons
        fmin0 = np.min(fitness) # current best fitness

        chameleon_best_positions = chameleon_positions.copy()
        g_position = chameleon_positions[np.argmin(fitness)] #  best position
        v = 0.1 * chameleon_best_positions # 8a initializing velocities 
        v0 = np.zeros_like(v) # 8b initializing velocities 

        cg_curve = np.zeros(self.max_iter)

        for t in range(1, self.max_iter + 1):
            # 11 - 13
            mu = 1.0 * np.exp(-(3.5 * t / self.max_iter) ** 3.0)
            omega = (1 - (t / self.max_iter)) ** (1.0 * np.sqrt(t / self.max_iter))
            a = 2590 * (1 - np.exp(-np.log(t)))

            p1 = 0.25
            p2 = 1.5

            for i in range(self.pop_size):
                if np.random.rand() >= 0.1:
                    chameleon_positions[i, :] += (
                        p1 * (chameleon_best_positions[i, :] - g_position) * np.random.rand()
                        + p2 * (g_position - chameleon_positions[i, :]) * np.random.rand())
                else:
                    chameleon_positions[i, :] += mu * (
                        (self.ub - self.lb) * np.random.rand(self.dim) + self.lb
                        ) * np.sign(np.random.rand(self.dim) - 0.5)

                v[i, :] = omega * v[i, :] + p1 * (chameleon_best_positions[i, :] - chameleon_positions[i, :]) * np.random.rand() + p2 * (g_position - chameleon_positions[i, :]) * np.random.rand()
                chameleon_positions[i, :] += (v[i, :] ** 2 - v0[i, :] ** 2) / (2 * a)

            v0 = v.copy()

            chameleon_positions = np.clip(chameleon_positions, self.lb, self.ub)
            fitness = np.apply_along_axis(self.fobj, 1, chameleon_positions)

            for i in range(self.pop_size):
                if fitness[i] < np.min(fitness):
                    chameleon_best_positions[i, :] = chameleon_positions[i, :]

            current_fmin = np.min(fitness)
            current_best_position = chameleon_positions[np.argmin(fitness)]

            if current_fmin < fmin0:
                g_position = current_best_position.copy()
                fmin0 = current_fmin

            cg_curve[t - 1] = fmin0

        return fmin0, g_position, cg_curve
