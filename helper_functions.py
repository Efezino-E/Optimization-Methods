# function that takes an swarm optimizer, and it's requirements, 
# and plots how the average and variannce of the best score 
# varies with the number of agents in the swarm

# function that takes a swarm optimizer and shows how quickly it converges 
# as expressed by a percentage of the maximum value attained vs the number of iterations. 
# Find a way to average this over multiple trials perhaps

def optimize(bounds, obj_f, optimizer_type, arg = None):
    """
    This function optimizes an objective function "obj_f"
        * "bounds": a tuple that specified the search space
        * "optimizer_type": a class that specifies the type of optimization algorithm being used
        * "arg": a dictionary that specifies the parameters being used 
        for the optimization algorithm
        * returns a list of best output found for each iteration along with the corresponding parameters.
        * or the best output and the corresponding parameter depening on ths history argument
    """
    optimizer = optimizer_type(bounds, obj_f)
    for key, value in arg.items():
        method_name = f"set_{key}"
        method = getattr(optimizer, method_name, None)
        if callable(method):
            method(value)
    
    return optimizer.optimize()

def multi_arg_optimize(bounds, obj_f, optimizer_type, args = None):
    """
    This function optimizes an objective function "obj_f" multiple times given different arg in args
        * "bounds": a tuple that specified the search space
        * "optimizer_type": a class that specifies the type of optimization algorithm being used
        * "args": a list that specifes multiple arg where arg is a dictionary that 
        specifies the parameters being used for the optimization algorithm
        * returns a list of best output found for each iteration and for each arg
        along with the corresponding parameters.
        * or the best output and the corresponding parameter depening on ths history argument
    """
    histories = []
    
    for arg in args:
        optimizer = optimizer_type(bounds, obj_f)
        for key, value in arg.items():
            method_name = f"set_{key}"
            method = getattr(optimizer, method_name, None)
            if callable(method):
                method(value)
        
        histories.append(optimizer.optimize())
    
    return histories