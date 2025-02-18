from mip import Model, xsum, BINARY, INTEGER, OptimizationStatus
from qubots.base_optimizer import BaseOptimizer
from collections import defaultdict

class MIPBatchOptimizer(BaseOptimizer):
    """
    MIP-based batch scheduling optimizer using Python-MIP and CBC solver.
    This implementation creates start and end time variables, enforces non-overlap 
    for tasks sharing the same resource via binary ordering variables, and handles 
    precedence constraints.
    """
    def __init__(self, time_limit=300):
        self.time_limit = time_limit

    def optimize(self, problem, initial_solution=None, **kwargs):
        m = Model()

        nb_tasks = problem.nb_tasks
        time_horizon = problem.time_horizon

        # Create start time variables for each task
        starts = [m.add_var(var_type=INTEGER, lb=0, ub=time_horizon, name=f"start_{t}") 
                  for t in range(nb_tasks)]
        # Create end time variables for each task
        ends = [m.add_var(var_type=INTEGER, lb=0, ub=time_horizon, name=f"end_{t}") 
                for t in range(nb_tasks)]
        durations = problem.duration

        # Link start and end times: end = start + duration
        for t in range(nb_tasks):
            m.add_constr(ends[t] == starts[t] + durations[t])

        # Non-overlap constraints per resource.
        # For tasks sharing the same resource, one must finish before the other starts.
        resource_tasks = defaultdict(list)
        for t in range(nb_tasks):
            resource = problem.resources[t]
            resource_tasks[resource].append(t)

        M = time_horizon  # Big-M constant
        for resource, tasks in resource_tasks.items():
            # For every pair of tasks on the same resource, add ordering constraints.
            for i in range(len(tasks)):
                for j in range(i + 1, len(tasks)):
                    t1, t2 = tasks[i], tasks[j]
                    # Binary variable: y == 1 if task t1 precedes t2.
                    y = m.add_var(var_type=BINARY, name=f"y_{t1}_{t2}")
                    m.add_constr(starts[t1] + durations[t1] <= starts[t2] + M * (1 - y))
                    m.add_constr(starts[t2] + durations[t2] <= starts[t1] + M * y)

        # Precedence constraints (if any)
        for t in range(nb_tasks):
            for s in problem.successors[t]:
                m.add_constr(ends[t] <= starts[s])

        # Define makespan as the maximum end time
        makespan = m.add_var(var_type=INTEGER, lb=0, ub=time_horizon, name="makespan")
        for t in range(nb_tasks):
            m.add_constr(makespan >= ends[t])
        m.objective = makespan  # minimize the makespan

        # Optimize with the specified time limit
        m.optimize(max_seconds=self.time_limit)

        # Check if we have an optimal or feasible solution
        if m.status in [OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE]:
            return self._extract_solution(m, problem, starts, ends)
        else:
            # If no solution was found, return a random solution with infinite cost.
            return problem.random_solution(), float('inf')

    def _extract_solution(self, m, problem, starts, ends):
        """
        Extracts the solution from the model, grouping tasks into batches 
        per resource based on their start times.
        """
        nb_tasks = problem.nb_tasks
        batches = defaultdict(list)
        for t in range(nb_tasks):
            resource = problem.resources[t]
            s = int(round(starts[t].x))
            e = int(round(ends[t].x))
            batches[resource].append((s, e, t))
        
        batch_schedule = []
        for resource, tasks in batches.items():
            # Sort tasks by start time
            sorted_tasks = sorted(tasks, key=lambda x: x[0])
            current_batch = []
            current_end = -1
            
            for s, e, t in sorted_tasks:
                if s >= current_end:
                    # Start a new batch if there is a gap
                    if current_batch:
                        batch_schedule.append({
                            'resource': resource,
                            'tasks': [task for _, _, task in current_batch],
                            'start': current_batch[0][0],
                            'end': current_batch[-1][1]
                        })
                    current_batch = [(s, e, t)]
                    current_end = e
                else:
                    # Continue the current batch
                    current_batch.append((s, e, t))
                    current_end = max(current_end, e)
            
            if current_batch:
                batch_schedule.append({
                    'resource': resource,
                    'tasks': [task for _, _, task in current_batch],
                    'start': current_batch[0][0],
                    'end': current_end
                })
        return {'batch_schedule': batch_schedule}, m.objective_value
