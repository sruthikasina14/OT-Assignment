import numpy as np

# Transportation cost matrix
cost = np.array([
    [9, 8, 11, 7],
    [9, 14, 4, 4],
    [5, 17, 6, 19]
], dtype=float)

# Supply and demand
supply = [20, 30, 25]
demand = [10, 25, 20, 20]

m = len(supply)
n = len(demand)

def print_allocation(allocation):

    print("       W1   W2   W3   W4")

    for i in range(m):
        print(
            "F" + str(i + 1),
            "    ",
            int(allocation[i][0]),
            "  ",
            int(allocation[i][1]),
            "  ",
            int(allocation[i][2]),
            "  ",
            int(allocation[i][3])
        )

def calculate_cost(allocation):

    total = 0

    for i in range(m):
        for j in range(n):
            total += allocation[i][j] * cost[i][j]

    return total

# VOGEL'S APPROXIMATION METHOD

def vogel_approximation():

    s = supply.copy()
    d = demand.copy()

    allocation = np.zeros((m, n))

    active_rows = [True] * m
    active_cols = [True] * n

    step = 1

    print("\nVOGEL'S APPROXIMATION METHOD")

    while sum(s) > 0:

        row_penalty = [-1] * m
        col_penalty = [-1] * n

        # Row penalties
        for i in range(m):

            if active_rows[i]:

                values = []

                for j in range(n):

                    if active_cols[j]:
                        values.append(cost[i][j])

                values.sort()

                if len(values) >= 2:
                    row_penalty[i] = values[1] - values[0]

                elif len(values) == 1:
                    row_penalty[i] = values[0]

        # Column penalties
        for j in range(n):

            if active_cols[j]:

                values = []

                for i in range(m):

                    if active_rows[i]:
                        values.append(cost[i][j])

                values.sort()

                if len(values) >= 2:
                    col_penalty[j] = values[1] - values[0]

                elif len(values) == 1:
                    col_penalty[j] = values[0]

        print("\nStep", step)

        print("Row Penalties:", row_penalty)
        print("Column Penalties:", col_penalty)

        max_row = max(row_penalty)
        max_col = max(col_penalty)

        # Select row or column with largest penalty
        if max_row >= max_col:

            i = row_penalty.index(max_row)

            minimum = float("inf")
            j = -1

            for col in range(n):

                if active_cols[col]:

                    if cost[i][col] < minimum:
                        minimum = cost[i][col]
                        j = col

        else:

            j = col_penalty.index(max_col)

            minimum = float("inf")
            i = -1

            for row in range(m):

                if active_rows[row]:

                    if cost[row][j] < minimum:
                        minimum = cost[row][j]
                        i = row

        # Allocation
        amount = min(s[i], d[j])

        allocation[i][j] = amount

        print(
            "Allocation: F" + str(i + 1) +
            " -> W" + str(j + 1) +
            " = " + str(int(amount))
        )

        s[i] -= amount
        d[j] -= amount

        if s[i] == 0:
            active_rows[i] = False

        if d[j] == 0:
            active_cols[j] = False

        step += 1

    return allocation

# FIND CLOSED LOOP FOR MODI

def find_loop(allocation, start):

    path = [start]

    def search(current, move_row):

        i, j = current

        if len(path) >= 4 and current == start:
            return True

        if move_row:

            for col in range(n):

                if col != j:

                    if allocation[i][col] > 0 or (i, col) == start:

                        if (i, col) == start and len(path) >= 4:
                            path.append((i, col))
                            return True

                        if (i, col) not in path:

                            path.append((i, col))

                            if search((i, col), False):
                                return True

                            path.pop()

        else:

            for row in range(m):

                if row != i:

                    if allocation[row][j] > 0 or (row, j) == start:

                        if (row, j) == start and len(path) >= 4:
                            path.append((row, j))
                            return True

                        if (row, j) not in path:

                            path.append((row, j))

                            if search((row, j), True):
                                return True

                            path.pop()

        return False

    if search(start, True):
        return path

    return None

# MODI METHOD

def modi(allocation):

    iteration = 1

    print("\n\nMODIFIED DISTRIBUTION METHOD (MODI)")

    while True:

        # Calculate u and v
        u = [None] * m
        v = [None] * n

        u[0] = 0

        changed = True

        while changed:

            changed = False

            for i in range(m):
                for j in range(n):

                    if allocation[i][j] > 0:

                        if u[i] is not None and v[j] is None:

                            v[j] = cost[i][j] - u[i]
                            changed = True

                        elif v[j] is not None and u[i] is None:

                            u[i] = cost[i][j] - v[j]
                            changed = True

        print("\nStep", iteration)

        print("u values:", u)
        print("v values:", v)

        # Calculate delta values
        delta = np.zeros((m, n))

        for i in range(m):
            for j in range(n):

                if allocation[i][j] == 0:

                    delta[i][j] = cost[i][j] - (u[i] + v[j])

                else:

                    delta[i][j] = 0

        print("\nDelta (Opportunity Cost) Table:")

        print("       W1   W2   W3   W4")

        for i in range(m):

            print(
                "F" + str(i + 1),
                "    ",
                int(delta[i][0]),
                "  ",
                int(delta[i][1]),
                "  ",
                int(delta[i][2]),
                "  ",
                int(delta[i][3])
            )

        # Find most negative delta
        min_delta = 0
        enter_i = -1
        enter_j = -1

        for i in range(m):
            for j in range(n):

                if allocation[i][j] == 0:

                    if delta[i][j] < min_delta:

                        min_delta = delta[i][j]
                        enter_i = i
                        enter_j = j

        # Optimality check
        if enter_i == -1:

            print("\nAll Delta values are >= 0.")
            print("Therefore, the solution is OPTIMAL.")

            break

        print(
            "\nEntering Cell: F" +
            str(enter_i + 1) +
            "-W" +
            str(enter_j + 1)
        )

        # Find closed loop
        loop = find_loop(
            allocation,
            (enter_i, enter_j)
        )

        print("\nClosed Loop:")

        for k in range(len(loop) - 1):

            i, j = loop[k]

            if k % 2 == 0:
                print(
                    "+ F" + str(i + 1) +
                    "-W" + str(j + 1)
                )

            else:
                print(
                    "- F" + str(i + 1) +
                    "-W" + str(j + 1)
                )

        # Calculate theta
        theta = float("inf")

        for k in range(1, len(loop) - 1, 2):

            i, j = loop[k]

            if allocation[i][j] < theta:
                theta = allocation[i][j]

        print("\nTheta =", int(theta))

        # Update allocation
        for k in range(len(loop) - 1):

            i, j = loop[k]

            if k % 2 == 0:

                allocation[i][j] += theta

            else:

                allocation[i][j] -= theta

        print("\nAllocation after Step", iteration, ":")

        print_allocation(allocation)

        print(
            "\nTransportation Cost = Rs.",
            int(calculate_cost(allocation))
        )

        iteration += 1

    return allocation

# MAIN PROGRAM

print("TRANSPORTATION PROBLEM")

print("\nCost Matrix:")
print(cost)

print("\nSupply:", supply)
print("Demand:", demand)


# VAM
allocation = vogel_approximation()

print("\n\nVAM INITIAL BASIC FEASIBLE SOLUTION")

print_allocation(allocation)

print(
    "\nInitial Transportation Cost = Rs.",
    int(calculate_cost(allocation))
)

# MODI
allocation = modi(allocation)

print("\n\nFINAL OPTIMAL ALLOCATION")

print_allocation(allocation)

print(
    "\nMinimum Transportation Cost = Rs.",
    int(calculate_cost(allocation))
)