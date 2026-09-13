import numpy as np

# Big-M Simplex Method
# Maximize Z = 40x1 + 30x2 + 50x3 + 45x4

M = 1000000

# Columns:
# x1, x2, x3, x4, s1, s2, a1, a2, RHS

tableau = np.array([
    [2, 1, 3, 2, 1,  0, 0, 0, 40],
    [1, 2, 1, 3, 0, -1, 1, 0, 20],
    [1, 1, 1, 1, 0,  0, 0, 1, 15]
], dtype=float)

# Objective function coefficients
C = np.array([
    40, 30, 50, 45, 0, 0, -M, -M
], dtype=float)

# Variable names
names = ["x1", "x2", "x3", "x4", "s1", "s2", "a1", "a2"]

# Initial basic variables
# s1, a1, a2
basic = [4, 6, 7]


# Function to print tableau
def print_tableau(tab, basic, iteration):

    print("\nITERATION", iteration)

    print(
        f"{'Basic':<10}",
        f"{'x1':>10}",
        f"{'x2':>10}",
        f"{'x3':>10}",
        f"{'x4':>10}",
        f"{'s1':>10}",
        f"{'s2':>10}",
        f"{'a1':>10}",
        f"{'a2':>10}",
        f"{'RHS':>10}"
    )

    for i in range(len(basic)):

        row = tab[i]

        print(
            f"{names[basic[i]]:<10}",
            f"{row[0]:>10.3f}",
            f"{row[1]:>10.3f}",
            f"{row[2]:>10.3f}",
            f"{row[3]:>10.3f}",
            f"{row[4]:>10.3f}",
            f"{row[5]:>10.3f}",
            f"{row[6]:>10.3f}",
            f"{row[7]:>10.3f}",
            f"{row[8]:>10.3f}"
        )


# Big-M Simplex function
def simplex_big_m(A, b, C, basic):

    tab = np.column_stack((A, b)).astype(float)

    iteration = 0

    while True:

        # Cb values of basic variables
        cb = C[basic]

        # Calculate Zj
        zj = cb @ tab[:, :-1]

        # Calculate Cj - Zj
        cj_zj = C - zj

        # Print current tableau
        print_tableau(tab, basic, iteration)

        # Print Cj - Zj
        print("\nCj - Zj:")

        for i in range(len(C)):
            print(
                f"{names[i]} = {cj_zj[i]:.3f}",
                end="   "
            )

        print()

        # Check optimality
        if np.max(cj_zj) <= 1e-9:

            print("\nNo positive Cj - Zj remains.")
            print("Therefore, the solution is OPTIMAL.")

            break

        # Select entering variable
        entering = int(np.argmax(cj_zj))

        print("\nEntering variable:", names[entering])

        # Ratio test
        ratios = []

        for i in range(len(basic)):

            if tab[i, entering] > 1e-9:

                ratio = tab[i, -1] / tab[i, entering]

                ratios.append(ratio)

                print(
                    "Ratio for",
                    names[basic[i]],
                    "=",
                    round(ratio, 3)
                )

            else:

                ratios.append(np.inf)

        # Select leaving variable
        leaving = int(np.argmin(ratios))

        print(
            "Leaving variable:",
            names[basic[leaving]]
        )

        # Pivot element
        pivot = tab[leaving, entering]

        print(
            "Pivot element:",
            round(pivot, 3)
        )

        # Make pivot element 1
        tab[leaving] = tab[leaving] / pivot

        # Make other entries in pivot column 0
        for i in range(len(basic)):

            if i != leaving:

                tab[i] = (
                    tab[i]
                    - tab[i, entering] * tab[leaving]
                )

        # Update basic variable
        basic[leaving] = entering

        iteration += 1

    # Store final solution
    solution = np.zeros(len(C))

    for i, col in enumerate(basic):

        solution[col] = tab[i, -1]

    # Calculate maximum profit
    z = C @ solution

    return solution, z


# Separate coefficient matrix and RHS
A = tableau[:, :-1]
b = tableau[:, -1]

# Solve using Big-M Simplex
solution, z = simplex_big_m(
    A, b, C, basic
)


# Print final answer
print("\nFINAL OPTIMAL SOLUTION")

print(f"x1 = {solution[0]:.2f}")
print(f"x2 = {solution[1]:.2f}")
print(f"x3 = {solution[2]:.2f}")
print(f"x4 = {solution[3]:.2f}")

print(f"\nMaximum Profit = Rs. {z:.2f}")