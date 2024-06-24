# exchange_economy.py

from types import SimpleNamespace
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class ExchangeEconomyClass:
    def __init__(self):
        par = self.par = SimpleNamespace()

        # a. Cobb-Douglas preference parameters of consumer A and B
        par.alpha = 1/3
        par.beta = 2/3

        # b. Initial endowments of consumer A and B
        par.w1A = 0.8
        par.w2A = 0.3
        par.w1B = 1 - par.w1A
        par.w2B = 1 - par.w2A

        # c. Numeraire
        par.p2 = 1

    def utility_A(self, x1A, x2A):
        """Cobb-Douglas utility function of consumer A."""
        return x1A ** self.par.alpha * x2A ** (1 - self.par.alpha)

    def utility_B(self, x1B, x2B):
        """Cobb-Douglas utility function of consumer B."""
        return x1B ** self.par.beta * x2B ** (1 - self.par.beta)

    def demand_A(self, p1):
        """Demand function of consumer A."""
        x1A_star = self.par.alpha * (p1 * self.par.w1A + self.par.p2 * self.par.w2A) / p1
        x2A_star = (1 - self.par.alpha) * (p1 * self.par.w1A + self.par.p2 * self.par.w2A) / self.par.p2
        return x1A_star, x2A_star

    def demand_B(self, p1):
        """Demand function of consumer B."""
        x1B_star = self.par.beta * (p1 * self.par.w1B + self.par.p2 * self.par.w2B) / p1
        x2B_star = (1 - self.par.beta) * (p1 * self.par.w1B + self.par.p2 * self.par.w2B) / self.par.p2
        return x1B_star, x2B_star

    def check_market_clearing(self, p1):
        """Check market clearing condition."""
        par = self.par
        x1A, x2A = self.demand_A(p1)
        x1B, x2B = self.demand_B(p1)
        eps1 = x1A - par.w1A + x1B - par.w1B
        eps2 = x2A - par.w2A + x2B - par.w2B
        return eps1, eps2

    def draw_W(self, num_elements=50):
        """Generate a set W of num_elements tuples (omega_1A, omega_2A)."""
        np.random.seed(5)  # For reproducibility
        self.W = np.random.uniform(0, 1, (num_elements, 2))
        list_of_tuples = [tuple(row) for row in self.W]
        return list_of_tuples

    def demand_A2(self, p1, omega_1A, omega_2A):
        """Demand function of consumer A with dynamic endowments."""
        x1A_eq = self.par.alpha * (p1 * omega_1A + self.par.p2 * omega_2A) / p1
        x2A_eq = (1 - self.par.alpha) * (p1 * omega_1A + self.par.p2 * omega_2A) / self.par.p2
        return x1A_eq, x2A_eq

    def demand_B2(self, p1, omega_1B, omega_2B):
        """Demand function of consumer B with dynamic endowments."""
        x1B_eq = self.par.beta * (p1 * omega_1B + self.par.p2 * omega_2B) / p1
        x2B_eq = (1 - self.par.beta) * (p1 * omega_1B + self.par.p2 * omega_2B) / self.par.p2
        return x1B_eq, x2B_eq

    def market_clearing_error_for_p1(self, p1, omega_1A, omega_2A):
        """Calculate the squared sum of market clearing errors for a given price p1 and dynamic endowments."""
        x1_A_eq, x2_A_eq = self.demand_A2(p1, omega_1A, omega_2A)
        omega_1B, omega_2B = 1 - omega_1A, 1 - omega_2A
        x1_B_eq, x2_B_eq = self.demand_B2(p1, omega_1B, omega_2B)
        error1 = (x1_A_eq + x1_B_eq - 1) ** 2
        error2 = (x2_A_eq + x2_B_eq - 1) ** 2
        return error1 + error2

    def find_market_clearing_price(self, omega_1A, omega_2A):
        """Find the market clearing price for given endowments by minimizing market clearing errors."""
        result = minimize(self.market_clearing_error_for_p1, x0=[1], args=(omega_1A, omega_2A), bounds=[(0.01, None)])
        if result.success:
            return result.x[0]
        else:
            raise ValueError("Failed to find a market clearing price.")

    def find_and_plot_equilibria(self):
        """Find equilibrium allocations for each pair in W and plot them in the Edgeworth box."""
        equilibria = []
        for omega_1A, omega_2A in self.draw_W():
            p1_star = self.find_market_clearing_price(omega_1A, omega_2A)
            x1_A_eq, x2_A_eq = self.demand_A2(p1_star, omega_1A, omega_2A)
            equilibria.append((x1_A_eq, x2_A_eq))
        
        x1_A_eq, x2_A_eq = zip(*equilibria)
        plt.figure(figsize=(8, 6))
        plt.scatter(x1_A_eq, x2_A_eq, c='blue', label='Equilibrium Allocations', alpha=0.5)
        plt.xlabel('$\omega_1^A$ (Good 1 consumed by A)')
        plt.ylabel('$\omega_2^A$ (Good 2 consumed by A)')
        plt.title('Edgeworth Box of Equilibrium Allocations')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_set_C(self, x1A_list, x2A_list):
        """Plot the combinations of (x1A, x2A) that make up the set C in the Edgeworth box."""
        fig = plt.figure(frameon=False, figsize=(6, 6), dpi=100)
        ax_A = fig.add_subplot(1, 1, 1)
        ax_A.set_xlabel("$x_1^A$")
        ax_A.set_ylabel("$x_2^A$")

        temp = ax_A.twinx()
        temp.set_ylabel("$x_2^B$")
        ax_B = temp.twiny()
        ax_B.set_xlabel("$x_1^B$")
        ax_B.invert_xaxis()
        ax_B.invert_yaxis()

        # Plot the points and endowment
        ax_A.scatter(x1A_list, x2A_list, s=4, marker='o', color='green', label='Pareto improvements')
        ax_A.scatter(self.par.w1A, self.par.w2A, s=4, marker='o', color='black', label='Endowment')

        # Plot the box limits
        w1bar = 1.0
        w2bar = 1.0
        ax_A.plot([0, w1bar], [0, 0], lw=2, color='black')
        ax_A.plot([0, w1bar], [w2bar, w2bar], lw=2, color='black')
        ax_A.plot([0, 0], [0, w2bar], lw=2, color='black')
        ax_A.plot([w1bar, w1bar], [0, w2bar], lw=2, color='black')

        # Set the limits and show the plot
        ax_A.set_xlim([-0.1, w1bar + 0.1])
        ax_A.set_ylim([-0.1, w2bar + 0.1])
        ax_B.set_xlim([w1bar + 0.1, -0.1])
        ax_B.set_ylim([w2bar + 0.1, -0.1])
        ax_A.legend(frameon=True, loc='upper right', bbox_to_anchor=(1.75, 1.0))
        plt.title('The Set C in the Edgeworth Box')
        plt.show()

    def plot_market_clearing_errors(self, p1_array, errors, intersection):
        """Plot the errors in market clearing conditions and the market clearing price."""
        plt.figure(figsize=(6, 6))
        plt.plot(p1_array, errors[0], color='black', label='$\epsilon_1$',)
        plt.plot(p1_array, errors[1], color='red', label='$\epsilon_2$',)
        plt.scatter(intersection, 0, s=40, marker='o', color='green', label='Market Clearing Price')
        plt.xlabel('$p_1$')
        plt.ylabel('Error')
        plt.title('Errors in Market Clearing Conditions')
        plt.legend(frameon=True, loc='upper right', bbox_to_anchor=(1.75, 1.0))
        plt.grid(True)
        plt.show()

    def plot_optimal_allocations_A_B(self, allocation_A, allocation_B):
        """Plot the optimal allocations for consumers A and B."""
        plt.figure(figsize=(8, 6))
        plt.scatter(allocation_A[0], allocation_A[1], color='blue', label='Optimal Allocation A')
        plt.scatter(allocation_B[0], allocation_B[1], color='red', label='Optimal Allocation B')
        plt.xlabel('$x_{1A}$, $x_{1B}$')
        plt.ylabel('$x_{2A}$, $x_{2B}$')
        plt.title('Optimal Allocations for A and B')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_utility_allocations(self, x1A_agg, x2A_agg):
        """Plot the optimal utility allocations for consumers A and B."""
        plt.figure(figsize=(8, 6))

        plt.subplot(2, 1, 1)
        plt.scatter(x1A_agg, x2A_agg, color='blue', label='Optimal Allocation A')
        plt.xlabel('$x_{1A}$')
        plt.ylabel('$x_{2A}$')
        plt.title('Optimal Allocation for A')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.scatter(1 - x1A_agg, 1 - x2A_agg, color='red', label='Optimal Allocation B')
        plt.xlabel('$x_{1B}$')
        plt.ylabel('$x_{2B}$')
        plt.title('Optimal Allocation for B')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def plot_edgeworth_box(self, allocations, x1A_list, x2A_list):
        """Plot the allocations in the Edgeworth box."""
        plt.figure(figsize=(12, 12))
        ax = plt.subplot(1, 1, 1)

        # Plot the Pareto improvements and endowment for Consumer A
        ax.scatter(x1A_list, x2A_list, s=10, marker='o', color='yellow', label='Pareto improvements')
        ax.scatter(self.par.w1A, self.par.w2A, s=30, marker='o', color='black', label='Endowment (A)')

        # Plot the allocations specifically for Consumer A
        for question, (x1A, x2A) in allocations.items():
            ax.scatter(x1A, x2A, label=question)

        # Set labels and invert the axes for Consumer B
        ax.set_xlabel("$x_1^A$ (Good 1 consumed by A)")
        ax.set_ylabel("$x_2^A$ (Good 2 consumed by A)")

        # Adding secondary axes for Consumer B with reversed direction
        ax_top = ax.twiny()
        ax_right = ax.twinx()
        ax_top.set_xlabel("$x_1^B$")
        ax_right.set_ylabel("$x_2^B$")
        ax_top.set_xlim(1, 0)
        ax_right.set_ylim(1, 0)

        # Set limits
        w1bar = 1.0
        w2bar = 1.0
        ax.set_xlim([0, w1bar])
        ax.set_ylim([0, w2bar])
        ax_top.set_xlim([w1bar, 0])
        ax_right.set_ylim([w2bar, 0])

        # Add a legend and title
        ax.legend(loc='upper right')
        plt.title('Allocations in the Edgeworth Box')
        plt.text(0.2, 0.8, "allocations closer to the centre of the yellow region are more optimal")
        plt.show()

    def plot_initial_endowments(self):
        """Plot the randomly drawn initial endowments for goods."""
        W = self.draw_W()
        plt.figure(figsize=(8, 6))
        plt.scatter(*zip(*W), c='blue', marker='o', s=20, label='Elements of W')
        plt.title('50 Randomly Drawn Initial Endowments for Goods')
        plt.xlabel('$\omega_1^A$ (Consumer A initial endowment of good 1)')
        plt.ylabel('$\omega_2^A$ (Consumer A initial endowment of good 2)')
        plt.grid(True)
        plt.legend()
        plt.show()
