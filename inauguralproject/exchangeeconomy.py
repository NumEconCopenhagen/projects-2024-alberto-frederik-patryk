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

    def utility_A(self,x1A,x2A):
        # Cobb-Douglas utility function of consumer A
        return x1A ** self.par.alpha * x2A ** (1 - self.par.alpha)

    def utility_B(self,x1B,x2B):
        # Cobb-Douglas utility function of consumer B
        return x1B ** self.par.beta * x2B ** (1 - self.par.beta)

    def demand_A(self,p1):
        # Demand function of consumer A
        x1A_star = self.par.alpha * (p1 * self.par.w1A + self.par.p2 * self.par.w2A) / p1
        x2A_star = (1 - self.par.alpha) * (p1 * self.par.w1A + self.par.p2 * self.par.w2A) / self.par.p2
        return x1A_star, x2A_star

    def demand_B(self,p1):
        # Demand function of consumer B
        x1B_star = self.par.beta * (p1 * self.par.w1B + self.par.p2 * self.par.w2B) / p1
        x2B_star = (1 - self.par.beta) * (p1 * self.par.w1B + self.par.p2 * self.par.w2B) / self.par.p2
        return x1B_star, x2B_star

    def check_market_clearing(self,p1):

        par = self.par

        x1A,x2A = self.demand_A(p1)
        x1B,x2B = self.demand_B(p1)

        eps1 = x1A-par.w1A + x1B-(par.w1B)
        eps2 = x2A-par.w2A + x2B-(par.w2B)

        return eps1,eps2

    def draw_W(self, num_elements=50):
        # Generate a set W as a list of num_elements tuples (omega_1A, omega_2A) 
        np.random.seed(5)  # For reproducibility
        self.W = np.random.uniform(0, 1, (num_elements, 2))
        # Convert to list of tuples
        list_of_tuples = [tuple(row) for row in self.W]
        return list_of_tuples

    def demand_A2(self, p1, omega_1A, omega_2A):
        # Demand function of consumer A with dynamic endowments
        x1A_eq = self.par.alpha * (p1 * omega_1A + self.par.p2 * omega_2A) / p1
        x2A_eq = (1 - self.par.alpha) * (p1 * omega_1A + self.par.p2 * omega_2A) / self.par.p2
        return x1A_eq, x2A_eq

    def demand_B2(self, p1, omega_1B, omega_2B):
        # Demand function of consumer B with dynamic endowments
        x1B_eq = self.par.beta * (p1 * omega_1B + self.par.p2 * omega_2B) / p1
        x2B_eq = (1 - self.par.beta) * (p1 * omega_1B + self.par.p2 * omega_2B) / self.par.p2
        return x1B_eq, x2B_eq
    
    def market_clearing_error_for_p1(self, p1, omega_1A, omega_2A):
        # Calculate the squared sum of market clearing errors for a given price p1 and dynamic endowments.
        x1_A_eq, x2_A_eq = self.demand_A2(p1, omega_1A, omega_2A)
        # Calculate demand for B using modified omega values for B
        omega_1B, omega_2B = 1 - omega_1A, 1 - omega_2A
        x1_B_eq, x2_B_eq = self.demand_B2(p1, omega_1B, omega_2B)
        error1 = (x1_A_eq + x1_B_eq - 1) ** 2
        error2 = (x2_A_eq + x2_B_eq - 1) ** 2
        return error1 + error2
  
    def find_market_clearing_price(self, omega_1A, omega_2A):
        # Find the market clearing price for given endowments by minimizing market clearing errors.
        result = minimize(self.market_clearing_error_for_p1, x0=[1], args=(omega_1A, omega_2A), bounds=[(0.01, None)])
        if result.success:
            return result.x[0]
        else:
            raise ValueError("Failed to find a market clearing price.")


    def find_and_plot_equilibria(self):
        # Find equilibrium allocations for each pair in W and plot them in the Edgeworth box.
        equilibria = []
        for omega_1A, omega_2A in self.draw_W():
            p1_star = self.find_market_clearing_price(omega_1A, omega_2A)
            x1_A_eq, x2_A_eq = self.demand_A2(p1_star, omega_1A, omega_2A)
            # Store the equilibrium allocation
            equilibria.append((x1_A_eq, x2_A_eq))
    
        # Unpack the equilibrium allocations for plotting
        x1_A_eq, x2_A_eq = zip(*equilibria)
        
        # Plotting
        plt.figure(figsize=(8, 6))
        plt.scatter(x1_A_eq, x2_A_eq, c='blue', label='Equilibrium Allocations', alpha = 0.5)
        plt.xlabel('$x_{1A}^*$')
        plt.ylabel('$x_{2A}^*$')
        plt.title('Edgeworth Box of equilibrium allocations')
        plt.legend()
        plt.grid(True)
        plt.show()