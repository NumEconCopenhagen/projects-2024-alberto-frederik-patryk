import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd

class OilCartelModel:
    def __init__(self, TD0=18, S0=6.5, CS0=0, R0=500, delta=0.05, N=40, slope_demand=0.13,
                 slope_supply=0.1, alpha=0.02, S_hat=7, g=0.015, c=0.5, start_year=1975):
        """
        Initializes Pindyck's OPEC Cartel Model with default parameters. The model is from the article: 
        
        Pindyck, Robert S. "Gains to producers from the cartelization of exhaustible resources." 
        The Review of Economics and Statistics (1978): 238-251.
        https://dspace.mit.edu/bitstream/handle/1721.1/27836/MIT-EL-76-012WP-03804475.pdf?sequence=1

        Args:
        TD0 (float): Initial total demand.
        S0 (float): Initial supply from competitive fringe.
        CS0 (float): Initial cumulative supply from competitive fringe.
        R0 (float): Initial cartel reserves.
        delta (float): Discount factor.
        N (int): Number of periods to simulate.
        slope_demand (float): Slope of the demand curve.
        slope_supply (float): Slope of the supply curve.
        alpha (float): Rate of depletion for the competitive fringe
        S_hat (float): Average annual competitive fringe production
        g (float): Growth rate in demand.
        c (float): Initial average production cost per barrel.
        start_year (int): Starting year for the plots.
        """
        self.TD0 = TD0
        self.S0 = S0
        self.CS0 = CS0
        self.R0 = R0
        self.delta = delta
        self.N = N
        self.slope_demand = slope_demand
        self.slope_supply = slope_supply
        self.alpha = alpha
        self.S_hat = S_hat
        self.g = g
        self.c = c
        self.start_year = start_year
        self.P_initial = np.full(N, 14)  # Initial price guess

    def calculate_next(self, TD_prev, S_prev, CS_prev, R_prev, P_t, t):
        """
        Calculates the next state of the market given the previous state and current price.

        Returns:
        Tuple of (TD_t, S_t, CS_t, D_t, R_t) representing the new state.
        """
        TD_t = 1 - self.slope_demand * P_t + 0.87 * TD_prev + 2.3 * (1 + self.g)**(t + 1)
        S_t = (1.1 + self.slope_supply * P_t) * (1 + self.alpha)**(-CS_prev / self.S_hat) + 0.75 * S_prev
        CS_t = CS_prev + S_t
        D_t = TD_t - S_t
        R_t = R_prev - D_t
        return TD_t, S_t, CS_t, D_t, R_t

    def objective_function(self, P, return_full_data=False):
        """
        Objective function to be maximized (minimize the negative) for optimizing oil prices over time.

        Returns:
        The optimal price trajectory, or other variables if `return_full_data` is True.
        """
        TD_t, S_t, CS_t, D_t, R_t = self.TD0, self.S0, self.CS0, 0, self.R0
        W = 0
        demands = []
        total_demands = []
        reserves = []
        for t in range(self.N):
            TD_t, S_t, CS_t, D_t, R_t = self.calculate_next(TD_t, S_t, CS_t, R_t, P[t], t)
            if R_t <= 1:
                R_t = 1
            demands.append(D_t)
            total_demands.append(TD_t)
            reserves.append(R_t)
            W += (1 / (1 + self.delta)**(t + 1)) * (P[t] - self.R0 * self.c / R_t) * D_t
        if return_full_data:
            return -W, demands, total_demands, reserves
        else:
            return -W

    def run_model(self):
        """
        Runs the optimization and visualizes the results.
        """
        result = minimize(self.objective_function, self.P_initial, method='Nelder-Mead', options={'maxiter': 10000, 'adaptive': True})
        optimized_prices = result.x
        _, optimal_demands, optimal_total_demands, optimal_reserves = self.objective_function(optimized_prices, return_full_data=True)
        
        # Visualization
        self.display_results(optimized_prices, optimal_demands, optimal_total_demands, optimal_reserves)

    def display_results(self, optimized_prices, optimal_demands, optimal_total_demands, optimal_reserves):
        """
        Creates and displays a DataFrame and plots for the simulation results.
        """
        years = np.arange(self.start_year, self.start_year + self.N)
        data = {
            'Year': years,
            'Optimized Prices ($)': optimized_prices,
            'Demand for Cartel Oil (D_t)': optimal_demands,
            'Total Oil Demand (TD_t)': optimal_total_demands,
            'Cartel Reserves (R_t)': optimal_reserves
        }
        df = pd.DataFrame(data)
        blankIndex = [''] * len(df)
        df.index = blankIndex
        display(df.round(2))

        plt.figure(figsize=(14, 7))
        plt.subplot(1, 2, 1)
        plt.plot(years, optimized_prices, 'b-', marker='o', label='Optimized Oil Prices')
        plt.title('Optimized Oil Prices Over Time')
        plt.xlabel('Year')
        plt.ylabel('Price per Barrel ($)')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(years, optimal_demands, 'r-', marker='o', label='Demand for Cartel Oil')
        plt.title('Demand for Cartel Oil Over Time')
        plt.xlabel('Year')
        plt.ylabel('Billions of Barrels')
        plt.tight_layout()
        plt.grid(True)
        plt.show()

        
"******************************************************************************************************************************"  
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd

class OilCartelModelExploration:
    """
    An extended version of Pindyck's model with oil exploration activities.

    New Attributes:
        P_base (float): Base price threshold for exploration activities.
        phi (float): Sensitivity coefficient for price above base affecting exploration.
        gamma (float): Exponential factor for reserve impact on exploration.
        K_fix (float): Fixed costs of exploration (set to zero).
        K_var (float): Variable costs of exploration.
        lambd (float): Cost exponent, affecting nonlinear cost increases with more exploration.
        P_initial (numpy.ndarray): Initial guess for oil prices over the simulated periods.

    Methods:
        calculate_exploration(P_t, R_t_minus_1): Calculates exploration based on current price and previous reserves.
        calculate_next(TD_prev, S_prev, CS_prev, R_prev, P_t, t): Updates the state for the next time period.
        objective_function(P, return_full_data=False): Computes the negative of the total welfare to be minimized.
        run_model(): Runs the optimization model to find the optimal price path.
        display_results(optimized_prices, optimal_demands, optimal_total_demands, optimal_reserves, optimal_exploration_costs): Displays the results of the simulation.
    """
    def __init__(self, TD0=18, S0=6.5, CS0=0, R0=500, delta=0.05, N=40, slope_demand=0.13,
                 slope_supply=0.1, alpha=0.02, S_hat=7, g=0.015, c=0.5, start_year=1975, P_base=16, phi=0.75, gamma=750, K_fix=0, K_var=0.1, lambd=1.1):
        self.TD0 = TD0
        self.S0 = S0
        self.CS0 = CS0
        self.R0 = R0
        self.delta = delta
        self.N = N
        self.slope_demand = slope_demand
        self.slope_supply = slope_supply
        self.alpha = alpha
        self.S_hat = S_hat
        self.g = g
        self.c = c
        self.start_year = start_year
        self.P_base = P_base
        self.phi = phi
        self.gamma = gamma
        self.K_fix = K_fix
        self.K_var = K_var
        self.lambd = lambd
        self.P_initial = np.full(N, 14)  # Initial guess for oil prices

    def calculate_exploration(self, P_t, R_t_minus_1):
        """
        Calculates the exploration efforts and associated costs based on the current oil price and previous reserves.

        Args:
            P_t (float): Current period oil price.
            R_t_minus_1 (float): Oil reserves from the previous period.

        Returns:
            tuple: Contains exploration volume and associated costs for the period.
        """
        Et = self.phi * (P_t - self.P_base) * self.gamma * (1 / R_t_minus_1)
        Et = max(Et, 0)  # Ensure exploration is non-negative
        ECt = self.K_fix + self.K_var * Et**self.lambd
        return Et, ECt

    def calculate_next(self, TD_prev, S_prev, CS_prev, R_prev, P_t, t):
        """
        Calculates the next state of the market including demand, supply, reserves, and exploration.

        Args:
            TD_prev (float): Total demand from the previous period.
            S_prev (float): Supply from non-cartel producers from the previous period.
            CS_prev (float): Cumulative supply from non-cartel producers up to the previous period.
            R_prev (float): Cartel reserves from the previous period.
            P_t (float): Current period oil price.
            t (int): Current time period index.

        Returns:
            tuple: Updated values for total demand, non-cartel supply, cumulative non-cartel supply, cartel demand, reserves, exploration costs, and exploration volume.
        """
        Et, ECt = self.calculate_exploration(P_t, R_prev)
        TD_t = 1 - self.slope_demand * P_t + 0.87 * TD_prev + 2.3 * (1 + self.g)**(t + 1)
        S_t = (1.1 + self.slope_supply * P_t) * (1 + self.alpha)**(-CS_prev / self.S_hat) + 0.75 * S_prev
        CS_t = CS_prev + S_t
        D_t = TD_t - S_t
        R_t = R_prev - D_t + Et
        return TD_t, S_t, CS_t, D_t, R_t, ECt, Et

    def objective_function(self, P, return_full_data=False):
        """
        Objective function that calculates the total negative welfare to be minimized by the optimizer.

        Args:
            P (numpy.ndarray): Array of prices to optimize.
            return_full_data (bool): Flag to determine if detailed results should be returned.

        Returns:
            float or tuple: Total negative welfare or detailed simulation data depending on `return_full_data`.
        """
        TD_t, S_t, CS_t, D_t, R_t = self.TD0, self.S0, self.CS0, 0, self.R0
        W = 0
        demands = []
        total_demands = []
        reserves = []
        exploration_costs = []
        explorations = []  # To store exploration amounts
        for t in range(self.N):
            TD_t, S_t, CS_t, D_t, R_t, ECt, Et = self.calculate_next(TD_t, S_t, CS_t, R_t, P[t], t)
            if R_t <= 1:
                R_t = 1
            demands.append(D_t)
            total_demands.append(TD_t)
            reserves.append(R_t)
            exploration_costs.append(ECt)
            explorations.append(Et)  # Append exploration to the list
            W += (1 / (1 + self.delta)**(t + 1)) * ((P[t] * D_t - 250 / R_t * D_t) - ECt)
        if return_full_data:
            return -W, demands, total_demands, reserves, exploration_costs, explorations
        else:
            return -W

    def run_model(self):
        """
        Executes the model by running the optimization and then displaying the results.
        """
        result = minimize(self.objective_function, self.P_initial, method='Nelder-Mead', options={'maxiter': 10000, 'adaptive': True})
        optimized_prices = result.x
        _, optimal_demands, optimal_total_demands, optimal_reserves, optimal_exploration_costs, optimal_explorations = self.objective_function(optimized_prices, return_full_data=True)
        self.display_results(optimized_prices, optimal_demands, optimal_total_demands, optimal_reserves, optimal_exploration_costs, optimal_explorations)

    def display_results(self, optimized_prices, optimal_demands, optimal_total_demands, optimal_reserves, optimal_exploration_costs, optimal_explorations):
        """
        Displays the results of the simulation in both tabular and graphical formats.
    
        Args:
        optimized_prices (numpy.ndarray): Array of optimized oil prices per period.
        optimal_demands (list): List of demand for cartel oil per period.
        optimal_total_demands (list): List of total oil demand per period.
        optimal_reserves (list): List of oil reserves per period.
        optimal_exploration_costs (list): List of exploration costs per period.
        optimal_explorations (list): List of exploration volumes per period.
        """
        years = np.arange(self.start_year, self.start_year + self.N)
        data = {
        'Year': years,
        'Optimized Prices ($)': optimized_prices,
        'Demand for Cartel Oil (D_t)': optimal_demands,
        'Total Oil Demand (TD_t)': optimal_total_demands,
        'Cartel Reserves (R_t)': optimal_reserves,
        'Exploration Costs (EC_t)': optimal_exploration_costs,
        'Exploration Volume (E_t)': optimal_explorations
        }
        df = pd.DataFrame(data)
        blankIndex = [''] * len(df)
        df.index = blankIndex
        display(df.round(2))

        # Plot setup
        plt.figure(figsize=(21, 7))  # Adjust the figure size to accommodate three subplots
        # First subplot for Optimized Oil Prices
        plt.subplot(1, 3, 1)
        plt.plot(years, optimized_prices, 'b-', marker='o', label='Optimized Oil Prices')
        plt.title('Optimized Oil Prices Over Time')
        plt.xlabel('Year')
        plt.ylabel('Price per Barrel ($)')
        plt.grid(True)

        # Second subplot for Demand for Cartel Oil
        plt.subplot(1, 3, 2)
        plt.plot(years, optimal_demands, 'r-', marker='o', label='Demand for Cartel Oil')
        plt.title('Demand for Cartel Oil Over Time')
        plt.xlabel('Year')
        plt.ylabel('Billions of Barrels')
        plt.grid(True)

        # Third subplot for Exploration Volume
        plt.subplot(1, 3, 3)
        plt.plot(years, optimal_explorations, 'g-', marker='o', label='Exploration Volume')
        plt.title('Exploration Volume Over Time')
        plt.xlabel('Year')
        plt.ylabel('Billions of Barrels Explored')
        plt.grid(True)

        plt.tight_layout()  # Adjust subplots to fit into the figure area.
        plt.show()
