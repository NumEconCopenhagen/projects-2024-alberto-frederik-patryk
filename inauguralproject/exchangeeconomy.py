from types import SimpleNamespace

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
