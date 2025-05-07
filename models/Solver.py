from API.Problem import Problem
from API.Solution import Solution
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
import pandas as pd
import logging


class Solver(ABC):
    def __init__(self, model, method, name, parameters):
        self.logger = logging.getLogger(name)
        logging.basicConfig(
            filename="solver.log",
            filemode="w",
            encoding="utf-8",
            level=logging.INFO,
            # format="%(asctime)s %(message)s",
            # datefmt="%d/%m/%Y %I:%M:%S"
        )
        self.model: Optional = model
        self.solver_method = method
        self.solver_parameters: Dict[str, Any] = parameters
        self.kpi_df = pd.DataFrame(
            columns=[
                "solution_count",
                "delta_treatment_release",
                "treatment_ranges",
                "nb_periods_violated",
                "days_periods_violated",
                "time_regularity",
                "machine_preferences",
                "time",
            ]
        )

    @abstractmethod
    def preprocess_problem_data(self, problem: Problem):
        pass

    @abstractmethod
    def init_model(self, problem: Problem):
        pass

    @abstractmethod
    def add_constraints(self, problem: Problem):
        pass

    @abstractmethod
    def add_objectives(self, problem: Problem):
        pass

    @abstractmethod
    def solve(self, problem: Problem) -> Solution:
        pass

    @abstractmethod
    def retrieve_solution(self, problem: Problem) -> Solution:
        pass
