"""
Created by Calum Hand based on `test_env_fun.py`, I've moved it into a class and
functionalised the repeating code blocks to make it easier for further refactoring.
I think if we can move the budget into this class rather than being tracked by the AMI that would be easier.
Also think if we can get the STATUS recording things then that will be simpler as already exists in AMI natively.
Haven't been able to run this as I don't have your test_toy_AME so I doubt this would run first time but this
overall structure will make it easier to implement in OOP fashion.
"""

import numpy as np


class ParallelScreener(object):

    def __init__(self, model, y, z, cz, cy, nthreads):
        self.model = model
        self.y = y  # np.array(), expensive test data
        self.z = z  # np.array(), cheap test data
        self.cz = cz  # float, cost of cheap
        self.cy = cy  # float, cost of expensive
        self.nthreads = nthreads  # int, number of threads to work on
        self.history = []  # lists to store simulation history
        self.workers = [(0, 0)] * self.nthreads  # now follow controller, set up initial jobs separately
        self.finish_time = np.zeros(self.nthreads)  # times the workers will finish at, here is all zero


    def _screener_init(self):
        """
        starts by selecting a material and performing cheap and expensive test on it
        """
        subject = 0
        self.model.uu.remove(subject)  # selects from untested and performs e
        self.model.tt.append(subject)
        self.model.b -= (self.cz + self.cy)  # update budget
        self.model.z[subject] = self.z[subject]  # update model
        self.model.y[subject] = self.y[subject]


    def _select_and_run_experiment(self, i):
        """
        Passed model selects a material to sample.
        If the material has not been tested before then a cheap test is run, otherwise run expensive.
        After each test, the budget is updated (contained within the model ?) and the worker finish time updated
        :param i: int, index of the worker to perform the task
        """
        ipick = self.model.pick()
        if ipick in self.model.uu:
            self.workers[i] = (ipick, 'z')
            self.model.tz.append(ipick)
            self.model.b -= self.cz
            self.finish_time[i] += np.random.uniform(self.cz, self.cz * 2)
        else:
            self.workers[i] = (ipick, 'y')
            self.model.ty.append(ipick)
            self.model.b -= self.cy
            self.finish_time[i] += np.random.uniform(self.cy, self.cy * 2)


    def _record_experiment(self, final):
        """
        After each experiment has been run, need to figure out the worker that will finish next.
        After each experiment, the model has to update its internal records of what has been tested and how.
        It then will update the history of the screening.
        Finally the index of the worker which has now finished is returned so that more work can be assigned.
        If the final parameter is `True` then there is no need to assign further work and so jobs are killed
        :param final: Boolean, indicates if on the final loop and should return anything or not
        :return: i: int, the index of the worker which is going to finish first
        """
        i = np.argmin(self.finish_time)  # get the worker which is closest to finishing
        idone = self.workers[i][0]
        if self.workers[i][1] == 'z':
            self.model.tz.remove(idone)
            self.model.z[idone] = self.z[idone]
            self.model.uu.remove(idone)
            self.model.tu.append(idone)
            self.history.append((idone, 'z'))
        else:
            self.model.ty.remove(idone)
            self.model.y[idone] = self.y[idone]
            self.model.tu.remove(idone)
            self.model.tt.append(idone)
            self.history.append((idone, 'y'))
        if final:
            self.workers[i] = None
            self.finish_time[i] = np.inf
        else:
            return i


    def full_screen(self,ploton=False):
        """
        Performs the full automated screening with multiple workers.
        First each worker (determined by the number of threads) is assigned a material to investigate.
        After this initialisation, the screener alternates selecting and recording experiments.
        This proceeds until the budget is spent (all the while recording the history of the work).
        After the budget is spent s.t. no expensive tests can be run, the remaining jobs finish.
        :return: self.history: list, full accounting of what materials were sampled when and where
        """
        self._screener_init()  # initialise the model with a single expensive test

        for i in range(self.nthreads):  # at the start, give the workers a job to do each
            self._select_and_run_experiment(i)

        while self.model.b >= self.cy:  # spend budget till cant afford any more expensive tests
            i = self._record_experiment(final=False)
            self._select_and_run_experiment(i)
            if ploton:
                self.model.plot(self.model.x,self.y,self.z)

        for i in range(self.nthreads):  # finish up any remaining jobs and record their results
            self._record_experiment(final=True)

        return self.history


#########################################################################################################################
## dummy main script
#
#x = np.random.randn(10, 10)
#y = np.random.randn(10)
#z = np.random.randn(10)
#cz, cy = 0.1, 1.0
#B = 500
#problem = '?'
#acquisition_function = 'Thompson'
#
#
#P = two_test_toy_AME.toy_prospector(x, cz, cy, B, problem, acquisition_function)  # init prospector
#
#nthreads = 5
#big_screen = ParallelScreener(P, y, z, cz, cy, nthreads)
#screening_history = big_screen.full_screen()