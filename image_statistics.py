#this class offers methods to analyse a image/array of numbers statistically
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from collections import Counter
import math


class image_stats:

    def __init__(self, array):
        self.list_nums = np.array(array)
        self.mean = 0
        self.var = 0
        self.std = 0

    def flatten_array(self):   #if list_nums is multidimention flatten it to single array
        self.list_nums = self.list_nums.flatten()
        return self.list_nums     

    def flatten_new_array(self, arr):
        self.list_nums = arr   
        return self.flatten_array()

    def general_stats(self): #calculate and return the mean, variance and standard deviation
        self.mean = np.mean(self.list_nums)
        self.var = np.var(self.list_nums)
        self.std = np.sqrt(self.var)
        return self.mean, self.var, self.std

    def get_PDF(self):
        self.x = np.linspace(min(self.list_nums), max(self.list_nums)) #create linspace for x-axis
        self.general_stats()
        self.PDF = stats.norm.pdf(self.x, self.mean, self.std) #create the PDF

    def plot_PDF(self):
        plt.plot(self.x, self.PDF, label="PDF")
        plt.show()

    def get_Zvalue(self, prob):
        return math.floor(stats.norm.ppf(prob))

    def get_rawScore(self, prob):
        return math.floor(self.mean + self.std*self.get_Zvalue(prob))

    


