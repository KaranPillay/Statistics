# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 15:09:12 2020

@author: HOME
"""
import numpy as np
import pandas as pd

class NaiveBayes:
    
    def __init__(self, data, inputColumns, outputColumn,continuesColumn, laplace_coff):
        self.data = data
        self.dataX = data[inputColumns]
        self.dataY = data[outputColumn]
        self.inputColumns = inputColumns
        self.outputColumn = outputColumn
        self.continuesColumn = continuesColumn
        self.laplace_coff = laplace_coff
        self.priorProb  = pd.DataFrame(self.calculatePriorProbability())
        self.colProb    = pd.DataFrame(self.calColumnProbability())
        
        
    def calculatePriorProbability(self):
        prob_dict = {}
        for inputCol in self.data.columns:
            if inputCol in self.continuesColumn:
                continue;
            inputClasses = self.data[inputCol].unique()
            class_dic = {}
            for inputClass in inputClasses:
                prob = round(len(self.data[self.data[inputCol] == inputClass]) / len(self.data[inputCol]),2)
                class_dic.update({inputClass : prob})
            prob_dict.update({inputCol : class_dic})
        return prob_dict
    
    def calColumnProbability(self):
        prob_dict = {}
        for inputCol in self.inputColumns:
            if inputCol in self.continuesColumn:
                continue;
            inputClasses = self.data[inputCol].unique()
            outputClasses = self.data[self.outputColumn].unique()
            class_dic = {}
            #print(len(inputClasses) * self.laplace_coff)
            for inputClass in inputClasses:
                for outputClass in outputClasses:
                    prob = round(((len(self.data[(self.data[inputCol] == inputClass)  
                    & (self.data[self.outputColumn] == outputClass)]) +  self.laplace_coff )
                    / (len(self.data[(self.data[self.outputColumn] == outputClass)]) + (len(inputClasses) * self.laplace_coff))),2)
                    class_dic.update({f'{inputClass} - {outputClass}' : prob})
            prob_dict.update({f'{inputCol} - {self.outputColumn}' : class_dic})
        return prob_dict
    
    def calculateColumnProb(self,column):
        value_count = self.data[column].value_counts(sort = False)
        prob = value_count /  len(self.data[column])
        return prob
    
    def calLikelyHood(self, inputColumnName, outputColumnName, laplaceCoeff):
        prob_list = {}
        inputClasses = self.data[outputColumnName].unique()
        for uniqueX in self.data[inputColumnName].unique():
            for uniqueY in self.data[outputColumnName].unique():
                #print(uniqueX,uniqueY)
                prob =  ((len( self.data[(self.data[inputColumnName] == uniqueX) & (self.data[outputColumnName] == uniqueY)])) + laplaceCoeff) / (len(self.data[self.data[outputColumnName] == uniqueY]) + (len(inputClasses) * self.laplace_coff))
                prob_list.update({f'{uniqueX} - {uniqueY}': prob})
        return prob_list
        
        
    def validateSet(self,data):
       predictedOutput = []
       for row in data.iterrows():
           outputCol =  self.outputColumn
           outputclassValues = self.data[self.outputColumn].unique()
           classes = {}
           for outputClass in outputclassValues:
               priorProb = self.priorProb.loc[outputClass , outputCol]
               #print(outputClass , outputCol,priorProb)
               sum = 0
               for inputCol in self.inputColumns:
                   if inputCol in self.continuesColumn:
                       sum +=  np.log(self.calLikelyHoodContinues(inputCol,outputClass,row[1][inputCol]))
                       #print("$$$",inputCol,np.log(self.calLikelyHoodContinues(inputCol,outputClass,row[1][inputCol])) )
                   else:
                   #print(f'{row[1][inputCol]} - {outputClass}, {inputCol} - {outputCol}',self.colProb.loc[f'{row[1][inputCol]} - {outputClass}', f'{inputCol} - {outputCol}'])
                       sum += np.log(self.colProb.loc[f'{row[1][inputCol]} - {outputClass}', f'{inputCol} - {outputCol}'])
               sum =  np.log(priorProb) +  sum
               classes.update({outputClass:sum})
           #print(classes)
           classes = {k: v for k, v in sorted(classes.items(), key=lambda item: item[1])}
           #print(classes)
           predictedOutput.append(list(classes.keys())[-1])
       return predictedOutput
   
    def calLikelyHoodContinues(self,regColumn,outputClass,value):
        x = self.data[self.data[self.outputColumn] == outputClass][regColumn]
        std  = np.std(x)
        mean = np.mean(x)
        prob = (1/np.sqrt(2* np.pi * ((std)**2))) * ( np.exp((-(value - mean)**2) / (2 * (std**2))))
        #print("$$$",regColumn,outputClass,value,"--->",prob)
        return prob
        
        
    
        