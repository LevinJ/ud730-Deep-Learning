
from optparse import OptionParser
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

class ExtractDataFromTraces:
    def __init__(self):
        self.options = self.parse_arguments() 
        
        return
    def run(self):
        self.processFile()
        return
    def addMoreOption(self, parser):
        pass
    def parse_arguments(self):
        parser = OptionParser()
        parser.description = "extract data"
    
        parser.add_option("-i", "--input", dest="input_path",
                           metavar="FILE", type="string",
                           help="path to the system resource log file")
        self.addMoreOption(parser)                                            
        options, _ = parser.parse_args()
    
        if options.input_path:
            if not os.path.exists(options.input_path):
                parser.error("Could not find the input file")
        else:
            parser.error("'input' option is required to run this program")

        return options 
    def processFile(self):
        with open(self.options.input_path) as f:
            for line in f:
                self.processline(line)
        self.processdata()
        return
    def processline(self, line):
        pass
    def processdata(self):
        pass
    


class TrainingChart(ExtractDataFromTraces):
    def __init__(self):
        ExtractDataFromTraces.__init__(self)
        self.steps = []
        self.losses = []
        self.minibatch_accuracy = []
        self.validation_accuracy = []
        return
    def addMoreOption(self, parser):
        parser.add_option("-s", "--start_step", dest="start_step",
                           metavar="FILE", type="int", default="0",
                           help="start_step")
        parser.add_option("-l", "--benchmark_lines", dest="benchmark_lines",
                           metavar="FILE", type="string", default='None',
                           help="benchmark_lines")
        
        return
    def processdata(self):
        tempdict ={'losses': self.losses, 'minibatch_accuracy': self.minibatch_accuracy, 'validation_accuracy': self.validation_accuracy}
        df = pd.DataFrame(tempdict, index= self.steps)
        print(df.describe())
        
#         df.plot()
#         plt.savefig(self.options.input_path + ".pdf")
        df.to_csv(self.options.input_path + ".csv")
#         plt.show()
        if 'None' in self.options.benchmark_lines:
            benmarklines = []
        else:  
            benmarklines = self.options.benchmark_lines.split(",")
            benmarklines= [float(num) for num in benmarklines]
        self.displayChart(startStep=self.options.start_step, benmarklines=benmarklines)
        return
    def displayChart(self, startStep=0, benmarklines=[]):
        selRec = np.array(self.steps) >=startStep
        steps = np.array(self.steps)[selRec]
        minibatch_accuracy = np.array(self.minibatch_accuracy)[selRec]
        validation_accuracy = np.array(self.validation_accuracy)[selRec]
        plt.plot(steps, minibatch_accuracy)
        plt.plot(steps, validation_accuracy)
#         plt.plot(steps, minibatch_accuracy,label="minibatch_accuracy")
#         plt.plot(steps, validation_accuracy,label="validation_accuracy")
        
        idlength = steps.shape[0]
        for bl in benmarklines:
            templine= np.empty(idlength)
            templine.fill(bl)
            plt.plot(steps, templine,label=str(bl))
        
        plt.legend(loc='upper right', shadow=True)
        plt.savefig(self.options.input_path + ".pdf")
        plt.show()
        return
#         print(self.steps)
    def processline(self, line):
        
        if self.process_steps(line):
            return
        if self.process_minibatch(line):
            return
        if self.process_validateset(line):
            return
        
        return
    def process_steps(self, line):
        if not line.startswith("DEBUG:root:Minibatch loss at step"):
            return False
        searchObj = re.search('DEBUG:root:Minibatch loss at step (.*)/(.*): (.*)', line, re.M|re.I)
        self.steps.append(int(searchObj.group(1)))
        self.losses.append(float(searchObj.group(3)))
        return True
    def process_minibatch(self, line):
        if not line.startswith("DEBUG:root:Minibatch accuracy"):
            return False
        searchObj = re.search('DEBUG:root:Minibatch accuracy: (.*)%', line, re.M|re.I)
        self.minibatch_accuracy.append(float(searchObj.group(1)))
        return True
    def process_validateset(self, line):
        if not line.startswith("DEBUG:root:Validation accuracy"):
            return False
        searchObj = re.search('DEBUG:root:Validation accuracy: (.*)%', line, re.M|re.I)
        self.validation_accuracy.append(float(searchObj.group(1)))
        if not (len(self.validation_accuracy) == len(self.minibatch_accuracy) and  (len(self.steps) == len(self.minibatch_accuracy))):
            raise "not matched set: " + line
        return True
    
if __name__ == "__main__":   
    obj= TrainingChart()
    obj.run()
#     line = r'DEBUG:root:Validation accuracy: 15.9%'
#     obj.process_validateset(line)
    
    