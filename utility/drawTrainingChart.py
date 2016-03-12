
from optparse import OptionParser
import os
import re
import pandas as pd
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
    def parse_arguments(self):
        parser = OptionParser()
        parser.description = "extract data"
    
        parser.add_option("-i", "--input", dest="input_path",
                           metavar="FILE", type="string",
                           help="path to the system resource log file")
                                                      
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
    def processdata(self):
        tempdict ={'losses': self.losses, 'minibatch_accuracy': self.minibatch_accuracy, 'validation_accuracy': self.validation_accuracy}
        df = pd.DataFrame(tempdict, index= self.steps)
        print(df.describe())
        
        df.plot()
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
    
    