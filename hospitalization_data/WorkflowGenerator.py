from HospitalData_v20210120 import HospitalData
import os
from pathlib import Path
from shutil import copyfile

import time
import datetime

class WorkflowGenerator(object):
    ''' Represents a workflow generator to make contents of covid19-forecasting/workflows/to_be_generated_workflow_folder

    Attributes
    ----------
    params_generator : ParamsGenerator
        ParamsGenerator obj 
    '''

    def __init__(self, params_generator):
        ''' Construct a HospitalData from provided input
        Args
        ----
        params_generator : ParamsGenerator
        ParamsGenerator obj 
        
        Returns
        -------
        Newly constructed WorkflowGenerator instance
        '''
        self.params_generator = params_generator 

        # self.generate_workflow()

    def run(self, workflow_name):
        # create to_be_generated_workflow_folder with name = workflow_name
        # create the params.json file within that folder
        # create/copy the config.json file within that folder
        # create/copy the Snakefile file within that folder
        
        # ts = time.time()
        # st = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d %H%M%S') # debugging purposes
        # workflow_name = workflow_name+st
        if not os.path.exists(workflow_name):
            os.makedirs(workflow_name)
            params_filename = self.params_generator.run(workflow_name+'/params.json') 
            # snakefile_src = str(Path(self.params_generator.template_json).parent) +'/Snakefile'
            # snakefile_dst = workflow_name+'/Snakefile'
            # copyfile(snakefile_src, snakefile_dst)

            # configfile_src = str(Path(self.params_generator.template_json).parent) +'/config.json'
            # configfile_dst = workflow_name+'/config.json'
            # copyfile(configfile_src, configfile_dst) 
        
        # returns path directory to workflow_name
        print(workflow_name, ' generated successfully')
        # return 



    



