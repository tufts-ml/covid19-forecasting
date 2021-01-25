from HospitalData_v20210120 import HospitalData
import json
class ParamsGenerator(object):
    ''' Represents a params.json generator 

    Attributes
    ----------
    template_json : str
            filepath to params.json that has the most up-to-date best fitted params, most of these params will be copied over to the newly genearted params.json 
    finalized_json : str
            filepath to params.json that is newly generated
    supplement_obj : HospitalData Obj or CovidEstim Obj 
        Obj has attr filtered_data SFrame which can yield relevant info for the params.json generation
    specified_params : list_of_param_names
            list of parameter names that are extracted from supplement_obj
    '''

    def __init__(self, template_json, supplement_obj, specified_params=[]):
        ''' Construct a HospitalData from provided input
        Args
        ----
        template_json : str
            filepath to params.json that has the most up-to-date best fitted params, most of these params will be copied over to the newly genearted params.json 
        supplement_obj : HospitalData Obj or CovidEstim Obj 
            Obj has attr filtered_data SFrame which can yield relevant info for the params.json generation
        specified_params : list_of_param_names
            list of parameter names that are extracted from supplement_obj
        
        Returns
        -------
        Newly constructed ParamsGenerator instance
        '''
        self.template_json = template_json #path to template json
        self.supplement_obj = supplement_obj #supplement_obj type
        # self.finalized_json = template_json #path to generated json
        self.specified_params= specified_params        

        # check if specified_params are all compatible with supplemental_obj's implementation/capabilities
        # self.validate_specified_params()
        # self.generate_json()


    def run(self, finalized_json):
        #generates json 
        # print('trying to tun json params generator from template ', self.template_json )
        with open(self.template_json) as json_file:
            params = json.load(json_file)
        new_params = dict()
        for k in params.keys():
            print('extracting key ', k, ' from ', self.template_json)
            if k in self.specified_params: # if a specified_params, then get params either from CovidEstim or HospitalData
                new_params[k] = self.supplement_obj.extract_param_value(k)
            else:    
                new_params[k] = params[k] # just copy over from template params.json
        
        with open(finalized_json, 'w') as outfile:
            json.dump(new_params, outfile)

        ## returns path to generated json
        print(finalized_json, ' generated successfully')
        # return self.finalized_json



    



