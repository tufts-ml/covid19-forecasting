import json

import numpy as np
import tensorflow_probability as tfp
import tensorflow as tf

from model import Comp, Vax


no = Vax.no.value
yes = Vax.yes.value

def replace_keys(old_dict, type, from_tensor=False):
    new_dict = { }
    for key in old_dict.keys():
        if isinstance(old_dict[key], dict):
            if key in ["0", "1"]:
                new_key = int(key)
            else:
                new_key = key
            new_dict[new_key] = replace_keys(old_dict[key], type, from_tensor=from_tensor)
        else:
            if isinstance(old_dict[key], str):
                if old_dict[key] in ['softplus', 'sigmoid', 'scale_100_softplus']:
                    new_dict[key] = old_dict[key]
                    continue
            if from_tensor:
                if isinstance(old_dict[key], tf.Tensor) or isinstance(old_dict[key], tf.Variable):
                    new_val = type(old_dict[key].numpy())
                else:
                    new_val = type(old_dict[key])

                new_dict[key] = new_val
            else:
                new_dict[key] = type(old_dict[key])
    return new_dict

class ModelConfig(object):

    def __init__(self):
        return

    def update_from_model(self, model):
        self.T_serial.value = {'loc': model.unconstrained_T_serial['loc'],
                                'scale': model.unconstrained_T_serial['scale']}
        self.epsilon.value = {'loc': model.unconstrained_epsilon['loc'],
                               'scale': model.unconstrained_epsilon['scale']}
        self.delta.value = {'loc': model.unconstrained_delta['loc'],
                               'scale': model.unconstrained_delta['scale']}

        self.rho_M.value = {0: {'loc': model.unconstrained_rho_M[0]['loc'],
                               'scale': model.unconstrained_rho_M[0]['scale']}}
        self.eff_M.value = {1: {'loc': model.unconstrained_eff_M[1]['loc'],
                                'scale': model.unconstrained_eff_M[1]['scale']}}
        self.lambda_M.value = {0: {'loc': model.unconstrained_lambda_M[0]['loc'],
                               'scale': model.unconstrained_lambda_M[0]['scale']},
                            1: {'loc': model.unconstrained_lambda_M[1]['loc'],
                               'scale': model.unconstrained_lambda_M[1]['scale']}}
        self.nu_M.value = {0: {'loc': model.unconstrained_nu_M[0]['loc'],
                               'scale': model.unconstrained_nu_M[0]['scale']},
                            1: {'loc': model.unconstrained_nu_M[1]['loc'],
                               'scale': model.unconstrained_nu_M[1]['scale']}}
        self.rho_G.value = {0: {'loc': model.unconstrained_rho_G[0]['loc'],
                                'scale': model.unconstrained_rho_G[0]['scale']}}
        self.eff_G.value = {1: {'loc': model.unconstrained_eff_G[1]['loc'],
                                'scale': model.unconstrained_eff_G[1]['scale']}}
        self.lambda_G.value = {0: {'loc': model.unconstrained_lambda_G[0]['loc'],
                                   'scale': model.unconstrained_lambda_G[0]['scale']},
                               1: {'loc': model.unconstrained_lambda_G[1]['loc'],
                                   'scale': model.unconstrained_lambda_G[1]['scale']}}
        self.nu_G.value = {0: {'loc': model.unconstrained_nu_G[0]['loc'],
                               'scale': model.unconstrained_nu_G[0]['scale']},
                           1: {'loc': model.unconstrained_nu_G[1]['loc'],
                               'scale': model.unconstrained_nu_G[1]['scale']}}
        self.rho_I.value = {0: {'loc': model.unconstrained_rho_I[0]['loc'],
                                'scale': model.unconstrained_rho_I[0]['scale']}}
        self.eff_I.value = {1: {'loc': model.unconstrained_eff_I[1]['loc'],
                                'scale': model.unconstrained_eff_I[1]['scale']}}
        self.lambda_I.value = {0: {'loc': model.unconstrained_lambda_I[0]['loc'],
                                   'scale': model.unconstrained_lambda_I[0]['scale']},
                               1: {'loc': model.unconstrained_lambda_I[1]['loc'],
                                   'scale': model.unconstrained_lambda_I[1]['scale']}}
        self.lambda_I_bar.value = {0: {'loc': model.unconstrained_lambda_I_bar[0]['loc'],
                                   'scale': model.unconstrained_lambda_I_bar[0]['scale']},
                               1: {'loc': model.unconstrained_lambda_I_bar[1]['loc'],
                                   'scale': model.unconstrained_lambda_I_bar[1]['scale']}}
        self.nu_I.value = {0: {'loc': model.unconstrained_nu_I[0]['loc'],
                               'scale': model.unconstrained_nu_I[0]['scale']},
                           1: {'loc': model.unconstrained_nu_I[1]['loc'],
                               'scale': model.unconstrained_nu_I[1]['scale']}}
        self.nu_I_bar.value = {0: {'loc': model.unconstrained_nu_I_bar[0]['loc'],
                               'scale': model.unconstrained_nu_I_bar[0]['scale']},
                           1: {'loc': model.unconstrained_nu_I_bar[1]['loc'],
                               'scale': model.unconstrained_nu_I_bar[1]['scale']}}
        self.rho_D.value = {0: {'loc': model.unconstrained_rho_D[0]['loc'],
                                'scale': model.unconstrained_rho_D[0]['scale']}}
        self.eff_D.value = {1: {'loc': model.unconstrained_eff_D[1]['loc'],
                                'scale': model.unconstrained_eff_D[1]['scale']}}
        self.lambda_D.value = {0: {'loc': model.unconstrained_lambda_D[0]['loc'],
                                   'scale': model.unconstrained_lambda_D[0]['scale']},
                               1: {'loc': model.unconstrained_lambda_D[1]['loc'],
                                   'scale': model.unconstrained_lambda_D[1]['scale']}}
        self.lambda_D_bar.value = {
            0: {'loc': model.unconstrained_lambda_D_bar[0]['loc'],
                'scale': model.unconstrained_lambda_D_bar[0]['scale']},
            1: {'loc': model.unconstrained_lambda_D_bar[1]['loc'],
                'scale': model.unconstrained_lambda_D_bar[1]['scale']}}
        self.nu_D.value = {0: {'loc': model.unconstrained_nu_D[0]['loc'],
                               'scale': model.unconstrained_nu_D[0]['scale']},
                           1: {'loc': model.unconstrained_nu_D[1]['loc'],
                               'scale': model.unconstrained_nu_D[1]['scale']}}
        self.nu_D_bar.value = {0: {'loc': model.unconstrained_nu_D_bar[0]['loc'],
                                   'scale': model.unconstrained_nu_D_bar[0]['scale']},
                               1: {'loc': model.unconstrained_nu_D_bar[1]['loc'],
                                   'scale': model.unconstrained_nu_D_bar[1]['scale']}}

        self.warmup_A.value = {0: {'slope': model.unconstrained_warmup_A_params[0]['slope'],
                               'intercept': model.unconstrained_warmup_A_params[0]['intercept'],
                               'scale': model.unconstrained_warmup_A_params[0]['scale']},
                               1: {'slope': model.unconstrained_warmup_A_params[1]['slope'],
                               'intercept': model.unconstrained_warmup_A_params[1]['intercept'],
                               'scale': model.unconstrained_warmup_A_params[1]['scale']}}
        self.warmup_M.value = {0: {'slope': model.unconstrained_warmup_M_params[0]['slope'],
                               'intercept': model.unconstrained_warmup_M_params[0]['intercept'],
                               'scale': model.unconstrained_warmup_M_params[0]['scale']},
                               1: {'slope': model.unconstrained_warmup_M_params[1]['slope'],
                               'intercept': model.unconstrained_warmup_M_params[1]['intercept'],
                               'scale': model.unconstrained_warmup_M_params[1]['scale']}}
        self.warmup_G.value = {0: {'slope': model.unconstrained_warmup_G_params[0]['slope'],
                               'intercept': model.unconstrained_warmup_G_params[0]['intercept'],
                               'scale': model.unconstrained_warmup_G_params[0]['scale']},
                               1: {'slope': model.unconstrained_warmup_G_params[1]['slope'],
                               'intercept': model.unconstrained_warmup_G_params[1]['intercept'],
                               'scale': model.unconstrained_warmup_G_params[1]['scale']}}
        self.warmup_GR.value = {0: {'slope': model.unconstrained_warmup_GR_params[0]['slope'],
                               'intercept': model.unconstrained_warmup_GR_params[0]['intercept'],
                               'scale': model.unconstrained_warmup_GR_params[0]['scale']},
                               1: {'slope': model.unconstrained_warmup_GR_params[1]['slope'],
                               'intercept': model.unconstrained_warmup_GR_params[1]['intercept'],
                               'scale': model.unconstrained_warmup_GR_params[1]['scale']}}
        self.warmup_I.value = {0: {'slope': model.unconstrained_warmup_I_params[0]['slope'],
                               'intercept': model.unconstrained_warmup_I_params[0]['intercept'],
                               'scale': model.unconstrained_warmup_I_params[0]['scale']},
                               1: {'slope': model.unconstrained_warmup_I_params[1]['slope'],
                               'intercept': model.unconstrained_warmup_I_params[1]['intercept'],
                               'scale': model.unconstrained_warmup_I_params[1]['scale']}}
        self.warmup_IR.value = {0: {'slope': model.unconstrained_warmup_IR_params[0]['slope'],
                               'intercept': model.unconstrained_warmup_IR_params[0]['intercept'],
                               'scale': model.unconstrained_warmup_IR_params[0]['scale']},
                               1: {'slope': model.unconstrained_warmup_IR_params[1]['slope'],
                               'intercept': model.unconstrained_warmup_IR_params[1]['intercept'],
                               'scale': model.unconstrained_warmup_IR_params[1]['scale']}}

        self.init_count_G.value = {0: {
            'loc': model.unconstrained_init_count_G_params[0]['loc'],
            'scale': model.unconstrained_init_count_G_params[0]['scale']},
            1: {
            'loc': model.unconstrained_init_count_G_params[1]['loc'],
            'scale': model.unconstrained_init_count_G_params[1]['scale']}}
        self.init_count_I.value = {0: {
            'loc': model.unconstrained_init_count_I_params[0]['loc'],
            'scale': model.unconstrained_init_count_I_params[0]['scale']},
            1: {
            'loc': model.unconstrained_init_count_I_params[1]['loc'],
            'scale': model.unconstrained_init_count_I_params[1]['scale']}}

        return self

    def to_json(self, filepath):

        data = {}
        data['T_serial']={}
        data['T_serial']['prior'] = self.T_serial.prior
        data['T_serial']['value'] = self.T_serial.value
        data['T_serial']['mean_transform'] = self.T_serial.mean_transform_str
        data['delta'] = {}
        data['delta']['prior'] = self.delta.prior
        data['delta']['value'] = self.delta.value
        data['delta']['mean_transform'] = self.delta.mean_transform_str
        data['epsilon'] = {}
        data['epsilon']['prior'] = self.epsilon.prior
        data['epsilon']['value'] = self.epsilon.value
        data['epsilon']['mean_transform'] = self.epsilon.mean_transform_str

        data['rho']={}
        data['rho']['mean_transform'] = self.rho_M.mean_transform_str
        data['rho']['M']={}
        data['rho']['M']['prior'] = self.rho_M.prior
        data['rho']['M']['value'] = self.rho_M.value
        data['rho']['G'] = {}
        data['rho']['G']['prior'] = self.rho_G.prior
        data['rho']['G']['value'] = self.rho_G.value
        data['rho']['I'] = {}
        data['rho']['I']['prior'] = self.rho_I.prior
        data['rho']['I']['value'] = self.rho_I.value
        data['rho']['D'] = {}
        data['rho']['D']['prior'] = self.rho_D.prior
        data['rho']['D']['value'] = self.rho_D.value

        data['eff'] = {}
        data['eff']['mean_transform'] = self.eff_M.mean_transform_str
        data['eff']['M'] = {}
        data['eff']['M']['prior'] = self.eff_M.prior
        data['eff']['M']['value'] = self.eff_M.value
        data['eff']['G'] = {}
        data['eff']['G']['prior'] = self.eff_G.prior
        data['eff']['G']['value'] = self.eff_G.value
        data['eff']['I'] = {}
        data['eff']['I']['prior'] = self.eff_I.prior
        data['eff']['I']['value'] = self.eff_I.value
        data['eff']['D'] = {}
        data['eff']['D']['prior'] = self.eff_D.prior
        data['eff']['D']['value'] = self.eff_D.value

        data['lambda'] = {}
        data['lambda']['mean_transform'] = self.lambda_M.mean_transform_str
        data['lambda']['M'] = {}
        data['lambda']['M']['prior'] = self.lambda_M.prior
        data['lambda']['M']['value'] = self.lambda_M.value
        data['lambda']['G'] = {}
        data['lambda']['G']['prior'] = self.lambda_G.prior
        data['lambda']['G']['value'] = self.lambda_G.value
        data['lambda']['I'] = {}
        data['lambda']['I']['prior'] = self.lambda_I.prior
        data['lambda']['I']['value'] = self.lambda_I.value
        data['lambda']['I_bar'] = {}
        data['lambda']['I_bar']['prior'] = self.lambda_I_bar.prior
        data['lambda']['I_bar']['value'] = self.lambda_I_bar.value
        data['lambda']['D'] = {}
        data['lambda']['D']['prior'] = self.lambda_D.prior
        data['lambda']['D']['value'] = self.lambda_D.value
        data['lambda']['D_bar'] = {}
        data['lambda']['D_bar']['prior'] = self.lambda_D_bar.prior
        data['lambda']['D_bar']['value'] = self.lambda_D_bar.value

        data['nu'] = {}
        data['nu']['mean_transform'] = self.nu_M.mean_transform_str
        data['nu']['M'] = {}
        data['nu']['M']['prior'] = self.nu_M.prior
        data['nu']['M']['value'] = self.nu_M.value
        data['nu']['M']['mean_transform'] = self.nu_M.mean_transform_str
        data['nu']['G'] = {}
        data['nu']['G']['prior'] = self.nu_G.prior
        data['nu']['G']['value'] = self.nu_G.value
        data['nu']['G']['mean_transform'] = self.nu_G.mean_transform_str
        data['nu']['I'] = {}
        data['nu']['I']['prior'] = self.nu_I.prior
        data['nu']['I']['value'] = self.nu_I.value
        data['nu']['I']['mean_transform'] = self.nu_I.mean_transform_str
        data['nu']['I_bar'] = {}
        data['nu']['I_bar']['prior'] = self.nu_I_bar.prior
        data['nu']['I_bar']['value'] = self.nu_I_bar.value
        data['nu']['I_bar']['mean_transform'] = self.nu_I_bar.mean_transform_str
        data['nu']['D'] = {}
        data['nu']['D']['prior'] = self.nu_D.prior
        data['nu']['D']['value'] = self.nu_D.value
        data['nu']['D']['mean_transform'] = self.nu_D.mean_transform_str
        data['nu']['D_bar'] = {}
        data['nu']['D_bar']['prior'] = self.nu_D_bar.prior
        data['nu']['D_bar']['value'] = self.nu_D_bar.value
        data['nu']['D_bar']['mean_transform'] = self.nu_D_bar.mean_transform_str

        data['warmup'] = {}
        data['warmup']['mean_transform'] = self.warmup_A.mean_transform_str
        data['warmup']['A'] = {}
        data['warmup']['A']['prior'] = self.warmup_A.prior
        data['warmup']['A']['value'] = self.warmup_A.value
        data['warmup']['A']['mean_transform'] = self.warmup_A.mean_transform_str
        data['warmup']['M'] = {}
        data['warmup']['M']['prior'] = self.warmup_M.prior
        data['warmup']['M']['value'] = self.warmup_M.value
        data['warmup']['M']['mean_transform'] = self.warmup_M.mean_transform_str
        data['warmup']['G'] = {}
        data['warmup']['G']['prior'] = self.warmup_G.prior
        data['warmup']['G']['value'] = self.warmup_G.value
        data['warmup']['G']['mean_transform'] = self.warmup_G.mean_transform_str
        data['warmup']['GR'] = {}
        data['warmup']['GR']['prior'] = self.warmup_GR.prior
        data['warmup']['GR']['value'] = self.warmup_GR.value
        data['warmup']['GR']['mean_transform'] = self.warmup_GR.mean_transform_str
        data['warmup']['I'] = {}
        data['warmup']['I']['prior'] = self.warmup_I.prior
        data['warmup']['I']['value'] = self.warmup_I.value
        data['warmup']['I']['mean_transform'] = self.warmup_I.mean_transform_str
        data['warmup']['IR'] = {}
        data['warmup']['IR']['prior'] = self.warmup_IR.prior
        data['warmup']['IR']['value'] = self.warmup_IR.value
        data['warmup']['IR']['mean_transform'] = self.warmup_IR.mean_transform_str

        data['init_count'] = {}
        data['init_count']['mean_transform'] = self.init_count_G.mean_transform_str
        data['init_count']['G'] = {}
        data['init_count']['G']['prior'] = self.init_count_G.prior
        data['init_count']['G']['value'] = self.init_count_G.value
        data['init_count']['G']['mean_transform'] = self.init_count_G.mean_transform_str
        data['init_count']['I'] = {}
        data['init_count']['I']['prior'] = self.init_count_I.prior
        data['init_count']['I']['value'] = self.init_count_I.value
        data['init_count']['I']['mean_transform'] = self.init_count_I.mean_transform_str

        data = replace_keys(data, str, from_tensor=True)

        with open(filepath, 'w') as json_file:
            json.dump(data, json_file, indent=4)


    @classmethod
    def from_json(cls, filepath):
        """Construct a model config from a json file

        This is the way of loading a model from disk
        """

        cnfg = cls()

        with open(filepath, 'r') as json_file:
            data = json.load(json_file)

        data = replace_keys(data, np.float32)


        cnfg.T_serial = ModelVar('T_serial', data['T_serial']['prior'], data['T_serial']['value'], data['T_serial']['mean_transform'])
        cnfg.epsilon = ModelVar('epsilon', data['epsilon']['prior'], data['epsilon']['value'], data['epsilon']['mean_transform'])
        cnfg.delta = ModelVar('delta', data['delta']['prior'], data['delta']['value'], data['delta']['mean_transform'])

        cnfg.rho_M = ModelVar('rho', data['rho']['M']['prior'], data['rho']['M']['value'], data['rho']['mean_transform'], compartment=Comp.M.value)
        cnfg.eff_M = ModelVar('eff', data['eff']['M']['prior'], data['eff']['M']['value'], data['eff']['mean_transform'], compartment=Comp.M.value)
        cnfg.lambda_M = ModelVar('lambda', data['lambda']['M']['prior'], data['lambda']['M']['value'], data['lambda']['mean_transform'], compartment=Comp.M.value)
        cnfg.nu_M = ModelVar('nu', data['nu']['M']['prior'], data['nu']['M']['value'], data['nu']['mean_transform'], compartment=Comp.M.value)

        cnfg.rho_G = ModelVar('rho', data['rho']['G']['prior'], data['rho']['G']['value'], data['rho']['mean_transform'], compartment=Comp.G.value)
        cnfg.eff_G = ModelVar('eff', data['eff']['G']['prior'], data['eff']['G']['value'], data['eff']['mean_transform'], compartment=Comp.G.value)
        cnfg.lambda_G = ModelVar('lambda', data['lambda']['G']['prior'], data['lambda']['G']['value'], data['lambda']['mean_transform'], compartment=Comp.G.value)
        cnfg.nu_G = ModelVar('nu', data['nu']['G']['prior'], data['nu']['G']['value'], data['nu']['mean_transform'], compartment=Comp.G.value)

        cnfg.rho_I = ModelVar('rho', data['rho']['I']['prior'], data['rho']['I']['value'], data['rho']['mean_transform'], compartment=Comp.I.value)
        cnfg.eff_I = ModelVar('eff', data['eff']['I']['prior'], data['eff']['I']['value'], data['eff']['mean_transform'], compartment=Comp.I.value)
        cnfg.lambda_I = ModelVar('lambda', data['lambda']['I']['prior'], data['lambda']['I']['value'], data['lambda']['mean_transform'], compartment=Comp.I.value)
        cnfg.lambda_I_bar = ModelVar('lambda', data['lambda']['I_bar']['prior'], data['lambda']['I_bar']['value'], data['lambda']['mean_transform'], compartment=Comp.GR.value)
        cnfg.nu_I = ModelVar('nu', data['nu']['I']['prior'], data['nu']['I']['value'], data['nu']['mean_transform'], compartment=Comp.I.value)
        cnfg.nu_I_bar = ModelVar('nu', data['nu']['I_bar']['prior'], data['nu']['I_bar']['value'], data['nu']['mean_transform'], compartment=Comp.GR.value)

        cnfg.rho_D = ModelVar('rho', data['rho']['D']['prior'], data['rho']['D']['value'], data['rho']['mean_transform'], compartment=Comp.D.value)
        cnfg.eff_D = ModelVar('eff', data['eff']['D']['prior'], data['eff']['D']['value'], data['eff']['mean_transform'], compartment=Comp.D.value)
        cnfg.lambda_D = ModelVar('lambda', data['lambda']['D']['prior'], data['lambda']['D']['value'], data['lambda']['mean_transform'], compartment=Comp.D.value)
        cnfg.lambda_D_bar = ModelVar('lambda', data['lambda']['D_bar']['prior'], data['lambda']['D_bar']['value'], data['lambda']['mean_transform'], compartment=Comp.IR.value)
        cnfg.nu_D = ModelVar('nu', data['nu']['D']['prior'], data['nu']['D']['value'], data['nu']['mean_transform'], compartment=Comp.D.value)
        cnfg.nu_D_bar = ModelVar('nu', data['nu']['D_bar']['prior'], data['nu']['D_bar']['value'], data['nu']['mean_transform'], compartment=Comp.IR.value)

        cnfg.warmup_A = ModelVar('warmup', data['warmup']['A']['prior'], data['warmup']['A']['value'],
                                 data['warmup']['mean_transform'],
                                 compartment=Comp.A.value)
        cnfg.warmup_M = ModelVar('warmup', data['warmup']['M']['prior'], data['warmup']['M']['value'],
                                 data['warmup']['mean_transform'],
                                 compartment=Comp.M.value)
        cnfg.warmup_G = ModelVar('warmup', data['warmup']['G']['prior'], data['warmup']['G']['value'],
                                 data['warmup']['mean_transform'],
                                 compartment=Comp.G.value)
        cnfg.warmup_GR = ModelVar('warmup', data['warmup']['GR']['prior'], data['warmup']['GR']['value'],
                                 data['warmup']['mean_transform'],
                                 compartment=Comp.GR.value)
        cnfg.warmup_I = ModelVar('warmup', data['warmup']['I']['prior'], data['warmup']['I']['value'],
                                 data['warmup']['mean_transform'],
                                 compartment=Comp.I.value)
        cnfg.warmup_IR = ModelVar('warmup', data['warmup']['IR']['prior'], data['warmup']['IR']['value'],
                                  data['warmup']['mean_transform'],
                                  compartment=Comp.IR.value)

        cnfg.init_count_G = ModelVar('init_count', data['init_count']['G']['prior'], data['init_count']['G']['value'],
                                     data['init_count']['mean_transform'],
                                     compartment=Comp.G.value)
        cnfg.init_count_I = ModelVar('init_count', data['init_count']['I']['prior'], data['init_count']['I']['value'],
                                     data['init_count']['mean_transform'],
                                     compartment=Comp.I.value)

        return cnfg


class ModelVar(object):

    def __init__(self, name, prior, value, mean_transform,
                 scale_transform=tfp.bijectors.Softplus(), compartment =None):
        if compartment is None:
            self.name = name
        else:
            self.name = '_'.join([name, str(compartment)])
        self.compartment = compartment

        self.prior = prior
        self.value = value
        self.mean_transform_str = mean_transform
        self.mean_transform = self._parse_transform(mean_transform)
        return

    @classmethod
    def _parse_transform(cls, str_transform):
        """Translate a string from the config into a tesnroflow probability bijector"""

        if str_transform == 'softplus':
            transform = tfp.bijectors.Softplus()
        elif str_transform == 'sigmoid':
            transform = tfp.bijectors.Sigmoid()
        elif str_transform == 'scale_100_softplus':
            transform = tfp.bijectors.Chain([tfp.bijectors.Scale(100), tfp.bijectors.Softplus()])
        else:
            raise ValueError(f'Transform {str_transform} is not one of the available transforms')

        return transform
