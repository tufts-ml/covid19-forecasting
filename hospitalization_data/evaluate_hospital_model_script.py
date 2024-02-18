from pathlib import Path
import os
from HospitalData_v20210120 import HospitalData
from ParamsGenerator import ParamsGenerator
from WorkflowGenerator import WorkflowGenerator

start_dates = ['20201001']
end_dates = ['20210109']
state_list = ['MA']

parent_dir = str(Path(__file__).resolve().parents[0])
grandparent_dir = str(Path(__file__).resolve().parents[1])
csv_filename = 'reported_hospital_utilization_timeseries_20210110_1007.csv'
template_json = grandparent_dir +'/workflows/example_match_cdc_table2/params.json'# full path to params.json of workflow

num_sims = 25

print('using params.json template at ',template_json)
for idx, (start_date, end_date) in enumerate(zip(start_dates,end_dates)):
	for us_state in state_list:
		workflow_name = us_state+'_'+str(start_date)+'_'+str(end_date)
		
		hd = HospitalData(csv_filename, us_state, start_date, end_date)
		init_params = [('init_num_'+s) for s in ["InGeneralWard", "OffVentInICU", "OnVentInICU"]]
		pg = ParamsGenerator(template_json, hd, specified_params=['num_timesteps','pmf_num_per_timestep_InGeneralWard']+init_params)
		wg = WorkflowGenerator(pg)

		wg.run(workflow_name)

		#call the main fcn of run_forecast.py with --config_file workflows --output_file --random_seed
		for sd in range(num_sims):
			config_flag = ' --config_file '+parent_dir+'/'+workflow_name+'/params.json'
			seed_flag = ' --random_seed '+str(sd)
			output_flag = ' --output_file '+parent_dir+'/'+workflow_name+'/results'+str(sd)+'.csv'
			os.system("python " + grandparent_dir+ "/run_forecast.py "+ config_flag+ output_flag+ seed_flag)
			# print(("python " + grandparent_dir+ "/run_forecast.py "+ config_flag+ output_flag+ seed_flag))


		#get the output_file and plot to compare against HospitalData and save results