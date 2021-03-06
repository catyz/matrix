import numpy as np
import argparse
import toast
from simons_array_python import sa_pipeline_filters as sa_pf
from simons_array_python import sa_pipeline_inputs as sa_pi
from simons_array_python import sa_tod as sa_tod
from simons_array_python import sa_observation as sa_ob
from simons_array_python import sa_config
from simons_array_python import sa_sql
from simons_array_python import sa_pointing as sa_p
from simons_array_python import sa_timestream_operators as sa_op

from simons_array_python import sa_toast_pipeline_tools as sa_tpt

import toast.pipeline_tools as tpt
from scipy import signal

parser = argparse.ArgumentParser(
        description="SA + TOAST simulations example pipeline. Run with python3 pipeline.py @pars",
        fromfile_prefix_chars="@",
    )

sa_tpt.add_general_args(parser)
tpt.add_noise_args(parser)
tpt.add_atmosphere_args(parser)
tpt.add_sss_args(parser)
tpt.add_sky_map_args(parser)
tpt.add_pointing_args(parser)
tpt.add_mapmaker_args(parser)
tpt.add_polyfilter_args(parser)
tpt.add_groundfilter_args(parser)

args = parser.parse_args()
data_prefix = 'data'


run_id = args.run_id
subrun_id = args.subrun_id

obs = sa_ob.Observation((run_id,subrun_id)) 
#obs.detectors = ['13.10_112.90B', '13.10_112.90T'] 
obs.detectors = ['13.13_135.150T','13.13_135.150B', '13.13_155.150T','13.13_155.150B']
obs.load_metadata()

pi = sa_pi.InputLevel0CachedByObsID(
    all_detectors = obs.detectors,
    n_per_cache = 6,
    load_slowdaq = False,
    load_hwp = False,
    load_dets = True, 
    load_g3 = True,
    ignore_faulty_frame = True,
    record_frame_time = True,
    input_name = data_prefix, #For TOAST purposes, default is 'data' anyway
)

sa_operator_stack = sa_pf.OperatorComposite(
    sa_p.ComputeBoresightQuaternions(),
    sa_pf.OperatorTelescopeDataInterpolator(prefix='corrected_'),
    sa_pf.OperatorScanCorrector('raw_scan_flag', 'raw_el_pos', 'raw_antenna_time_mjd'),
    sa_pf.OperatorScanCorrector('raw_scan_flag', 'raw_az_pos', 'raw_antenna_time_mjd'),
    sa_pf.OperatorDataInitializer(pi),
)
sa_operator_stack.filter_obs(obs)

sims_prefix = 'mc'  
comm = toast.Comm()

data = sa_tpt.make_toast_data(args, comm, obs_list=[obs], sims_prefix=sims_prefix, data_prefix=data_prefix)

tpt.expand_pointing(args, comm, data)

# #print(data.obs[0]['tod'].cache.keys())
# #print(data.obs[0]['tod'].cache['pixels_13.13_155.150B'])

for mc in range(args.mc_start, args.mc_start+args.nsims):
    total_prefix = sims_prefix+str(mc) 
    print(f'Processing {total_prefix}')
    outpath = "{}/{}".format(args.outpath, mc) 
# #    tpt.simulate_atmosphere(args, comm, data, mc, total_prefix) 
# #    tpt.scale_atmosphere_by_frequency(args, comm, data, freq=None, mc=mc, cache_name=total_prefix) 
    tpt.scan_sky_signal(args, comm, data, total_prefix, mc=mc)
# #    tpt.simulate_noise(args, comm, data, mc, total_prefix)
# #    tpt.simulate_sss(args, comm, data, mc, total_prefix)  
#     #high_pass_filter(data, total_prefix)
# #     tpt.apply_polyfilter(args, comm, data, total_prefix)
# #     tpt.apply_groundfilter(args, comm, data, total_prefix)
    tpt.apply_mapmaker(args, comm, data, outpath, total_prefix, bin_only=True) 

# if rank == 0:
#     print(data.obs[0]['tod'].cache['mc0_13.13_135.150B'])
#     print(data.obs[0]['tod'].cache['mc1_13.13_135.150B'])
# #if rank == 1:
#     print(data.obs[0]['tod'].cache['mc2_13.13_135.150B'])
#     print(data.obs[0]['tod'].cache['mc3_13.13_135.150B'])

#sa_tpt.add_suffix_to_detname(data, sims_prefix, data_prefix, suffix='-I')

def high_pass_filter(data, total_prefix):            
    print('High pass filtering')
    sos = signal.butter(2, 0.1, 'hp', fs=152.58789, output='sos')
    for obs in data.obs:
        for det in obs['tod'].detectors:
            ref = obs['tod'].cache[f'{total_prefix}_{det}']
            filtered = signal.sosfilt(sos, ref)
            obs['tod'].cache[f'{total_prefix}_{det}'] = filtered
            
# import matplotlib.pyplot as plt
# fig,ax = plt.subplots(2,1, sharex=True)

# for obs in data.obs:
#     for tod in obs['tod']:
#         signal = tod.read('13.10_112.90B-I')
#         if tod._source_prefix == data_prefix+'_':
#             ax[0].plot(signal)
#         else:
#             ax[1].plot(signal)

# fig.savefig('tods.png')