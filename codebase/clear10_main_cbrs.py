import subprocess
import os
import tempfile
import json
import argparse
import time
import numpy as np





if __name__ == "__main__":
    start_time=time.time()
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--gpu', type=int, default=0, metavar='S',help='gpu id (default: 0)') 
    parser.add_argument('--ds', type=str, default="clear10_cbrs", metavar='S',help='dataset name')
    parser.add_argument('--lr', type=float, default=0.001, metavar='S',help='batch memory ratio(default: 0.2)')
    parser.add_argument('--w_d', type=float, default=0.001, metavar='S',help='labeled ratio (default: 0.1)')
   

    args = parser.parse_args()
    auc_results = {}
    seed_list = [1,2,3,4,5]
    curr_dir = os.getcwd()
    for seed_value in seed_list:
        print("seed is",seed_value)
        fd, temp_file_name = tempfile.mkstemp() # create temporary file
        
        os.close(fd) # close the file
        proc = subprocess.Popen(["python clear10_cbrs.py --seed="+str(seed_value)+" --gpu="+str(args.gpu)+" --filename="+str(temp_file_name)],shell=True,cwd=curr_dir)
        proc.communicate()
        with open(temp_file_name) as fp:
            result = json.load(fp)
            # auc_results[str(seed_value)] = result#result[str(seed_value)]
            auc_results[str(seed_value)] = result[str(seed_value)]
        os.unlink(temp_file_name)    

    print("{:<20}  {:<20}".format('Argument','Value'))
    print("*"*80)
    for arg in vars(args):
        print("{:<20}  {:<20}".format(arg, getattr(args, arg)))
    print("*"*80)    
    print("{:<20}  {:<20}  {:<20}  {:<20}  {:<20}  {:<20}".format('seed','PR-AUC(O)', 'PR-AUC(I)', 'ROC-AUC','Total train time','Total MIR time'))
    print("*"*80)
    for key, value in auc_results.items():
        # print(key,value)
        pr_auc_o, pr_auc_1,roc_auc,tot_time,mrp_time = value
        print("{:<20}  {:<20}  {:<20}  {:<20} {:<20}  {:<20}".format(key,pr_auc_o, pr_auc_1,roc_auc,tot_time,mrp_time))
    print("-"*80)
    auc_results_values = list(auc_results.values())
    auc_average = [sum(sub_list) / len(sub_list) for sub_list in zip(*auc_results_values)]
    import pandas as pd
    df = pd.DataFrame(auc_results)
    #auc_std = np.std(sublist for sub_list in zip(*auc_results_values))
    print("{:<20}  {:<20}  {:<20}  {:<20}  {:<20}  {:<20}".format('avg',float(str(auc_average[0])[:5]), float (str(auc_average[1])[:5]), float(str(auc_average[2])[:5]),float(str(auc_average[3])[:5]),float(str(auc_average[4])[:5])))
    print(df.std(axis=1))
    print("-"*80)
    total_time = time.time()-start_time
    print("total execution time is %.3f seconds" % (total_time))
    print("avg execution time %.3f seconds"%(total_time/len(seed_list)))
