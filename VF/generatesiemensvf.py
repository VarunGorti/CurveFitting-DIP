import os
import vectorfit
import skrf as rf
import shutil

class Case:
    def __init__(self):
        self.truth = None
        self.sampled = []
    
    def __bool__(self):
        return bool(self.truth) and bool(self.sampled)
    
    def __repr__(self):
        return self.truth + '\n    ' + "\n    ".join(self.sampled)

def _get_cases():
    cases = []
    for directory in os.listdir('.'):
        if not os.path.isdir(directory):
            continue
        
        case = Case()
        for file in os.listdir(directory):
            pth = directory + '/' + file
            if not os.path.isfile(pth):
                continue
            if "HLAS" in file:
                continue
            
            base, ext = os.path.splitext(file)
            if ext[1] != 's' or ext[-1] != 'p':
                continue
            
            if base == directory:
                case.truth = pth
            else:
                case.sampled.append(pth)
        if case:
            cases.append(case)
    return cases


path_to_hlasconsole = "C:/Users/jpingeno/Dev/common1/bin/SDD_HOME/Nimbic/Release"
for case in _get_cases():
    ground_truth = rf.Network(case.truth)
    if ground_truth.f[0] == 0:
        ground_truth.resample(ground_truth.frequency[1:])
    full_sweep = ground_truth.f
    
    print("\n\n" + "#"*50 + "\n## " + case.truth)
    for sampled_file in case.sampled:
        base, ext = os.path.splitext(sampled_file)
        local_name = "sampled" + ext
        shutil.copyfile(sampled_file, local_name)
        vf_result = base + ".HLAS" + ext
        print(f"\n\n    {sampled_file} -> {vf_result}")
        vfit = vectorfit.SiemensVectorFit.vector_fit(local_name, full_sweep, path_to_hlasconsole)
        vfit.write_touchstone(vf_result)
    