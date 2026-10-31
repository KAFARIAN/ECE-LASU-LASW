# run_totalsegmentator.py
# Runs as an ISOLATED subprocess so all memory is guaranteed back to the OS
# the moment it exits — no reliance on Python/PyTorch garbage collection.
#
# Usage: python run_totalsegmentator.py <input_dicom_or_nifti_path> <output_dir>
# Prints "RESULT_OK" on success or "RESULT_FAIL: <reason>" on failure so the
# parent process (app.py) can check stdout without parsing tracebacks.

import sys
import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()

import traceback
import os
import torch

torch.set_num_threads(os.cpu_count())

def main():
    if len(sys.argv) != 3:
        print(f"RESULT_FAIL: expected 2 args (input_path, output_dir), got argv={sys.argv}")
        sys.exit(1)

    input_path, output_dir = sys.argv[1], sys.argv[2]

    try:
        from totalsegmentator.python_api import totalsegmentator

        # fast=True uses the 3mm model — much lighter on RAM/CPU than the
        # full-resolution model. roi_subset limits which structures get
        # segmented, keeping memory/time within what an 8GB machine can handle.
        totalsegmentator(
            input_path, output_dir, fast=True, fastest=True,
            nr_thr_resamp=1,
            nr_thr_saving=1,
            quiet=True,
            device="cpu",
            roi_subset=[
                "liver", "spleen", "kidney_right", "kidney_left", "heart",
                "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right",
                "lung_upper_lobe_left", "lung_lower_lobe_left","pancreas", "stomach", 
                "gallbladder", "aorta","esophagus", "trachea", "urinary_bladder", "colon"
            ]
        )

        print("RESULT_OK")
        sys.exit(0)

    except Exception as e:
        print(f"RESULT_FAIL: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()