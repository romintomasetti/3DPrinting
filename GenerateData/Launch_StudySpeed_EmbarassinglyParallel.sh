#!/bin/bash
#
#SBATCH --ntasks=2
#SBATCH --time=60:00
#SBATCH --mem-per-cpu=10000


FILES="SamplingParameterSpaceTestSpeed_"

module load Python/3.6.3-foss-2017b

which python3

mpirun python3 /home/ulg/costmo/rtoma/3DPrinting/GenerateData/GenerateData_IncreasingSizewhatIsTheLimit.py $FILES$JOBID


